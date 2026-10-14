#!/usr/bin/env python3
"""Verify completed model packages against actual CPU palettized references.

Checks serialized files, protobuf interfaces/states, every compressed parameter
hash, and completeness. No Apple SDK compilation or inference is performed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
PART_IDS = ["embedding", *[f"block_{i:02d}" for i in range(7)], "head"]
BLOCK_RANGES = [[i * 4, (i + 1) * 4] for i in range(7)]
PROVISIONAL_DEVICE_TOLERANCE = 1.0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    contents = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary.write_bytes(contents)
    temporary.replace(path)
    return hashlib.sha256(contents).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Invalid SHA256: {label}")
    return value


def expected_shapes(part_id):
    if part_id == "embedding":
        return {"embed_tokens.weight": [128272, 3072]}
    if part_id == "head":
        return {f"lm_heads.{i}.weight": [16032 if i < 7 else 16048, 3072, 1, 1] for i in range(8)}
    matrices = {"self_attn.q_proj": [3072, 3072, 1, 1], "self_attn.k_proj": [1024, 3072, 1, 1],
                "self_attn.v_proj": [1024, 3072, 1, 1], "self_attn.o_proj": [3072, 3072, 1, 1],
                "mlp.gate_proj": [8192, 3072, 1, 1], "mlp.up_proj": [8192, 3072, 1, 1],
                "mlp.down_proj": [3072, 8192, 1, 1]}
    return {f"layers.{layer}.{name}.weight": shape for layer in range(4) for name, shape in matrices.items()}


def verify_package_files(package, entries):
    require(isinstance(entries, list) and entries, f"Missing file inventory: {package.name}")
    names = [entry["path"] for entry in entries]
    require(len(names) == len(set(names)), f"Duplicate file inventory entries: {package.name}")
    actual = {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}
    require(actual == set(names), f"Package inventory has missing/extra files: {package.name}")
    verified = []
    for entry in entries:
        path = (package / entry["path"]).resolve()
        require(path.is_relative_to(package.resolve()), f"Package path escapes root: {entry['path']}")
        require(path.is_file(), f"Package file missing: {path}")
        require(path.stat().st_size == entry["bytes"], f"Package byte count mismatch: {path}")
        digest = file_hash(path)
        require(digest == checked_digest(entry["sha256"], str(path)), f"Package SHA256 mismatch: {path}")
        verified.append({"path": entry["path"], "bytes": entry["bytes"], "sha256": digest})
    return verified


def inspect_serialized_spec(package):
    # Protobuf parsing does not instantiate MLModel or execute an Apple runtime.
    from coremltools.proto import FeatureTypes_pb2, Model_pb2
    package_manifest = read_json(package / "Manifest.json")
    root_item = package_manifest["itemInfoEntries"][package_manifest["rootModelIdentifier"]]
    model_path = (package / "Data" / root_item["path"]).resolve()
    require(model_path.is_relative_to(package.resolve()), "Model specification path escapes package")
    spec = Model_pb2.Model()
    spec.ParseFromString(model_path.read_bytes())
    require(spec.WhichOneof("Type") == "mlProgram", f"Package is not an ML Program: {package.name}")
    require(spec.specificationVersion == 9, f"Unexpected specification version: {package.name}")
    datatype = FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name
    inputs = {feature.name: list(feature.type.multiArrayType.shape) for feature in spec.description.input}
    outputs = {feature.name: list(feature.type.multiArrayType.shape) for feature in spec.description.output}
    states = {feature.name: list(feature.type.stateType.arrayType.shape) for feature in spec.description.state}
    input_types = {feature.name: datatype(feature.type.multiArrayType.dataType) for feature in spec.description.input}
    output_types = {feature.name: datatype(feature.type.multiArrayType.dataType) for feature in spec.description.output}
    state_types = {feature.name: datatype(feature.type.stateType.arrayType.dataType) for feature in spec.description.state}
    counts = Counter()
    for function in spec.mlProgram.functions.values():
        stack = list(function.block_specializations.values())
        while stack:
            for operation in stack.pop().operations:
                counts[operation.type] += 1
                stack.extend(operation.blocks)
    return {"inputs": inputs, "outputs": outputs, "states": states,
            "input_dtypes": input_types, "output_dtypes": output_types, "state_dtypes": state_types,
            "operation_counts": dict(counts), "metadata": dict(spec.description.metadata.userDefined)}


def reconstruct_serialized_luts(package, expected_bits=None):
    """Hash reconstructions from the saved package's actual LUT/index payloads.

    Loading the MIL program decodes stored constants without executing Core ML.
    Only one package program is live, and each weight is hashed group by group.
    No pre-save in-memory compressed model is reused at this boundary.
    """
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as load_mil
    from dolphin_ane.exporter import (dense_lut_hash, parameter_name_for_compressed_op,
                                     walk_ops)
    model = spec = program = function = operation = None
    try:
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE,
                                  skip_model_load=True)
        spec = model.get_spec()
        program = load_mil(spec, spec.specificationVersion, file_weights_dir=model.weights_dir)
        weights = []
        for function in program.functions.values():
            for operation in walk_ops(function):
                if operation.op_type == "constexpr_lut_to_dense":
                    weights.append({**dense_lut_hash(operation, expected_bits=expected_bits),
                        "parameter_name": parameter_name_for_compressed_op(operation.name)})
        return weights
    finally:
        # Loop variables retain enclosing blocks. Clear them before collection.
        operation = function = program = spec = model = None
        gc.collect()


def compression_bits(manifest, expected_bits=None):
    weights = manifest.get("weights", {})
    bits = weights.get("index_bits")
    if bits is None and manifest.get("format_version") == 1 and weights.get("format") == "LUT4":
        bits = 4  # Explicit compatibility with the original rejected artifact.
    require(type(bits) is int and bits in (4, 6), "Manifest index precision must be 4 or 6 bits")
    require(weights.get("format") == f"LUT{bits}" and weights.get("group_size") == 16 and
            weights.get("channel_axis") == 0 and weights.get("lut_dtype") == "float16", "Manifest compression format mismatch")
    require(expected_bits is None or bits == expected_bits, "Manifest precision differs from --bits assertion")
    return bits


def verify_sidecars(models, report, context, source_dir=None, source_verification=None, source_snapshot_out=None):
    """Check source metadata against pinned hashes and RoPE against its equations."""
    from download_source import assert_pinned_source_snapshot, verify_pinned_source_snapshot
    from dolphin_ane.source import make_rope_tables
    require(source_dir is not None, "A source directory is required for fresh pinned-input re-hashing")
    source_dir = Path(source_dir).resolve()
    verification_path = Path(source_verification) if source_verification else Path(__file__).parent / "source-verification.json"
    source = read_json(verification_path)
    require(source.get("all_verified") is True and source.get("revision") == REVISION and
            source.get("repo") == "dphn/Dolphin3.0-Llama3.2-3B", "Pinned source verification identity mismatch")
    source_weight_identity, source_snapshot = verify_pinned_source_snapshot(source_dir, verification_path)
    mapping = {name: name for name in ("config.json", "generation_config.json", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json")}
    mapping["SOURCE_MODEL_CARD.md"] = "README.md"
    verified = []
    for destination, original in mapping.items():
        expected = source.get("files", {}).get(original, {})
        path = models / destination
        require(expected.get("verified") is True, f"Source sidecar lacks pinned verification: {original}")
        require(path.is_file() and path.stat().st_size == expected.get("size"), f"Sidecar byte count mismatch: {destination}")
        digest = file_hash(path)
        require(digest == checked_digest(expected.get("sha256"), original), f"Sidecar SHA256 mismatch: {destination}")
        verified.append({"path": destination, "bytes": path.stat().st_size, "sha256": digest,
                         "verification_source": "pinned source file SHA256", "source_file": original})
    config = read_json(models / "config.json")
    identity = report.get("source_identity")
    if report.get("schema_version", 1) >= 2:
        require(isinstance(identity, dict) and identity.get("config") == config, "Reference and exported model config differ")
        require(identity.get("source_weight_identity") == source_weight_identity,
                "Reference source shard identity differs from the pinned verification")
        require(identity.get("config_file_sha256") == file_hash(models / "config.json"), "Reference config file hash differs")
        for record in identity.get("tokenizer_files", []):
            require(file_hash(models / record["file"]) == record["sha256"], "Reference tokenizer hash differs from exported sidecar")
    cos, sin = make_rope_tables(config, context)
    for name, tensor in (("rope_cos.fp16", cos), ("rope_sin.fp16", sin)):
        expected_bytes = tensor.numpy().astype("<f2", copy=False).tobytes(order="C")
        path = models / name
        require(path.is_file() and path.stat().st_size == context * 128 * 2, f"RoPE table byte count mismatch: {name}")
        require(path.read_bytes() == expected_bytes, f"RoPE values differ from exact reference Llama3 equations: {name}")
        verified.append({"path": name, "bytes": len(expected_bytes), "sha256": hashlib.sha256(expected_bytes).hexdigest(),
                         "verification_source": "exact Llama3 RoPE recomputation from pinned config; FP16 little-endian bytes"})
    assert_pinned_source_snapshot(source_dir, source_snapshot, verification_path)
    if source_snapshot_out is not None:
        source_snapshot_out.update(source_snapshot)
    return verified, {"file": str(verification_path.resolve()), "sha256": file_hash(verification_path),
                      "source_weight_identity": source_weight_identity}


def verify_fixtures(reference_dir, context, vocab_size, bits=4, report=None):
    from dolphin_ane.reference import numerical_quality_gate
    report = read_json(reference_dir / "reference-report.json") if report is None else report
    candidate_key = f"lut{bits}_reference"
    data = read_json(reference_dir / "reference-fixtures.json")
    require(data.get("schema_version") == report.get("schema_version"),
            "Reference fixture schema differs from the reference report schema")
    require(data.get("context_length") == context, "Reference fixture context differs from exported models")
    if data.get("schema_version", 1) >= 2:
        require(data.get("index_bits") == bits, "Reference fixture precision differs from exported models")
    cases = data.get("fixtures")
    require(isinstance(cases, list) and cases, "No actual quantized CPU reference fixtures")
    require(len({case["id"] for case in cases}) == len(cases), "Duplicate reference fixture IDs")
    canonical = report.get("fixtures", [])
    require(len(canonical) == len(cases) and all(all(case.get(key) == value for key, value in fixture.items())
            for case, fixture in zip(cases, canonical)), "Sparse fixture identities differ from reference report")
    cases_out, comparisons = [], []
    reported_comparisons = report.get(f"source_vs_lut{bits}")
    require(isinstance(reported_comparisons, list) and len(reported_comparisons) == len(cases),
            "Actual source/candidate comparisons are missing")
    with np.load(reference_dir / "reference-logits.npz", allow_pickle=False) as full:
        require(candidate_key in full and "source_fp16" in full, "Both source and candidate full logits are required")
        candidate_rows, source_rows = full[candidate_key], full["source_fp16"]
        for label, rows in (("source", source_rows), ("candidate", candidate_rows)):
            require(rows.shape == (len(cases), vocab_size) and rows.dtype == np.float16 and np.isfinite(rows).all(),
                    f"Invalid full {label} logits: expected finite FP16 [fixtures, vocabulary]")
        for index, (case, reported_comparison) in enumerate(zip(cases, reported_comparisons)):
            ids = case["token_ids"]
            require(isinstance(ids, list) and 0 < len(ids) <= context,
                    f"Fixture token count out of bounds: {case['id']}")
            require(all(isinstance(token, int) and 0 <= token < vocab_size for token in ids),
                    f"Fixture token outside vocabulary: {case['id']}")
            require(case["logits_npz_key"] == candidate_key and case["logits_npz_row"] == index,
                    "Fixture does not reference the correct candidate row and precision")
            logits = candidate_rows[index]
            require(logits.shape == (vocab_size,) and logits.dtype == np.float16 and np.isfinite(logits).all(),
                    f"Invalid full reference logits: {case['id']}")
            expected = int(np.argmax(logits))
            source_choice = int(np.argmax(source_rows[index]))
            require(case["greedy_token_id"] == expected, f"Fixture greedy token differs from actual logits: {case['id']}")
            for key, rows, choice in (("source_fp16", source_rows, source_choice), (candidate_key, candidate_rows, expected)):
                summaries = report.get("results", {}).get(key, [])
                require(len(summaries) == len(cases), f"Missing full-logit summaries: {key}")
                summary = summaries[index]
                require(summary.get("fixture_id") == case["id"] and summary.get("finite") is True and
                        summary.get("vocabulary_size") == vocab_size and summary.get("greedy_token_id") == choice,
                        f"Reference summary differs from actual logits: {key}.{case['id']}")
                for probe in summary.get("top_10", []):
                    token = probe.get("token_id")
                    require(type(token) is int and 0 <= token < vocab_size and float(rows[index, token]) == probe.get("logit"),
                            f"Reference top logits differ from saved full logits: {key}.{case['id']}")
            original, candidate = source_rows[index].astype(np.float64), logits.astype(np.float64)
            original -= original.max()
            candidate -= candidate.max()
            source_logprob = original - np.log(np.exp(original).sum())
            candidate_logprob = candidate - np.log(np.exp(candidate).sum())
            kl = float(np.sum(np.exp(source_logprob) * (source_logprob - candidate_logprob)))
            comparison = {"fixture_id": case["id"], "greedy_token_equal": expected == source_choice,
                          "source_greedy_token_id": source_choice, "candidate_greedy_token_id": expected}
            for key, value in {"fixture_id": case["id"], "greedy_token_equal": expected == source_choice,
                               "source_greedy_token_id": source_choice, f"lut{bits}_greedy_token_id": expected}.items():
                require(reported_comparison.get(key) == value, f"Reported comparison differs from actual logits: {case['id']}.{key}")
            require(math.isclose(reported_comparison.get(f"kl_source_to_lut{bits}_nats", math.nan), kl,
                                 rel_tol=1e-9, abs_tol=1e-10), f"Reported KL differs from actual logits: {case['id']}")
            comparisons.append(comparison)
            probes = case.get("probe_logits")
            require(isinstance(probes, list) and probes, f"Missing actual probe logits: {case['id']}")
            require(len({probe["token_id"] for probe in probes}) == len(probes), "Duplicate probe token IDs")
            for probe in probes:
                token = probe["token_id"]
                require(isinstance(token, int) and 0 <= token < vocab_size and
                        isinstance(probe["logit"], (int, float)) and math.isfinite(probe["logit"]),
                        f"Invalid probe: {case['id']}")
                require(float(logits[token]) == probe["logit"], f"Probe differs from saved full logits: {case['id']} token {token}")
            cases_out.append({"name": case["id"], "input_ids": ids, "expected_next_token": expected,
                "reference_logits": [{"id": probe["token_id"], "logit": probe["logit"]} for probe in probes],
                "max_abs_logit_error": PROVISIONAL_DEVICE_TOLERANCE})
    gate = numerical_quality_gate(canonical, comparisons)
    if report.get("schema_version", 1) >= 2:
        require(report.get("numerical_quality_gate_passed") is gate["passed"] and
                report.get("numerical_quality_gate") == gate and data.get("numerical_quality_gate_passed") is gate["passed"],
                "Reported numerical sanity gate differs from recomputed full-logit result")
    return {"reference_backend": data["backend"], "source_revision": REVISION, "index_bits": bits,
            "numerical_quality_gate_passed": gate["passed"], "numerical_quality_gate": gate,
            "coreml_runtime_validated": False, "ane_hardware_validated": False,
            "tolerance_policy": {"value": PROVISIONAL_DEVICE_TOLERANCE, "units": "absolute logit difference",
                "status": "provisional device acceptance tolerance; not measured or calibrated on Apple hardware"},
            "quality_acceptance_decision": "not_assigned_beyond_fixed_prompt_sanity", "cases": cases_out}


def verify_export(models, reference, output, source, bits=None, source_verification=None):
    from download_source import assert_pinned_source_snapshot
    models, reference, output = Path(models).resolve(), Path(reference).resolve(), Path(output).resolve()
    verification_path = Path(source_verification) if source_verification else Path(__file__).parent / "source-verification.json"
    result = {"schema_version": 3, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "artifact_integrity_and_weight_parity_passed": False, "compressed_weight_hash_parity_passed": False,
              "serialized_lut_reconstruction_verified": False,
              "apple_sdk_compiled": False, "coreml_runtime_executed": False,
              "ane_compute_plan_verified": False, "ane_runtime_trace_verified": False,
              "numerical_quality_gate_passed": False,
              "quality_acceptance_decision": "not_assigned_beyond_fixed_prompt_sanity", "packages": {}}
    try:
        manifest = read_json(models / "manifest.json")
        require(manifest.get("format_version") == 3,
                "Unsupported manifest format; expected format 3 with cryptographic source-input binding")
        bits = compression_bits(manifest, expected_bits=bits)
        result["index_bits"] = bits
        require(manifest.get("source_revision") == REVISION, "Manifest source revision mismatch")
        require(manifest.get("block_names") == PART_IDS[1:-1], "Manifest must list all seven default blocks in order")
        require(manifest.get("block_layer_ranges") == BLOCK_RANGES, "Manifest layers contain holes, overlaps, or unexpected block sizes")
        expected_files = [name + ".mlpackage" for name in PART_IDS]
        require(manifest.get("model_files") == expected_files, "Manifest must contain exactly the nine expected model parts")
        require({path.name for path in models.glob("*.mlpackage")} == set(expected_files), "Model package set differs from manifest")
        require(manifest.get("hidden_size") == 3072 and manifest.get("head_dim") == 128 and
                manifest.get("vocab_size") == 128258 and manifest.get("query_length") == 1 and
                manifest.get("head_sizes") == [16032] * 7 + [16034], "Manifest dimensions/vocabulary mismatch")
        require(manifest.get("states") == "float16" and manifest.get("activations") == "float16", "Manifest precision mismatch")
        context = manifest.get("context_length")
        require(isinstance(context, int) and context > 0, "Invalid manifest context")
        expected_state_bytes = 28 * 2 * 8 * 128 * context * 2
        require(manifest.get("state_bytes") == expected_state_bytes, "Manifest state byte count mismatch")
        if context == 2048:
            require(expected_state_bytes == 224 * 1024 * 1024, "2048-token cache must contain exactly 224 MiB")
        ref_hashes = read_json(reference / "reference-weight-hashes.json")
        require(ref_hashes.get("schema_version") == 3,
                "Unsupported reference weight schema; expected schema 3 after regeneration or migration")
        require(ref_hashes.get("complete") is True, "CPU reference weight hashes are incomplete")
        require(set(ref_hashes.get("parts", {})) == set(PART_IDS), "CPU reference is missing or has extra model parts")
        ref_report = read_json(reference / "reference-report.json")
        require(ref_report.get("schema_version") == 3,
                "Unsupported reference report schema; expected schema 3 after regeneration or migration")
        require(ref_report.get("complete") is True and ref_report.get("context_length") == context,
                "CPU reference report is incomplete or uses another context")
        require(ref_report.get("source_revision_requested") == REVISION, "CPU reference revision mismatch")
        require(ref_report.get("results", {}).get(f"lut{bits}_reference"), f"CPU LUT{bits} reference execution is missing")
        if ref_report.get("schema_version", 1) >= 2:
            require(ref_report.get("index_bits") == bits and ref_hashes.get("index_bits") == bits,
                    "Reference compression precision mismatch")
            for label, record in (("report", ref_report.get("compression", {})), ("weight hashes", ref_hashes)):
                require(record.get("group_size") == 16 and record.get("channel_axis") == 0,
                        f"Reference compression grouping mismatch: {label}")
            source_identity = {part_id: {name: {key: value[key] for key in ("shape", "source_fp16_sha256")}
                               for name, value in part_hashes.items()} for part_id, part_hashes in ref_hashes["parts"].items()}
            signature = hashlib.sha256(json.dumps(source_identity, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            require(ref_hashes.get("source_fp16_parameter_identity_sha256") == signature ==
                    ref_report.get("source_identity", {}).get("source_fp16_parameter_identity_sha256"),
                    "Reference source parameter identity digest mismatch")
            reuse = ref_report.get("source_reference_reuse")
            if reuse is not None:
                require(reuse.get("all_source_fp16_parameter_hashes_verified") is True and
                        reuse.get("source_parameter_count_verified") == sum(len(items) for items in ref_hashes["parts"].values()),
                        "Reused source baseline does not verify every loaded FP16 parameter")
        else:
            require(bits == 4, "Legacy reference schema 1 only described LUT4")
        swift_fixtures = verify_fixtures(reference, context, 128258, bits=bits, report=ref_report)
        result["numerical_quality_gate"] = swift_fixtures["numerical_quality_gate"]
        require(swift_fixtures["numerical_quality_gate_passed"] is True,
                "Fixed-prompt numerical sanity gate failed: every original prompt must match the source greedy next token")
        result["numerical_quality_gate_passed"] = True
        source_snapshot = {}
        result["model_sidecar_files"], result["pinned_source_verification"] = verify_sidecars(
            models, ref_report, context, source_dir=source, source_verification=source_verification,
            source_snapshot_out=source_snapshot)
        source_weight_identity = result["pinned_source_verification"]["source_weight_identity"]
        require(manifest.get("source_weight_identity") == source_weight_identity,
                "Manifest source shard identity differs from the pinned verification")
        require(ref_hashes.get("source_weight_identity") == source_weight_identity,
                "Reference weight-hash source shard identity differs from the pinned verification")
        result["source_weight_identity"] = source_weight_identity
        result["rope_tables_recomputed_and_verified"] = True
        total_weights = total_bytes = actual_state_bytes = 0
        for index, part_id in enumerate(PART_IDS):
            evidence = read_json(models / "evidence" / (part_id + ".json"))
            kind = "embedding" if part_id == "embedding" else "head" if part_id == "head" else "block"
            start, end = (0, 0) if kind == "embedding" else (28, 28) if kind == "head" else BLOCK_RANGES[index - 1]
            expected_metadata = {"stage": f"lut{bits}_compression", "part": kind, "layer_start": start,
                                 "layer_end": end, "context_length": context, "source_revision": REVISION,
                                 "source_weight_identity": source_weight_identity}
            for key, value in expected_metadata.items():
                require(evidence.get(key) == value, f"Part evidence metadata mismatch: {part_id}.{key}")
            package = models / (part_id + ".mlpackage")
            inventory = verify_package_files(package, evidence.get("files"))
            serialized = inspect_serialized_spec(package)
            for key in ("inputs", "outputs", "states", "input_dtypes", "output_dtypes", "operation_counts"):
                require(serialized[key] == evidence.get(key), f"Serialized protobuf differs from evidence: {part_id}.{key}")
            for key, value in {"source_revision": REVISION, "part": kind, "layer_start": str(start),
                               "layer_end": str(end), "context_length": str(context), "weight_format": f"LUT{bits} grouped-channel-16"}.items():
                require(serialized["metadata"].get(key) == value, f"Serialized metadata mismatch: {part_id}.{key}")
            if manifest.get("format_version", 1) >= 2:
                require(serialized["metadata"].get("weight_index_bits") == str(bits), f"Serialized bit width metadata mismatch: {part_id}")
            expected_states = ({"keyCache": [end - start, 1024, 1, context],
                                "valueCache": [end - start, 1024, 1, context]} if kind == "block" else {})
            require(serialized["states"] == expected_states, f"Incorrect state shape: {part_id}")
            require(all(value == "FLOAT16" for value in serialized["state_dtypes"].values()), "State precision must be FP16")
            state_bytes = sum(math.prod(shape) * 2 for shape in serialized["states"].values())
            shapes = expected_shapes(part_id)
            reference_weights = ref_hashes["parts"][part_id]
            all_shapes = dict(shapes)
            if kind == "head":
                all_shapes["norm.weight"] = [3072]
            elif kind == "block":
                all_shapes.update({f"layers.{layer}.{norm}.weight": [3072] for layer in range(4)
                                   for norm in ("input_layernorm", "post_attention_layernorm")})
            require(set(reference_weights) == set(all_shapes), f"Reference parameter inventory mismatch: {part_id}")
            for name, item in reference_weights.items():
                require(item.get("shape") == all_shapes[name], f"Reference parameter shape mismatch: {part_id}.{name}")
                checked_digest(item.get("source_fp16_sha256"), f"source {part_id}.{name}")
            expected_quantized = {name for name, item in reference_weights.items() if item.get("quantized") is True}
            require(expected_quantized == set(shapes), f"CPU reference quantized weight set incomplete: {part_id}")
            weights = evidence.get("weights", [])
            names = [weight["parameter_name"] for weight in weights]
            require(len(names) == len(set(names)) and set(names) == expected_quantized,
                    f"Export compressed weight set incomplete or duplicated: {part_id}")
            require(evidence.get("compression", {}).get("large_weight_tensors_compressed") == len(shapes) and
                    evidence["compression"].get("large_weight_tensors_expected") == len(shapes) and
                    serialized["operation_counts"].get("constexpr_lut_to_dense") == len(shapes),
                    f"Compressed operation count mismatch: {part_id}")
            require(all(evidence["compression"].get(key) == value for key, value in
                        {"index_bits": bits, "group_size": 16, "channel_axis": 0}.items()), "Compression format mismatch")
            reconstructed = reconstruct_serialized_luts(package, expected_bits=bits)
            reconstructed_names = [weight["parameter_name"] for weight in reconstructed]
            require(len(reconstructed_names) == len(set(reconstructed_names)) and
                    set(reconstructed_names) == expected_quantized,
                    f"Saved LUT/index payload set incomplete or duplicated: {part_id}")
            stored_by_name = {weight["parameter_name"]: weight for weight in reconstructed}
            for weight in weights:
                name = weight["parameter_name"]
                expected = reference_weights[name]
                stored = stored_by_name[name]
                for field in ("shape", "index_dtype", "lut_shape", "dense_fp16_sha256"):
                    require(stored.get(field) == weight.get(field),
                            f"Saved LUT/index reconstruction differs from exporter evidence: {part_id}.{name}.{field}")
                require(weight.get("shape") == expected.get("shape") == shapes[name], f"Compressed parameter shape mismatch: {part_id}.{name}")
                require(weight.get("index_dtype") == f"uint{bits}" and stored.get("index_bits") == bits,
                        f"Compressed index precision mismatch: {part_id}.{name}")
                require(weight.get("index_bits", 4 if bits == 4 else None) == bits and
                        expected.get("index_bits", 4 if bits == 4 else None) == bits,
                        f"Export/reference index precision metadata mismatch: {part_id}.{name}")
                expected_lut_shape = [shapes[name][0] // 16, *([1] * (len(shapes[name]) - 1)), 1 << bits, 1]
                require(stored.get("lut_shape") == expected_lut_shape,
                        f"Serialized LUT grouping or palette size mismatch: {part_id}.{name}")
                require(expected.get("dtype") == "float16" and expected.get("byte_order") == "little" and
                        expected.get("order") == "C" and expected.get("group_size") == 16 and expected.get("channel_axis") == 0,
                        f"Reference dense representation mismatch: {part_id}.{name}")
                digest = checked_digest(stored.get("dense_fp16_sha256"), part_id + "." + name)
                require(digest == checked_digest(expected.get("dense_reference_sha256"), name),
                        f"Export/reference dense weight SHA256 mismatch: {part_id}.{name}")
            total_weights += len(weights)
            actual_state_bytes += state_bytes
            package_bytes = sum(entry["bytes"] for entry in inventory)
            total_bytes += package_bytes
            result["packages"][part_id] = {"package_files_verified": inventory, "package_bytes": package_bytes,
                "serialized_protobuf_checked": True, "compressed_weights_verified": len(weights),
                "serialized_lut_reconstruction_verified": True,
                "compressed_weight_hash_parity_passed": True, "states": serialized["states"], "state_bytes": state_bytes}
        require(total_weights == 205, "Expected all 205 compressed tensors (196 block + 1 embedding + 8 head)")
        require(actual_state_bytes == expected_state_bytes, "Summed package states differ from manifest state budget")
        # The nine reconstructions can take a long time. A snapshot checked only
        # inside verify_sidecars cannot bind the inputs through this phase.
        assert_pinned_source_snapshot(source, source_snapshot, verification_path)
        fixture_path = models / "evidencefixtures.json"
        atomic_json(fixture_path, swift_fixtures)
        reference_files = []
        for path in sorted(reference.rglob("*")):
            require(not path.is_symlink(), f"Reference evidence must not contain symbolic links: {path}")
            if path.is_file():
                reference_files.append({"path": path.relative_to(reference).as_posix(),
                                        "bytes": path.stat().st_size, "sha256": file_hash(path)})
        evidence_files = []
        for path in sorted((models / "evidence").rglob("*")):
            relative = path.relative_to(models).as_posix()
            if (path.resolve() == output or relative.startswith("evidence/reference/")
                    or relative == "evidence/source-verification.json"):
                continue  # These are bound separately by the release packager.
            require(not path.is_symlink(), f"Evidence must not contain symbolic links: {path}")
            if path.is_file():
                evidence_files.append({"path": relative, "bytes": path.stat().st_size,
                                       "sha256": file_hash(path)})
        # Publish the measured narrow gate only after every package and sidecar
        # has passed. The report hashes this final manifest, not its prior state.
        manifest.setdefault("validation", {})["numerical_quality_gate_passed"] = True
        atomic_json(models / "manifest.json", manifest)
        result.update(artifact_integrity_and_weight_parity_passed=True, compressed_weight_hash_parity_passed=True,
            serialized_lut_reconstruction_verified=True,
            package_count=len(PART_IDS), compressed_tensor_count=total_weights, package_bytes=total_bytes,
            total_state_bytes=actual_state_bytes, total_state_mib=actual_state_bytes / (1024 * 1024),
            context_length=context, manifest_sha256=file_hash(models / "manifest.json"),
            reference_files=reference_files, model_evidence_files=evidence_files,
            swift_fixtures={"file": fixture_path.name, "sha256": file_hash(fixture_path),
                "case_count": len(swift_fixtures["cases"]), "max_abs_logit_error": PROVISIONAL_DEVICE_TOLERANCE,
                "tolerance_status": "provisional device acceptance tolerance; not measured or calibrated"},
            interpretation="Package integrity, exact reconstructed weight equality from saved LUT/index payloads, freshly re-hashed pinned source-shard and metadata identity, source sidecar and RoPE identity, and three fixed-prompt next-token sanity checks passed; Apple runtime, ANE placement, device parity and broad model quality remain unverified.")
        assert_pinned_source_snapshot(source, source_snapshot, verification_path)
        report_sha256 = atomic_json(output, result)
        # Return the digest of the exact serialized bytes, not a later reread.
        # Operators retain this outside the mutable export for packaging.
        return {**result, "verification_report_sha256": report_sha256}
    except Exception as error:
        result.update(artifact_integrity_and_weight_parity_passed=False,
                      compressed_weight_hash_parity_passed=False, serialized_lut_reconstruction_verified=False)
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path, help="Model output directory containing manifest.json and evidence/")
    parser.add_argument("--reference", required=True, type=Path, help="Completed CPU reference evidence directory")
    parser.add_argument("--output", required=True, type=Path, help="Verification JSON file")
    parser.add_argument("--source", required=True, type=Path,
                        help="Pinned source directory; all shards and metadata are freshly re-hashed")
    parser.add_argument("--bits", type=int, choices=(4, 6), help="Optional precision assertion; otherwise inferred and checked from manifest")
    parser.add_argument("--source-verification", type=Path, help="Pinned source-verification.json; defaults to the toolkit's recorded evidence")
    args = parser.parse_args()
    try:
        result = verify_export(args.models, args.reference, args.output, args.source, bits=args.bits,
                               source_verification=args.source_verification)
        print(json.dumps({key: result[key] for key in ("artifact_integrity_and_weight_parity_passed", "package_count",
            "compressed_tensor_count", "serialized_lut_reconstruction_verified", "numerical_quality_gate_passed", "total_state_bytes", "apple_sdk_compiled", "coreml_runtime_executed",
            "ane_compute_plan_verified", "ane_runtime_trace_verified", "verification_report_sha256")}))
    except Exception as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
