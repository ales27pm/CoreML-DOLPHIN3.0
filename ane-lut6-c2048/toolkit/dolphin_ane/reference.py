"""Streamed CPU numerical references for the actual source and palettized weights.

This is a PyTorch CPU execution of the export architecture, not Core ML/ANE
execution and not a general model-quality benchmark. It evaluates next-token
distributions for fixed, tokenizer-templated prompts. At most one model part is
resident: embedding, then contiguous four-layer blocks, then the final head.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
from transformers import AutoTokenizer

from download_source import (assert_pinned_source_snapshot, verify_pinned_source_shards,
                             verify_pinned_source_snapshot)
from .model import BlockPart, EmbeddingPart, HeadPart
from .output_lock import exclusive_output_directory
from .palette import PALETTE_ALGORITHMS, palette
from .source import (SOURCE_MODEL_ID, SOURCE_REVISION, load_config, load_part,
                     make_rope_tables)


DEFAULT_FIXTURES = (
    {"id": "greeting_en", "messages": [{"role": "user", "content": "Say hello in one short sentence."}]},
    {"id": "arithmetic", "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}]},
    {"id": "greeting_fr", "messages": [{"role": "user", "content": "Dis bonjour en français, en une courte phrase."}]},
)
PALETTE_BATCH_GROUPS = 8


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _dense_hash(parameter):
    array = parameter.detach().cpu().contiguous().numpy()
    if array.dtype != np.float16:
        raise TypeError("Reference hashes require FP16 parameter bytes")
    if sys.byteorder != "little":
        array = array.astype("<f2")
    # memoryview avoids an embedding-sized Python bytes allocation.
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def _quantize_weight_groups(parameter, bits, palette_workers):
    """Fit at most eight disjoint read-only groups; update only on this thread."""
    array = parameter.detach().numpy()
    if array.shape[0] % 16:
        raise ValueError("Output channels are not divisible by group 16")
    context = ThreadPoolExecutor(max_workers=palette_workers) if palette_workers > 1 else nullcontext(None)
    with context as executor:
        for batch_start in range(0, array.shape[0], 16 * PALETTE_BATCH_GROUPS):
            starts = list(range(batch_start, min(batch_start + 16 * PALETTE_BATCH_GROUPS, array.shape[0]), 16))
            groups = [array[start:start + 16].view() for start in starts]
            for group in groups:
                group.setflags(write=False)
            if executor is None:
                results = [palette(group, bits=bits) for group in groups]
            else:
                futures = [executor.submit(palette, group, bits=bits) for group in groups]
                # Resolve the entire ordered batch before mutating parameter
                # storage. Worker inputs remain read-only and disjoint.
                results = [future.result() for future in futures]
            for start, group, (lut, indices) in zip(starts, groups, results):
                parameter[start:start + 16].copy_(torch.from_numpy(lut[indices].reshape(group.shape)))


@torch.no_grad()
def quantize_and_hash_part(part, quantize=True, min_elements=65536, bits=6, expected_source=None, palette_workers=4):
    """Mutate eligible weights to exact group-16 palette reconstructions.

    Hash names are the part's named_parameters() names. All hashes cover dense,
    little-endian FP16 bytes in C order, including intentionally padded rows.
    This matches Core ML per_grouped_channel palettization on channel_axis=0.
    Norms and weights with size <= min_elements are left unchanged, matching
    Core ML Tools 9's strict weight_threshold comparison.
    """
    if bits not in (4, 6):
        raise ValueError("Supported palette bit widths are 4 and 6")
    if type(palette_workers) is not int or palette_workers < 1:
        raise ValueError("palette_workers must be a positive integer")
    if expected_source is not None and set(expected_source) != {name for name, _ in part.named_parameters()}:
        raise ValueError("Source-reference parameter set differs from loaded model part")
    eligible = {f"{name}.weight" if name else "weight"
                for name, module in part.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Embedding))}
    result = {}
    for name, parameter in part.named_parameters():
        source_hash = _dense_hash(parameter)
        if expected_source is not None:
            expected = expected_source[name]
            if (expected.get("source_fp16_sha256") != source_hash or expected.get("shape") != list(parameter.shape)
                    or expected.get("dtype") != "float16" or expected.get("byte_order") != "little"
                    or expected.get("order") != "C"):
                raise ValueError(f"Source-reference FP16 parameter identity mismatch: {name}")
        selected = bool(quantize and name in eligible and parameter.numel() > min_elements)
        if selected:
            _quantize_weight_groups(parameter, bits, palette_workers)
        result[name] = {
            "shape": list(parameter.shape), "dtype": "float16", "byte_order": "little", "order": "C",
            "source_fp16_sha256": source_hash, "dense_reference_sha256": _dense_hash(parameter) if selected else source_hash,
            "quantized": selected, "group_size": 16 if selected else None,
            "channel_axis": 0 if selected else None, "index_bits": bits if selected else None,
        }
    return result


def _source_preflight(source_dir, expected_shards=None):
    index = source_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    if expected_shards is not None and set(shards) != set(expected_shards):
        raise ValueError("Source weight map does not reference exactly the verified pinned shards")
    details = []
    for name in shards:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Source download is incomplete: {path}")
        # safe_open validates file length/header without materializing the weights.
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            count = len(handle.keys())
        details.append({"file": name, "bytes": path.stat().st_size, "tensor_count": count})
    return details


def _fixtures(tokenizer, config, context_length, fixture_specs):
    result = []
    for spec in fixture_specs:
        ids = tokenizer.apply_chat_template(spec["messages"], tokenize=True, add_generation_prompt=True)
        ids = [int(token) for token in ids]
        if not ids or len(ids) > context_length or any(token < 0 or token >= config["vocab_size"] for token in ids):
            raise ValueError(f"Fixture {spec['id']} violates context/vocabulary constraints")
        result.append({"id": spec["id"], "messages": spec["messages"], "token_ids": ids,
                       "prompt_token_count": len(ids), "rendered_prompt": tokenizer.apply_chat_template(
                           spec["messages"], tokenize=False, add_generation_prompt=True)})
    return result


def _source_identity(source_dir, config, fixtures, source_weight_identity):
    tokenizer_files = []
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        path = source_dir / name
        if path.is_file():
            tokenizer_files.append({"file": name, "bytes": path.stat().st_size, "sha256": _file_hash(path)})
    return {"config": config, "config_canonical_sha256": _identity_hash(config),
            "config_file_sha256": _file_hash(source_dir / "config.json"),
            "fixture_identity_sha256": _identity_hash(fixtures), "tokenizer_files": tokenizer_files,
            "source_weight_identity": source_weight_identity}


def _load_source_reference(directory, config_path, identity, source_files, fixtures, context_length, part_ids):
    """Validate an immutable prior FP16 baseline; check weight hashes on load.

    Schema 1 did not record its configuration. It therefore requires an explicit
    companion export config and matching manifest. The provenance records this
    limitation rather than inventing a retrospective baseline config hash.
    """
    directory = Path(directory).resolve()
    paths = [directory / name for name in
             ("reference-report.json", "reference-weight-hashes.json", "reference-logits.npz")]
    report, hashes = (json.loads(path.read_text(encoding="utf-8")) for path in paths[:2])
    report_schema = report.get("schema_version", 1)
    hashes_schema = hashes.get("schema_version", 1)
    if (type(report_schema) is not int or type(hashes_schema) is not int
            or report_schema != hashes_schema or report_schema not in (1, 2, 3)):
        raise ValueError("Source-reference report/hash schemas must match and be one of 1, 2, or 3")
    if report.get("complete") is not True or hashes.get("complete") is not True:
        raise ValueError("Source-reference report and parameter hashes must be complete")
    if (report.get("source_model") != SOURCE_MODEL_ID or report.get("source_revision_requested") != SOURCE_REVISION
            or report.get("context_length") != context_length or report.get("query_length") != 1):
        raise ValueError("Source-reference model, revision, context, or query length differs")
    if report.get("fixtures") != fixtures:
        raise ValueError("Source-reference fixture identity differs (messages, token IDs, or rendered prompts)")
    if report.get("source_files") != source_files:
        raise ValueError("Source-reference shard inventory differs")
    if set(hashes.get("parts", {})) != set(part_ids):
        raise ValueError("Source-reference model part set differs")
    prior_identity = report.get("source_identity")
    if report_schema != 1 and not isinstance(prior_identity, dict):
        raise ValueError(f"Reference schema {report_schema} is missing its required source identity")
    if prior_identity is not None:
        if any(prior_identity.get(key) != identity[key] for key in
               ("config", "config_canonical_sha256", "fixture_identity_sha256", "tokenizer_files")):
            raise ValueError("Source-reference configuration or tokenizer identity differs")
        recorded_identity = prior_identity.get("source_weight_identity")
        hash_identity = hashes.get("source_weight_identity")
        if recorded_identity is not None or hash_identity is not None:
            if (recorded_identity != identity["source_weight_identity"]
                    or hash_identity != identity["source_weight_identity"]):
                raise ValueError("Source-reference weight-hash report has another source shard identity")
            binding = "recorded cryptographic source-input identity"
        elif report.get("schema_version", 1) >= 3 or hashes.get("schema_version", 1) >= 3:
            raise ValueError("Reference schema 3 is missing its required cryptographic source-input identity")
        else:
            binding = "schema 2 migration: freshly pinned source inputs plus every loaded FP16 parameter hash must match"
        config_provenance = {"kind": "recorded baseline configuration and tokenizer hashes",
                             "source_binding": binding,
                             "config_canonical_sha256": identity["config_canonical_sha256"]}
    else:
        if config_path is None:
            raise ValueError("Legacy source-reference schema 1 needs --source-reference-config from its companion export")
        config_path = Path(config_path).resolve()
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        companion_manifest = config_path.parent / "manifest.json"
        manifest = json.loads(companion_manifest.read_text(encoding="utf-8"))
        if (prior_config != identity["config"] or manifest.get("source_revision") != SOURCE_REVISION
                or manifest.get("context_length") != context_length):
            raise ValueError("Legacy companion export configuration or manifest identity differs")
        config_provenance = {"kind": "explicit companion export config; schema 1 baseline did not record a config hash",
                             "file": str(config_path), "sha256": _file_hash(config_path),
                             "manifest_file": str(companion_manifest), "manifest_sha256": _file_hash(companion_manifest),
                             "config_canonical_sha256": _identity_hash(prior_config)}
    with np.load(paths[2], allow_pickle=False) as archive:
        if "source_fp16" not in archive:
            raise ValueError("Source-reference has no full source_fp16 logits")
        array = archive["source_fp16"]
    if array.shape != (len(fixtures), identity["config"]["vocab_size"]) or array.dtype != np.float16 or not np.isfinite(array).all():
        raise ValueError("Source-reference full FP16 logits have invalid shape, dtype, or nonfinite values")
    summaries = report.get("results", {}).get("source_fp16", [])
    if len(summaries) != len(fixtures):
        raise ValueError("Source-reference FP16 summaries are missing")
    for case, row, summary in zip(fixtures, array, summaries):
        if (summary.get("fixture_id") != case["id"] or summary.get("finite") is not True
                or summary.get("vocabulary_size") != row.size or summary.get("greedy_token_id") != int(row.argmax())):
            raise ValueError(f"Source-reference summary differs from full logits: {case['id']}")
        for probe in summary.get("top_10", []):
            token = probe.get("token_id")
            if not isinstance(token, int) or not 0 <= token < row.size or float(row[token]) != probe.get("logit"):
                raise ValueError(f"Source-reference top logits differ: {case['id']}")
    provenance = {"directory": str(directory), "schema_version": report.get("schema_version", 1),
                  "artifacts": [{"file": path.name, "bytes": path.stat().st_size, "sha256": _file_hash(path)} for path in paths],
                  "configuration": config_provenance, "fixture_identity_sha256": _identity_hash(fixtures),
                  "source_logits_reexecuted": False, "all_source_fp16_parameter_hashes_verified": False,
                  "source_parameter_count_verified": 0,
                  "verification_scope": "Fresh pinned shard SHA256, exact config and fixture identity, plus each loaded FP16 parameter's shape and SHA256; prior source logits reused verbatim."}
    return [torch.from_numpy(row.copy()) for row in array], hashes["parts"], provenance


def numerical_quality_gate(fixtures, comparisons):
    """Narrow release sanity gate; this is not a broad quality assessment."""
    required = {item["id"]: item["messages"] for item in DEFAULT_FIXTURES}
    cases = {item["id"]: item for item in fixtures}
    required_identity = all(name in cases and cases[name].get("messages") == messages
                            for name, messages in required.items())
    measurements = comparisons or []
    complete = (len(measurements) == len(fixtures) and
                [item.get("fixture_id") for item in measurements] == [item["id"] for item in fixtures])
    all_agree = bool(complete and all(item.get("greedy_token_equal") is True for item in measurements))
    source_choices = [item["source_greedy_token_id"] for item in measurements]
    candidate_choices = [item["candidate_greedy_token_id"] for item in measurements]
    collapsed_wrong = (len(candidate_choices) >= 2 and len(set(candidate_choices)) == 1
                       and any(a != b for a, b in zip(source_choices, candidate_choices)))
    passed = bool(required_identity and complete and all_agree and not collapsed_wrong)
    return {"passed": passed, "required_fixture_ids": list(required),
            "criteria": "Complete finite FP16 logits for all three original prompts; candidate greedy next token agrees with the FP16 source for every fixture.",
            "scope": "Three fixed next-token probes only; no generated-answer, multilingual breadth, long-context, or Apple device quality claim.",
            "required_fixture_identity_verified": required_identity, "all_fixture_measurements_complete": complete,
            "fixture_count": len(fixtures), "greedy_agreement_count": sum(item.get("greedy_token_equal") is True for item in measurements),
            "all_fixture_greedy_agreement": all_agree, "degenerate_same_wrong_next_token": collapsed_wrong,
            "source_greedy_token_ids": source_choices, "candidate_greedy_token_ids": candidate_choices}


def _check_finite(tensor, context):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Nonfinite reference activations: {context}")


@torch.inference_mode()
def _embed(part, fixtures):
    sequences = []
    for fixture in fixtures:
        rows = [part(torch.tensor([[token]], dtype=torch.int32)) for token in fixture["token_ids"]]
        sequence = torch.cat(rows, dim=0).contiguous()
        _check_finite(sequence, fixture["id"] + ":embedding")
        sequences.append(sequence)
    return sequences


@torch.inference_mode()
def _blocks(part, sequences, fixtures, cos, sin, context_length):
    outputs = []
    dimension = cos.shape[-1]
    for sequence, fixture in zip(sequences, fixtures):
        part.reset_state()
        write = torch.zeros((1, 1, 1, context_length), dtype=torch.float16)
        causal = torch.full_like(write, -65504.0)
        result = torch.empty_like(sequence)
        for position in range(sequence.shape[0]):
            write.zero_()
            write[..., position] = 1.0
            causal[..., position] = 0.0
            hidden = part(sequence[position:position + 1], cos[position].reshape(1, dimension, 1, 1),
                          sin[position].reshape(1, dimension, 1, 1), write, causal)
            result[position:position + 1].copy_(hidden)
        _check_finite(result, fixture["id"] + ":block-output")
        _check_finite(part.keyCache, fixture["id"] + ":key-cache")
        _check_finite(part.valueCache, fixture["id"] + ":value-cache")
        outputs.append(result)
    return outputs


@torch.inference_mode()
def _head(part, sequences, fixtures):
    outputs = []
    for sequence, fixture in zip(sequences, fixtures):
        logits = torch.cat(part(sequence[-1:]), dim=1).flatten().contiguous()
        _check_finite(logits, fixture["id"] + ":logits")
        outputs.append(logits)
    return outputs


def _logits_summary(logits, tokenizer):
    values, ids = logits.float().topk(min(10, logits.numel()))
    winner = int(logits.argmax())
    return {"finite": True, "vocabulary_size": logits.numel(), "greedy_token_id": winner,
            "greedy_tie_break": "lowest token id",
            "greedy_token_text": tokenizer.decode([winner], skip_special_tokens=False),
            "top_10": [{"token_id": int(token), "token_text": tokenizer.decode([int(token)], skip_special_tokens=False),
                        "logit": float(value)} for token, value in zip(ids, values)]}


def _compare(source, quantized, bits=6):
    _check_finite(source, "source-logits-comparison")
    _check_finite(quantized, "candidate-logits-comparison")
    if source.shape != quantized.shape:
        raise ValueError("Source and candidate logits must have identical vocabulary shapes")
    original = source.double()
    candidate = quantized.double()
    difference = candidate - original
    original_logprob, candidate_logprob = original.log_softmax(0), candidate.log_softmax(0)
    source_choice, quantized_choice = int(original.argmax()), int(candidate.argmax())
    source_top = set(original.topk(min(10, original.numel())).indices.tolist())
    candidate_top = set(candidate.topk(min(10, candidate.numel())).indices.tolist())
    return {"greedy_token_equal": source_choice == quantized_choice,
            "source_greedy_token_id": source_choice, f"lut{bits}_greedy_token_id": quantized_choice,
            "candidate_greedy_token_id": quantized_choice,
            "logit_max_absolute_difference": float(difference.abs().max()),
            "logit_mean_absolute_difference": float(difference.abs().mean()),
            "logit_root_mean_square_difference": float(difference.square().mean().sqrt()),
            f"kl_source_to_lut{bits}_nats": float((original_logprob.exp() * (original_logprob - candidate_logprob)).sum()),
            "top_10_overlap_count": len(source_top & candidate_top),
            f"source_greedy_rank_in_lut{bits}": int((candidate > candidate[source_choice]).sum()) + 1,
            "source_greedy_probability_source": float(original_logprob[source_choice].exp()),
            f"source_greedy_probability_lut{bits}": float(candidate_logprob[source_choice].exp())}


def run_reference(source_dir, output_dir, context_length=2048, mode="both", threads=8, fixture_specs=None,
                  bits=6, source_reference=None, source_reference_config=None, palette_workers=4,
                  source_verification=None):
    """Execute references; optionally reuse a fully checked prior FP16 baseline."""
    with exclusive_output_directory(output_dir):
        return _run_reference_locked(
            source_dir, output_dir, context_length, mode, threads, fixture_specs,
            bits, source_reference, source_reference_config, palette_workers, source_verification)


def _run_reference_locked(source_dir, output_dir, context_length, mode, threads, fixture_specs,
                          bits, source_reference, source_reference_config, palette_workers,
                          source_verification):
    if mode not in {"both", "source", "quantized"}:
        raise ValueError("mode must be both, source, or quantized")
    if bits not in (4, 6):
        raise ValueError("Supported palette bit widths are 4 and 6")
    if type(palette_workers) is not int or palette_workers < 1:
        raise ValueError("palette_workers must be a positive integer")
    if source_reference is not None and mode == "source":
        raise ValueError("--source-reference cannot be combined with --source-only")
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_reference is not None and output_dir == Path(source_reference).resolve():
        raise ValueError("Candidate output must not overwrite the prior source reference")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(threads)))
    started = time.perf_counter()
    status = {"status": "running", "phase": "source_preflight", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "palette_workers": palette_workers, "palette_batch_groups": PALETTE_BATCH_GROUPS}

    def progress(phase, part_id=None, **extra):
        status.update(phase=phase, elapsed_seconds=round(time.perf_counter() - started, 3), **extra)
        if part_id is not None:
            status["part_id"] = part_id
        _write_json(output_dir / "reference-progress.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    try:
        progress("source_preflight")
        verification_path = (Path(source_verification).resolve() if source_verification is not None
                             else source_dir.parent / "source-verification.json")
        source_weight_identity, source_snapshot = verify_pinned_source_snapshot(source_dir, verification_path)
        expected_shards = {item["file"] for item in source_weight_identity["shards"]}
        source_files = _source_preflight(source_dir, expected_shards=expected_shards)
        config = load_config(source_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True, trust_remote_code=False)
        fixtures = _fixtures(tokenizer, config, context_length, fixture_specs or DEFAULT_FIXTURES)
        identity = _source_identity(source_dir, config, fixtures, source_weight_identity)
        cos, sin = make_rope_tables(config, context_length)
        include_source, include_quantized = mode != "quantized" and source_reference is None, mode != "source"
        candidate_key = f"lut{bits}_reference"
        source_hidden, quantized_hidden = None, None
        source_logits, quantized_logits = None, None
        prior_source_hashes, reuse_provenance = None, None
        hashes, timings, hidden_summaries = {}, {}, {}
        specs = [("embedding", None, None)]
        specs.extend((f"block_{index:02d}", start, min(start + 4, config["num_hidden_layers"]))
                     for index, start in enumerate(range(0, config["num_hidden_layers"], 4)))
        specs.append(("head", None, None))
        if source_reference is not None:
            progress("validate_prior_source_reference")
            source_logits, prior_source_hashes, reuse_provenance = _load_source_reference(
                source_reference, source_reference_config, identity, source_files, fixtures, context_length,
                [item[0] for item in specs])
        for part_id, start_layer, end_layer in specs:
            assert_pinned_source_snapshot(source_dir, source_snapshot, verification_path)
            part_started = time.perf_counter()
            progress("load_part", part_id)
            with torch.device("meta"):
                if part_id == "embedding":
                    part = EmbeddingPart(config)
                elif part_id == "head":
                    part = HeadPart(config)
                else:
                    part = BlockPart(config, start_layer, end_layer, context_length)
            part = load_part(part, source_dir, expected_shards=expected_shards)
            if include_source:
                progress("source_fp16_forward", part_id)
                if part_id == "embedding":
                    source_hidden = _embed(part, fixtures)
                elif part_id == "head":
                    source_logits = _head(part, source_hidden, fixtures)
                else:
                    source_hidden = _blocks(part, source_hidden, fixtures, cos, sin, context_length)
            progress("quantize_and_hash" if include_quantized else "hash_source", part_id)
            hashes[part_id] = quantize_and_hash_part(part, quantize=include_quantized, bits=bits, palette_workers=palette_workers,
                expected_source=prior_source_hashes[part_id] if prior_source_hashes is not None else None)
            if reuse_provenance is not None:
                reuse_provenance["source_parameter_count_verified"] += len(hashes[part_id])
            _write_json(output_dir / "reference-weight-hashes.json", {"schema_version": 3, "complete": False,
                "index_bits": bits if include_quantized else None, "group_size": 16, "channel_axis": 0,
                "palette_workers": palette_workers, "palette_batch_groups": PALETTE_BATCH_GROUPS,
                "source_weight_identity": source_weight_identity,
                "dense_bytes": "little-endian FP16, C order", "parts": hashes})
            if include_quantized:
                progress(candidate_key + "_forward", part_id)
                if part_id == "embedding":
                    quantized_hidden = _embed(part, fixtures)
                elif part_id == "head":
                    quantized_logits = _head(part, quantized_hidden, fixtures)
                else:
                    quantized_hidden = _blocks(part, quantized_hidden, fixtures, cos, sin, context_length)
                if part_id != "head":
                    hidden_summaries[part_id] = [{"fixture_id": fixture["id"],
                        "final_token_rms": float(sequence[-1].float().square().mean().sqrt()),
                        "final_token_max_abs": float(sequence[-1].abs().max())}
                        for fixture, sequence in zip(fixtures, quantized_hidden)]
            timings[part_id] = time.perf_counter() - part_started
            del part
            gc.collect()
            assert_pinned_source_snapshot(source_dir, source_snapshot, verification_path)
            progress("part_complete", part_id, part_seconds=round(timings[part_id], 3),
                     candidate_final_token_hidden=hidden_summaries.get(part_id))
        if verify_pinned_source_shards(source_dir, verification_path) != source_weight_identity:
            raise RuntimeError("Pinned source shards changed while the reference was running")
        reports, full_logits = {}, {}
        if source_logits is not None:
            reports["source_fp16"] = [{"fixture_id": case["id"], **_logits_summary(logits, tokenizer)}
                                      for case, logits in zip(fixtures, source_logits)]
            full_logits["source_fp16"] = torch.stack(source_logits).numpy()
        if quantized_logits is not None:
            reports[candidate_key] = [{"fixture_id": case["id"], **_logits_summary(logits, tokenizer)}
                                        for case, logits in zip(fixtures, quantized_logits)]
            full_logits[candidate_key] = torch.stack(quantized_logits).numpy()
        comparisons = ([{"fixture_id": case["id"], **_compare(original, quantized, bits=bits)}
                        for case, original, quantized in zip(fixtures, source_logits, quantized_logits)]
                       if source_logits is not None and quantized_logits is not None else None)
        gate = numerical_quality_gate(fixtures, comparisons)
        if reuse_provenance is not None:
            reuse_provenance["all_source_fp16_parameter_hashes_verified"] = True
        source_hash_identity = {part_id: {name: {key: value[key] for key in ("shape", "source_fp16_sha256")}
                               for name, value in part_hashes.items()} for part_id, part_hashes in hashes.items()}
        identity["source_fp16_parameter_identity_sha256"] = _identity_hash(source_hash_identity)
        np.savez_compressed(output_dir / "reference-logits.npz", **full_logits)
        fixture_data = {"schema_version": 3, "backend": f"PyTorch CPU eager, reconstructed LUT{bits} FP16 weights",
                        "index_bits": bits, "numerical_quality_gate_passed": gate["passed"],
                        "coreml_runtime_validation": "pending", "ane_hardware_validation": "pending",
                        "interpretation": "Fixed-prompt numerical sanity is reported separately; Apple device parity and broad model quality are unverified.",
                        "context_length": context_length, "fixtures": []}
        if quantized_logits is not None:
            for index, (case, logits) in enumerate(zip(fixtures, quantized_logits)):
                probe_ids = set(logits.float().topk(min(10, logits.numel())).indices.tolist())
                probe_ids.update((0, 1, logits.numel() - 2, logits.numel() - 1))
                probe_ids.update(np.linspace(0, logits.numel() - 1, 17, dtype=int).tolist())
                fixture_data["fixtures"].append({**case, "logits_npz_key": candidate_key, "logits_npz_row": index,
                    "greedy_token_id": int(logits.argmax()), "probe_logits": [
                        {"token_id": int(token), "logit": float(logits[token])} for token in sorted(probe_ids)]})
        _write_json(output_dir / "reference-fixtures.json", fixture_data)
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {"schema_version": 3, "complete": True, "mode": "candidate_with_reused_source" if reuse_provenance else mode,
                  "source_model": SOURCE_MODEL_ID, "source_identity": identity, "source_reference_reuse": reuse_provenance,
                  "source_revision_requested": SOURCE_REVISION, "source_files": source_files,
                  "backend": "PyTorch CPU eager", "torch_version": torch.__version__,
                  "platform": platform.platform(), "machine": platform.machine(), "threads": torch.get_num_threads(),
                  "context_length": context_length, "query_length": 1, "execution_order": "one model part across all fixture tokens before the next part",
                  "index_bits": bits if include_quantized else None,
                  "palette_workers": palette_workers, "palette_batch_groups": PALETTE_BATCH_GROUPS,
                  "compression": {"index_bits": bits if include_quantized else None, "group_size": 16,
                      "channel_axis": 0, "weight_threshold_exclusive": 65536, "algorithm": PALETTE_ALGORITHMS[bits],
                      "palette_workers": palette_workers, "palette_batch_groups": PALETTE_BATCH_GROUPS,
                      "parallelism": "Read-only independent group fits; batches of at most eight; ordered parameter.copy_ updates on the main thread"},
                  "weight_quantization": f"palette(bits={bits}), scalar LUT{1 << bits}, output-channel groups of 16; Conv2d/Embedding weights >65536 elements; norms unchanged",
                  "coreml_runtime_validation": "pending", "ane_hardware_validation": "pending",
                  "measurement_scope": "Next-token distributions after three fixed short prompts; does not establish generated-answer quality or device performance.",
                  "quality_acceptance_decision": "not_assigned_beyond_fixed_prompt_sanity",
                  "numerical_quality_gate_passed": gate["passed"], "numerical_quality_gate": gate,
                  "fixtures": fixtures, "results": reports,
                  f"source_vs_lut{bits}": comparisons, "part_seconds_including_load_hash_and_quantization": timings,
                  "candidate_final_token_hidden_by_part": hidden_summaries,
                  "total_seconds": time.perf_counter() - started,
                  "peak_process_rss_bytes": int(max_rss if platform.system() == "Darwin" else max_rss * 1024)}
        _write_json(output_dir / "reference-report.json", report)
        _write_json(output_dir / "reference-weight-hashes.json", {"schema_version": 3, "complete": True,
            "index_bits": bits if include_quantized else None, "group_size": 16, "channel_axis": 0,
            "palette_workers": palette_workers, "palette_batch_groups": PALETTE_BATCH_GROUPS,
            "source_weight_identity": source_weight_identity,
            "source_fp16_parameter_identity_sha256": identity["source_fp16_parameter_identity_sha256"],
            "dense_bytes": "little-endian FP16, C order", "parts": hashes})
        status["status"] = "completed"
        progress("complete", numerical_quality_gate_passed=gate["passed"])
        return report
    except Exception as error:
        status["status"] = "failed"
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Evidence output directory")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--bits", type=int, choices=(4, 6), default=6, help="Palette precision; default 6, legacy 4 retained for reproducing rejected evidence")
    parser.add_argument("--palette-workers", type=int, default=4, help="Independent read-only group fit threads; default 4, use 1 for serial reproduction; at most eight groups buffered")
    parser.add_argument("--source-reference", type=Path, help="Reuse this completed reference's FP16 logits after config, fixture and every source weight hash check")
    parser.add_argument("--source-reference-config", type=Path, help="For legacy schema 1 only: config.json from the prior run's companion export directory")
    parser.add_argument("--source-verification", type=Path,
                        help="Pinned source-verification.json; defaults beside the source directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quantized", action="store_true", help="Run only the candidate forward path; --source-reference supplies a prior baseline for comparison")
    group.add_argument("--source-only", action="store_true", help="Run only the uncompressed FP16 reference")
    args = parser.parse_args()
    mode = "quantized" if args.quantized else "source" if args.source_only else "both"
    report = run_reference(args.source, args.output, args.context, mode, args.threads, bits=args.bits,
                           source_reference=args.source_reference, source_reference_config=args.source_reference_config,
                           palette_workers=args.palette_workers, source_verification=args.source_verification)
    return 2 if report.get(f"source_vs_lut{args.bits}") is not None and not report["numerical_quality_gate_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
