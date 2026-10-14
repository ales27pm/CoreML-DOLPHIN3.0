#!/usr/bin/env python3
"""Export the pinned Dolphin checkpoint, one disposable process per stage."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from download_source import assert_pinned_source_snapshot, verify_pinned_source_snapshot
from dolphin_ane.output_lock import exclusive_output_directory


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_stage_source_identity(report_path, source_weight_identity):
    """Require the worker itself to bind its report to the verified source."""
    report = read_json(report_path)
    recorded = report.get("source_weight_identity")
    if recorded is None:
        raise RuntimeError(f"Stage report is missing its required source identity: {report_path}")
    if recorded != source_weight_identity:
        raise RuntimeError(f"Stage report has another source identity: {report_path}")


def validate_completed(model_path, report_path, kind, start, end, context, stage,
                       source_weight_identity, bits=None):
    if model_path.is_symlink() or report_path.is_symlink():
        raise RuntimeError(f"Incomplete prior output requires inspection: {model_path}")
    if not model_path.exists() and not report_path.exists():
        return False
    if not model_path.is_dir() or not report_path.is_file():
        raise RuntimeError(f"Incomplete prior output requires inspection: {model_path}")
    report = read_json(report_path)
    expected = {"source_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364", "part": kind,
                "layer_start": start, "layer_end": end, "context_length": context, "stage": stage,
                "source_weight_identity": source_weight_identity}
    if bits is not None:
        expected["weight_format"] = f"LUT{bits} grouped-channel-16"
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"Existing output metadata mismatch: {model_path}: {key}")
    # Enumerate the actual confined tree, never paths supplied by the report.
    # An omitted, duplicate, traversal, or additional entry must fail closed.
    actual_files = []
    for path in sorted(model_path.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RuntimeError(f"Existing package integrity check failed: {path}")
        if path.is_file():
            actual_files.append({"path": str(path.relative_to(model_path)),
                                 "bytes": path.stat().st_size, "sha256": file_hash(path)})
    if not actual_files or report.get("files") != actual_files:
        raise RuntimeError(f"Existing package integrity check failed: {model_path}: exact inventory differs")
    return True


def worker(stage, source, source_verification, output, kind, start, end, context, threads, bits,
           palette_workers, log_path, input_path=None):
    command = [sys.executable, "-m", "dolphin_ane.exporter", stage, "--source", str(source),
               "--source-verification", str(source_verification),
               "--output", str(output), "--kind", kind, "--start", str(start), "--end", str(end),
               "--context", str(context), "--threads", str(threads), "--bits", str(bits),
               "--palette-workers", str(palette_workers)]
    if input_path is not None:
        command += ["--input", str(input_path)]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    print(json.dumps({"stage": stage, "part": kind, "start": start, "end": end,
                      "log": str(log_path)}), flush=True)
    with open(log_path, "w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log_path.read_text(errors="replace")[-12000:]
        print(tail, file=sys.stderr)
        raise RuntimeError(f"{stage} {kind} exited {result.returncode}; inspect {log_path}")


def build_manifest(args, parts, block_names, source_weight_identity):
    """Build the serialized runtime contract shared with importers."""
    return {
        "format_version": 3, "source_model": "dphn/Dolphin3.0-Llama3.2-3B",
        "source_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
        "source_weight_identity": source_weight_identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_ios": "18.0", "minimum_macos": "15.0", "context_length": args.context,
        "query_length": 1, "prefill": "token_by_token", "hidden_size": 3072,
        "head_dim": 128, "vocab_size": 128258, "eos_token_ids": [128256, 128001, 128008, 128009],
        "head_sizes": [16032] * 7 + [16034], "block_names": block_names,
        "block_layer_ranges": [[start, end] for _, kind, start, end in parts if kind == "block"],
        "model_files": [name + ".mlpackage" for name, _, _, _ in parts],
        "weights": {"format": f"LUT{args.bits}", "index_bits": args.bits, "group_size": 16,
                    "channel_axis": 0, "lut_dtype": "float16",
                    "palette_method": "coremltools9_weighted_kmeans1d" if args.bits == 6 else "legacy_histogram_lloyd30"},
        "activations": "float16", "states": "float16", "state_bytes": 28 * 2 * 8 * 128 * args.context * 2,
        "rope_layout": {"files": ["rope_cos.fp16", "rope_sin.fp16"], "shape": [args.context, 128], "dtype": "little_endian_float16"},
        "recommended_compute_units": "cpuAndNeuralEngine",
        "validation": {"model_specification_exported": True, "compression_inspected": True,
                       "apple_sdk_compiled": False, "coreml_runtime_executed": False,
                       "ane_compute_plan_verified": False, "ane_runtime_trace_verified": False,
                       "numerical_quality_gate_passed": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source")
    parser.add_argument("--source-verification", type=Path,
                        help="Pinned source-verification.json; defaults beside the source directory")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--layers-per-block", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bits", type=int, choices=(4, 6), default=6,
                        help="Palette index precision; 6 is the fidelity-oriented default. Legacy 4 is experimental.")
    parser.add_argument("--palette-workers", type=int, default=4,
                        help="Threads for independent palette groups; use 1 for sequential fitting.")
    args = parser.parse_args()
    if args.context < 1 or args.layers_per_block < 1 or args.threads < 1 or args.palette_workers < 1:
        parser.error("context, layers-per-block, threads and palette-workers must be positive")
    output = args.output.resolve()
    build = args.build_dir.resolve() if args.build_dir is not None else None
    if build == output:
        parser.error("--build-dir must differ from --output (including path aliases)")
    # Explicit build directories contain reusable intermediate packages and
    # must not be shared by concurrent exports, even with distinct outputs.
    directories = {output} if build is None else {output, build}
    with ExitStack() as locks:
        bound = {path: locks.enter_context(exclusive_output_directory(path)) for path in sorted(directories)}
        args.output = bound[output]
        args.build_dir = bound[build] if build is not None else None
        export(args, parser)


def export(args, parser):
    """Run with the caller's bound output and optional shared-build locks."""
    source, output = args.source.resolve(), args.output
    verification_path = (args.source_verification.resolve() if args.source_verification is not None
                         else source.parent / "source-verification.json")
    source_weight_identity, export_source_snapshot = verify_pinned_source_snapshot(source, verification_path)
    config = read_json(source / "config.json")
    assert_pinned_source_snapshot(source, export_source_snapshot, verification_path)
    required = {"hidden_size": 3072, "vocab_size": 128258, "num_hidden_layers": 28,
                "num_attention_heads": 24, "num_key_value_heads": 8, "tie_word_embeddings": True}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Checkpoint does not match the pinned Dolphin 3B contract")
    if args.context > config["max_position_embeddings"]:
        raise ValueError("Requested context exceeds the source checkpoint")
    build = args.build_dir if args.build_dir is not None else Path(tempfile.mkdtemp(prefix="dolphin3-coreml-"))
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    print(json.dumps({"output": str(output), "build_dir": str(build)}), flush=True)
    parts = [("embedding", "embedding", 0, 0)]
    block_names = []
    for index, start in enumerate(range(0, 28, args.layers_per_block)):
        name = f"block_{index:02d}"
        block_names.append(name)
        parts.append((name, "block", start, min(start + args.layers_per_block, 28)))
    parts.append(("head", "head", 28, 28))
    compression_stage = f"lut{args.bits}_compression"
    for name, kind, start, end in parts:
        final = output / (name + ".mlpackage")
        final_report = evidence / (name + ".json")
        if validate_completed(final, final_report, kind, start, end, args.context, compression_stage,
                              source_weight_identity, args.bits):
            print(json.dumps({"verified_existing": name}), flush=True)
            continue
        fp16 = build / (name + "_fp16.mlpackage")
        fp16_report = Path(str(fp16) + ".json")
        if not validate_completed(fp16, fp16_report, kind, start, end, args.context, "fp16_conversion",
                                  source_weight_identity):
            worker("convert", source, verification_path, fp16, kind, start, end, args.context, args.threads, args.bits, args.palette_workers,
                   evidence / (name + "_convert.log"))
            verify_stage_source_identity(fp16_report, source_weight_identity)
        packed = build / (name + ".mlpackage")
        packed_report = Path(str(packed) + ".json")
        if not validate_completed(packed, packed_report, kind, start, end, args.context, compression_stage,
                                  source_weight_identity, args.bits):
            worker("compress", source, verification_path, packed, kind, start, end, args.context, args.threads, args.bits, args.palette_workers,
                   evidence / (name + "_compress.log"), fp16)
            verify_stage_source_identity(packed_report, source_weight_identity)
            validate_completed(fp16, fp16_report, kind, start, end, args.context, "fp16_conversion",
                               source_weight_identity)
            validate_completed(packed, packed_report, kind, start, end, args.context, compression_stage,
                               source_weight_identity, args.bits)
        # Move only fully written, independently hashed packages into the result directory.
        shutil.move(str(packed), str(final))
        shutil.move(str(packed_report), str(final_report))
        shutil.copy2(fp16_report, evidence / (name + "_fp16.json"))
        shutil.rmtree(fp16)
        fp16_report.unlink()
        print(json.dumps({"finished": name, "bytes": sum(f["bytes"] for f in read_json(final_report)["files"])}), flush=True)
    assert_pinned_source_snapshot(source, export_source_snapshot, verification_path)
    final_source_weight_identity, sidecar_source_snapshot = verify_pinned_source_snapshot(
        source, verification_path)
    if final_source_weight_identity != source_weight_identity:
        raise RuntimeError("Pinned source shards changed while the export was running")
    from dolphin_ane.source import make_rope_tables
    cosine, sine = make_rope_tables(config, args.context)
    cosine.numpy().astype("<f2", copy=False).tofile(output / "rope_cos.fp16")
    sine.numpy().astype("<f2", copy=False).tofile(output / "rope_sin.fp16")
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "README.md", output / "SOURCE_MODEL_CARD.md")
    assert_pinned_source_snapshot(source, sidecar_source_snapshot, verification_path)
    manifest = build_manifest(args, parts, block_names, source_weight_identity)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"export_complete": True, "model_count": len(parts), "manifest": str(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
