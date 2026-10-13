#!/usr/bin/env python3
"""Export the pinned Dolphin checkpoint, one disposable process per stage."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validate_completed(model_path, report_path, kind, start, end, context, stage, bits=None):
    if not model_path.exists() and not report_path.exists():
        return False
    if not model_path.is_dir() or not report_path.is_file():
        raise RuntimeError(f"Incomplete prior output requires inspection: {model_path}")
    report = read_json(report_path)
    expected = {"source_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364", "part": kind,
                "layer_start": start, "layer_end": end, "context_length": context, "stage": stage}
    if bits is not None:
        expected["weight_format"] = f"LUT{bits} grouped-channel-16"
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"Existing output metadata mismatch: {model_path}: {key}")
    for entry in report["files"]:
        path = model_path / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise RuntimeError(f"Existing package integrity check failed: {path}")
    return True


def worker(stage, source, output, kind, start, end, context, threads, bits, palette_workers, log_path, input_path=None):
    command = [sys.executable, "-m", "dolphin_ane.exporter", stage, "--source", str(source),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "source")
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
    source, output = args.source.resolve(), args.output.resolve()
    if args.context < 1 or args.layers_per_block < 1 or args.threads < 1 or args.palette_workers < 1:
        parser.error("context, layers-per-block, threads and palette-workers must be positive")
    config = read_json(source / "config.json")
    required = {"hidden_size": 3072, "vocab_size": 128258, "num_hidden_layers": 28,
                "num_attention_heads": 24, "num_key_value_heads": 8, "tie_word_embeddings": True}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Checkpoint does not match the pinned Dolphin 3B contract")
    if args.context > config["max_position_embeddings"]:
        raise ValueError("Requested context exceeds the source checkpoint")
    for name, size in (("model-00001-of-00002.safetensors", 4965811384),
                       ("model-00002-of-00002.safetensors", 1459729952)):
        path = source / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing/incomplete source shard: {path}; run download_source.py")
    build = args.build_dir.resolve() if args.build_dir else Path(tempfile.mkdtemp(prefix="dolphin3-coreml-"))
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
        if validate_completed(final, final_report, kind, start, end, args.context, compression_stage, args.bits):
            print(json.dumps({"verified_existing": name}), flush=True)
            continue
        fp16 = build / (name + "_fp16.mlpackage")
        fp16_report = Path(str(fp16) + ".json")
        if not validate_completed(fp16, fp16_report, kind, start, end, args.context, "fp16_conversion"):
            worker("convert", source, fp16, kind, start, end, args.context, args.threads, args.bits, args.palette_workers,
                   evidence / (name + "_convert.log"))
        packed = build / (name + ".mlpackage")
        packed_report = Path(str(packed) + ".json")
        if not validate_completed(packed, packed_report, kind, start, end, args.context, compression_stage, args.bits):
            worker("compress", source, packed, kind, start, end, args.context, args.threads, args.bits, args.palette_workers,
                   evidence / (name + "_compress.log"), fp16)
        # Move only fully written, independently hashed packages into the result directory.
        shutil.move(str(packed), str(final))
        shutil.move(str(packed_report), str(final_report))
        shutil.copy2(fp16_report, evidence / (name + "_fp16.json"))
        shutil.rmtree(fp16)
        fp16_report.unlink()
        print(json.dumps({"finished": name, "bytes": sum(f["bytes"] for f in read_json(final_report)["files"])}), flush=True)
    from dolphin_ane.source import make_rope_tables
    cosine, sine = make_rope_tables(config, args.context)
    cosine.numpy().astype("<f2", copy=False).tofile(output / "rope_cos.fp16")
    sine.numpy().astype("<f2", copy=False).tofile(output / "rope_sin.fp16")
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "README.md", output / "SOURCE_MODEL_CARD.md")
    manifest = {
        "format_version": 2, "source_model": "dphn/Dolphin3.0-Llama3.2-3B",
        "source_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
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
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"export_complete": True, "model_count": len(parts), "manifest": str(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
