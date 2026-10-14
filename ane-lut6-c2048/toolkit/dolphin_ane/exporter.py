"""Run one memory-bounded Core ML conversion/compression stage per process.

Linux generates the model specification and compressed weights. It cannot
compile or execute Apple's Core ML runtime, or establish physical ANE use.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
from functools import partial
import re
import resource
import time

import coremltools as ct
from coremltools.converters.mil.mil import types as mil_types
import numpy as np
import torch

from .model import EmbeddingPart, BlockPart, HeadPart
from .palette import palette, CachedPalettes, PALETTE_ALGORITHMS
from .source import load_config, load_part, SOURCE_MODEL_ID, SOURCE_REVISION
from download_source import assert_pinned_source_snapshot, verify_pinned_source_snapshot


COMPRESSION_WEIGHT_THRESHOLD = 65536


def log(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_inventory(path):
    root = Path(path)
    records = []
    for entry in sorted(root.rglob("*")):
        if entry.is_symlink():
            raise RuntimeError(f"Symlink is not permitted in model package: {entry}")
        if not entry.is_file():
            continue
        records.append({"path": str(entry.relative_to(root)), "bytes": entry.stat().st_size,
                        "sha256": sha256_file(entry)})
    return records


def validate_compression_input(package, source, verification_path):
    """Hash and bind the FP16 package which this compression phase consumes."""
    package = Path(package).resolve()
    report_path = Path(str(package) + ".json")
    identity, source_snapshot = verify_pinned_source_snapshot(source, verification_path)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("source_weight_identity") != identity:
        raise RuntimeError("FP16 input report is not bound to the pinned source identity")
    inventory = tree_inventory(package)
    if report.get("files") != inventory:
        raise RuntimeError("FP16 input package differs from its hashed report")
    return identity, inventory, report_bytes, source_snapshot


def assert_compression_input_unchanged(package, inventory, report_bytes):
    """Re-hash the exact compression input at the end of the phase.

    Core ML Tools benignly rewrites package metadata while loading a model, so
    inode/ctime snapshots produce false positives. Exact bytes and the recorded
    report remain the stable boundary and are checked both before and after the
    compression worker.
    """
    package = Path(package).resolve()
    if (tree_inventory(package) != inventory
            or Path(str(package) + ".json").read_bytes() != report_bytes):
        raise RuntimeError("FP16 input package or report changed during compression")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_model(model, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing model: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{os.getpid()}.mlpackage")
    if temporary.exists():
        raise FileExistsError(temporary)
    model.save(str(temporary))
    os.replace(temporary, output)


def create_part(config, kind, start, end, context):
    with torch.device("meta"):
        if kind == "embedding":
            return EmbeddingPart(config)
        if kind == "head":
            return HeadPart(config)
        return BlockPart(config, start, end, context)


def io_description(part, kind):
    if kind == "embedding":
        examples = (torch.zeros((1, 1), dtype=torch.int32),)
        inputs = [ct.TensorType(name="inputIds", shape=(1, 1), dtype=np.int32)]
        outputs = [ct.TensorType(name="hiddenStates", dtype=np.float16)]
        return examples, inputs, outputs, []
    hidden = (1, part.norm.hidden_size if kind == "head" else
              part.layers[0].input_layernorm.hidden_size, 1, 1)
    if kind == "head":
        return ((torch.zeros(hidden, dtype=torch.float16),),
                [ct.TensorType(name="hiddenStates", shape=hidden, dtype=np.float16)],
                [ct.TensorType(name=f"logits_{i}", dtype=np.float16)
                 for i in range(part.num_splits)], [])
    head_dim = part.layers[0].self_attn.head_dim
    examples = (
        torch.zeros(hidden, dtype=torch.float16),
        torch.ones((1, head_dim, 1, 1), dtype=torch.float16),
        torch.zeros((1, head_dim, 1, 1), dtype=torch.float16),
        torch.zeros((1, 1, 1, part.context_length), dtype=torch.float16),
        torch.full((1, 1, 1, part.context_length), -65504.0, dtype=torch.float16),
    )
    examples[3][..., 0] = 1
    examples[4][..., 0] = 0
    inputs = [ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float16)
              for name, value in zip(("hiddenStates", "ropeCos", "ropeSin",
                                      "writeMask", "causalMask"), examples)]
    # Core ML explicitly rejects a name shared between function input/output.
    outputs = [ct.TensorType(name="outputHiddenStates", dtype=np.float16)]
    states = [ct.StateType(name=name, wrapped_type=ct.TensorType(
        shape=tuple(getattr(part, name).shape), dtype=np.float16))
        for name in ("keyCache", "valueCache")]
    return examples, inputs, outputs, states


def walk_ops(block):
    for op in block.operations:
        yield op
        for nested in op.blocks:
            yield from walk_ops(nested)


def spec_report(model, kind, expected_states=None):
    spec = model.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("Output must be an ML Program")
    inputs = {f.name: list(f.type.multiArrayType.shape) for f in spec.description.input}
    outputs = {f.name: list(f.type.multiArrayType.shape) for f in spec.description.output}
    states = {f.name: list(f.type.stateType.arrayType.shape) for f in spec.description.state}
    array_type = ct.proto.FeatureTypes_pb2.ArrayFeatureType
    input_dtypes = {f.name: array_type.ArrayDataType.Name(f.type.multiArrayType.dataType)
                    for f in spec.description.input}
    output_dtypes = {f.name: array_type.ArrayDataType.Name(f.type.multiArrayType.dataType)
                     for f in spec.description.output}
    expected_input_dtype = "INT32" if kind == "embedding" else "FLOAT16"
    if any(dtype != expected_input_dtype for dtype in input_dtypes.values()):
        raise ValueError(f"Unexpected input precision: {input_dtypes}")
    if any(dtype != "FLOAT16" for dtype in output_dtypes.values()):
        raise ValueError(f"Outputs must remain FP16: {output_dtypes}")
    if set(inputs).intersection(outputs):
        raise ValueError("Core ML input/output names must be distinct")
    if kind == "block" and set(states) != {"keyCache", "valueCache"}:
        raise ValueError(f"Actual state buffers were lost: {states}")
    if kind != "block" and states:
        raise ValueError("Only transformer blocks should contain states")
    if expected_states is not None and states != expected_states:
        raise ValueError(f"Wrong exported cache shapes: {states}")
    float16_code = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
    if any(s.type.stateType.arrayType.dataType != float16_code for s in spec.description.state):
        raise ValueError("States must remain FP16")
    op_counts = Counter()
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            stack = [block]
            while stack:
                current = stack.pop()
                for op in current.operations:
                    op_counts[op.type] += 1
                    stack.extend(op.blocks)
    if kind == "block" and (op_counts["read_state"] < 2 or op_counts["write_state"] < 2):
        raise ValueError(f"Cache read/write operations missing: {dict(op_counts)}")
    return {"specification_version": spec.specificationVersion, "inputs": inputs,
            "input_dtypes": input_dtypes, "outputs": outputs,
            "output_dtypes": output_dtypes, "states": states,
            "operation_counts": dict(op_counts)}


def annotate(model, args, compressed=False):
    bits = getattr(args, "bits", 6)
    model.author = "Dolphin community checkpoint; stateful export for ales27pm"
    model.license = "Source model: Llama 3.2 Community License (see bundled source model card)"
    model.short_description = f"Dolphin 3.0 Llama 3.2 3B {args.kind}; iOS 18 stateful ANE-oriented export"
    model.version = "0.1.0"
    model.user_defined_metadata.update({
        "source_model": SOURCE_MODEL_ID, "source_revision": SOURCE_REVISION,
        "context_length": str(args.context), "query_length": "1",
        "part": args.kind, "layer_start": str(args.start), "layer_end": str(args.end),
        "weight_format": f"LUT{bits} grouped-channel-16" if compressed else "FP16",
        "weight_index_bits": str(bits) if compressed else "0",
        "activation_and_state_precision": "FP16", "minimum_os": "iOS18/macOS15",
        "ane_hardware_verified": "false", "apple_runtime_verified": "false",
        "rope": "Exact Llama3 frequency-scaled host tables; no base-theta approximation",
    })


def convert(args):
    started = time.monotonic()
    source_identity, source_snapshot = verify_pinned_source_snapshot(args.source, args.source_verification)
    config = load_config(args.source)
    part = create_part(config, args.kind, args.start, args.end, args.context)
    log("loading_weights", kind=args.kind, start=args.start, end=args.end)
    load_part(part, args.source, expected_shards={item["file"] for item in source_identity["shards"]})
    examples, inputs, outputs, states = io_description(part, args.kind)
    expected_states = {n: list(b.shape) for n, b in part.named_buffers()}
    with torch.no_grad():
        traced = torch.jit.trace(part, examples, check_trace=False)
    if args.kind == "block":
        part.reset_state()
        # TorchScript may share storage, but clear explicitly to preserve the contract.
        traced.keyCache.zero_()
        traced.valueCache.zero_()
    log("converting", parameters=sum(p.numel() for p in part.parameters()))
    model = ct.convert(
        traced, inputs=inputs, outputs=outputs, states=states,
        convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True,
    )
    annotate(model, args)
    report = spec_report(model, args.kind, expected_states)
    assert_pinned_source_snapshot(args.source, source_snapshot, args.source_verification)
    save_model(model, args.output)
    report.update({"stage": "fp16_conversion", "part": args.kind,
                   "source_model": SOURCE_MODEL_ID, "source_revision": SOURCE_REVISION,
                   "source_weight_identity": source_identity,
                   "context_length": args.context, "layer_start": args.start,
                   "layer_end": args.end, "requested_compression_bits": getattr(args, "bits", 6),
                   "seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "files": tree_inventory(args.output), "apple_runtime_verified": False,
                   "ane_hardware_verified": False})
    atomic_json(str(args.output) + ".json", report)
    log("converted", output=str(args.output), seconds=report["seconds"], peak_rss_kib=report["peak_rss_kib"])


def dense_lut_hash(op, expected_bits=None):
    """Hash actual serialized-palette reconstruction without a full dense copy."""
    indices = op.indices.val
    lut = op.lut.val
    shape = tuple(int(d) for d in indices.shape)
    if np.asarray(lut).dtype != np.dtype(np.float16):
        raise ValueError("Expected FP16 palette values")
    if len(lut.shape) != len(shape) + 2 or lut.shape[-1] != 1 or lut.shape[-2] not in (16, 64):
        raise ValueError(f"Expected scalar LUT4 or LUT6, got LUT shape {lut.shape}")
    levels = int(lut.shape[-2])
    bits = {16: 4, 64: 6}[levels]
    if expected_bits is not None and bits != expected_bits:
        raise ValueError(f"Expected LUT{expected_bits}, found LUT{bits}")
    groups = lut.shape[0]
    if shape[0] != groups * 16 or any(d != 1 for d in lut.shape[1:-2]):
        raise ValueError(f"Expected groups of 16 along axis 0: {shape}, {lut.shape}")
    # MIL's parameterized integer classes all have Python __name__ == "int".
    # Its own type formatter preserves the signedness and bit width.
    dtype_name = mil_types.builtin_to_string(op.indices.dtype)
    if dtype_name != f"uint{bits}":
        raise ValueError(f"Compression indices are {dtype_name}, not uint{bits}")
    palettes = lut.reshape(groups, levels)
    h = hashlib.sha256()
    for g in range(groups):
        index = np.asarray(indices[g * 16:(g + 1) * 16], dtype=np.uint8)
        if np.any(index >= levels):
            raise ValueError(f"A LUT{bits} index exceeds the {levels}-entry palette")
        dense = np.ascontiguousarray(palettes[g][index], dtype=np.dtype("<f2"))
        if not np.isfinite(dense).all():
            raise ValueError("Non-finite decompressed weight")
        h.update(dense.tobytes())
    return {"operation": op.name, "shape": list(shape), "index_dtype": dtype_name, "index_bits": bits,
            "lut_shape": list(lut.shape), "dense_fp16_sha256": h.hexdigest()}


def parameter_name_for_compressed_op(operation_name):
    """Recover this exporter's Torch parameter from observed ct9 const names."""
    name = operation_name.removesuffix("_palettized")
    for suffix in ("_promoted_to_fp16", "_to_fp16"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    if name == "embed_tokens_weight":
        return "embed_tokens.weight"
    match = re.fullmatch(r"lm_heads_(\d+)_weight", name)
    if match:
        return f"lm_heads.{match[1]}.weight"
    match = re.fullmatch(r"layers_(\d+)_(self_attn|mlp)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)_weight", name)
    if match:
        return f"layers.{match[1]}.{match[2]}.{match[3]}.weight"
    raise ValueError(f"Unrecognized compressed weight operation: {operation_name}")


def expected_weight_parameters(args):
    if args.kind == "embedding":
        return {"embed_tokens.weight"}
    if args.kind == "head":
        return {f"lm_heads.{i}.weight" for i in range(8)}
    suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    return {f"layers.{i}.{suffix}.weight" for i in range(args.end - args.start) for suffix in suffixes}


def precompute_palette_cache(model, args, bits, workers):
    """Read expected saved weights, fit bounded groups, release the source MIL.

    The public per-constant name configuration uses this exact inventory, so
    large state/mask constants cannot accidentally enter the weight cache.
    No coremltools implementation or worker-pool global is modified.
    """
    from coremltools.converters.mil.frontend.milproto.load import load as load_mil
    spec = program = operation = function = value = None
    selected = []
    try:
        spec = model.get_spec()
        program = load_mil(spec, spec.specificationVersion, file_weights_dir=model.weights_dir)
        expected = expected_weight_parameters(args)
        found, op_names = set(), []
        for function in program.functions.values():
            for operation in walk_ops(function):
                if operation.op_type != "const":
                    continue
                try:
                    parameter = parameter_name_for_compressed_op(operation.name)
                except ValueError:
                    continue
                if parameter not in expected:
                    continue
                value = operation.outputs[0].val
                if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float16)
                        or value.size <= COMPRESSION_WEIGHT_THRESHOLD or value.ndim < 2
                        or value.shape[0] % 16):
                    raise ValueError(f"Invalid eligible FP16 weight constant: {operation.name}")
                valid_consumer = any(
                    (child.op_type == "conv" and child.weight is operation.outputs[0])
                    or (child.op_type == "gather" and child.x is operation.outputs[0])
                    for child in operation.outputs[0].child_ops)
                if not valid_consumer or parameter in found:
                    raise ValueError(f"Expected one Conv/Embedding constant for {parameter}")
                found.add(parameter)
                op_names.append(operation.name)
                selected.append((parameter, value))
        if found != expected:
            raise ValueError(f"Saved FP16 weight inventory mismatch: missing={sorted(expected - found)}")
        cache = CachedPalettes(bits=bits, workers=workers)
        log("precomputing_palettes", weights=len(selected), workers=workers,
            batch_size=cache.batch_size)
        cache.prepare(selected, progress=lambda name, groups, cache_bytes: log(
            "palette_weight_ready", parameter=name, cumulative_groups=groups, cache_bytes=cache_bytes))
        return cache, op_names
    finally:
        # Constants and loop variables retain their parent program. Release all
        # of them before Core ML loads its independent compression-pass program.
        selected.clear()
        operation = function = value = program = spec = None
        gc.collect()


def compress(args):
    started = time.monotonic()
    verification_path = (args.source_verification.resolve() if args.source_verification is not None
                         else args.source.resolve().parent / "source-verification.json")
    source_identity, input_inventory, input_report_bytes, source_snapshot = validate_compression_input(
        args.input, args.source, verification_path)
    bits = getattr(args, "bits", 6)
    if bits not in (4, 6):
        raise ValueError("Supported palette bit widths are 4 and 6")
    workers = getattr(args, "palette_workers", 4)
    if workers < 1:
        raise ValueError("palette_workers must be positive")
    original = ct.models.MLModel(str(args.input), compute_units=ct.ComputeUnit.CPU_AND_NE,
                                skip_model_load=True)
    before = spec_report(original, args.kind)
    for key, value in {"part": args.kind, "context_length": args.context,
                       "layer_start": args.start, "layer_end": args.end}.items():
        if original.user_defined_metadata.get(key) != str(value):
            raise ValueError(f"Input package metadata does not match {key}={value}")
    expected_parameters = expected_weight_parameters(args)
    expected = len(expected_parameters)
    cache = None
    if workers > 1:
        cache, op_names = precompute_palette_cache(original, args, bits, workers)
    palettizer = ct.optimize.coreml.OpPalettizerConfig(
        mode="custom", lut_function=cache if cache is not None else partial(palette, bits=bits),
        granularity="per_grouped_channel", group_size=16, channel_axis=0,
        weight_threshold=COMPRESSION_WEIGHT_THRESHOLD)
    config = (ct.optimize.coreml.OptimizationConfig(op_name_configs={name: palettizer for name in op_names})
              if cache is not None else ct.optimize.coreml.OptimizationConfig(global_config=palettizer))
    log("palettizing", kind=args.kind, index_bits=bits, expected_weight_tensors=expected)
    model = ct.optimize.coreml.palettize_weights(original, config=config)
    if cache is not None:
        cache.assert_consumed()
    del original
    gc.collect()
    compressed_ops = [op for op in walk_ops(model._mil_program.functions["main"])
                      if op.op_type == "constexpr_lut_to_dense"]
    if len(compressed_ops) != expected:
        raise RuntimeError(f"Compression incomplete: expected {expected} weight tensors, got {len(compressed_ops)}")
    weight_evidence = [{**dense_lut_hash(op, expected_bits=bits),
                        "parameter_name": parameter_name_for_compressed_op(op.name)}
                       for op in compressed_ops]
    if {weight["parameter_name"] for weight in weight_evidence} != expected_parameters:
        raise RuntimeError("Compressed weight names do not match the required model parameters")
    report = spec_report(model, args.kind, before["states"])
    if any(report[field] != before[field] for field in
           ("inputs", "outputs", "input_dtypes", "output_dtypes")):
        raise RuntimeError("Compression changed the public model interface")
    if report["operation_counts"].get("constexpr_lut_to_dense") != expected:
        raise RuntimeError("The saved specification is missing compressed weight operations")
    annotate(model, args, compressed=True)
    assert_compression_input_unchanged(args.input, input_inventory, input_report_bytes)
    assert_pinned_source_snapshot(args.source, source_snapshot, verification_path)
    save_model(model, args.output)
    report.update({"stage": f"lut{bits}_compression", "part": args.kind,
                   "index_bits": bits, "weight_format": f"LUT{bits} grouped-channel-16",
                   "source_model": SOURCE_MODEL_ID, "source_revision": SOURCE_REVISION,
                   "source_weight_identity": source_identity,
                   "context_length": args.context, "layer_start": args.start,
                   "layer_end": args.end,
                   "seconds": time.monotonic() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "weights": weight_evidence,
                   "palette_execution": cache.report() if cache is not None else {
                       "workers": 1, "mechanism": "serial CUSTOM callback"},
                   "compression": {"index_bits": bits, "group_size": 16, "channel_axis": 0,
                     "algorithm": PALETTE_ALGORITHMS[bits],
                     "large_weight_tensors_expected": expected, "large_weight_tensors_compressed": len(compressed_ops)},
                   "files": tree_inventory(args.output), "apple_runtime_verified": False,
                   "ane_hardware_verified": False})
    atomic_json(str(args.output) + ".json", report)
    log("compressed", output=str(args.output), seconds=report["seconds"],
        bytes=sum(f["bytes"] for f in report["files"]), peak_rss_kib=report["peak_rss_kib"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("convert", "compress"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-verification", type=Path,
                        help="Pinned source-verification.json; defaults beside --source")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("embedding", "block", "head"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--bits", type=int, choices=(4, 6), default=6,
                        help="Palette bits; default 6 uses weighted global 1D k-means; 4 reproduces rejected legacy evidence")
    parser.add_argument("--palette-workers", type=int, default=4,
                        help="Bounded palette-fitting threads; 1 selects the original serial callback")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.palette_workers < 1:
        parser.error("--palette-workers must be positive")
    torch.set_num_threads(args.threads)
    if args.stage == "convert" and args.source is None:
        parser.error("--source is required for convert")
    if args.stage == "compress" and args.input is None:
        parser.error("--input is required for compress")
    if args.stage == "compress" and args.source is None:
        parser.error("--source is required to bind compression to pinned provenance")
    (convert if args.stage == "convert" else compress)(args)


if __name__ == "__main__":
    main()
