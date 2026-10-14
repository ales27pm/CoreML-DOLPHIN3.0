"""Real small Core ML exports and serialized compression evidence on Linux."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import pytest
from safetensors.torch import save_file
import torch

import download_source
from dolphin_ane import exporter
from dolphin_ane.model import BlockPart, EmbeddingPart, HeadPart


def pin_test_source(source, monkeypatch, shard_bytes=None):
    """Create a tiny, fully pinned source and return its verification path."""
    source.mkdir(parents=True, exist_ok=True)
    shard = source / "model.safetensors"
    if not shard.exists():
        shard.write_bytes(shard_bytes or b"compression-only pinned source")
    config = source / "config.json"
    if not config.exists():
        config.write_text("{}")
    shards = {shard.name: {"size": shard.stat().st_size,
                           "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}}
    metadata = {config.name: {"size": config.stat().st_size,
                              "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}}
    monkeypatch.setattr(download_source, "SHARDS", shards)
    monkeypatch.setattr(download_source, "PINNED_METADATA", metadata)
    report = {"repo": download_source.REPO, "revision": download_source.REVISION,
              "all_verified": True, "files": {
                  shard.name: {**shards[shard.name], "verified": True},
                  config.name: {**metadata[config.name], "verified": True}}}
    verification = source.parent / "source-verification.json"
    verification.write_text(json.dumps(report))
    return verification, download_source.pinned_source_weight_identity(report)


def bind_compression_input(args, tmp_path, monkeypatch):
    source = tmp_path / "pinned-source"
    verification, identity = pin_test_source(source, monkeypatch)
    args.source = source
    args.source_verification = verification
    Path(str(args.input) + ".json").write_text(json.dumps({
        "source_weight_identity": identity, "files": exporter.tree_inventory(args.input)}))


@pytest.fixture
def recorded_compression_input(tmp_path, monkeypatch):
    package = tmp_path / "fp16.mlpackage"
    package.mkdir()
    (package / "payload.bin").write_bytes(b"FP16 package")
    args = argparse.Namespace(input=package)
    bind_compression_input(args, tmp_path, monkeypatch)
    return args


def test_compression_rehashes_current_source_before_loading_model(recorded_compression_input, monkeypatch):
    args = recorded_compression_input
    shard = args.source / "model.safetensors"
    contents = bytearray(shard.read_bytes())
    contents[-1] ^= 1
    shard.write_bytes(contents)
    calls = []
    monkeypatch.setattr(exporter.ct.models, "MLModel", lambda *a, **kw: calls.append(a))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        exporter.compress(args)
    assert calls == []


def test_compression_binds_the_report_bytes_it_parsed(recorded_compression_input, monkeypatch):
    args = recorded_compression_input
    report_path = Path(str(args.input) + ".json")
    original_report = report_path.read_bytes()
    inventory = exporter.tree_inventory

    def change_report_while_hashing(package):
        report_path.write_bytes(original_report + b" ")
        return inventory(package)

    monkeypatch.setattr(exporter, "tree_inventory", change_report_while_hashing)
    _, files, report_bytes, _ = exporter.validate_compression_input(
        args.input, args.source, args.source_verification)
    assert report_bytes == original_report
    with pytest.raises(RuntimeError, match="report changed during compression"):
        exporter.assert_compression_input_unchanged(args.input, files, report_bytes)


@pytest.fixture(scope="module", params=[4, 6])
def exported_block(tmp_path_factory, request):
    """Four tiny layers exercise the production expectation of 28 matrices."""
    tmp_path = tmp_path_factory.mktemp("exporter")
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    torch.manual_seed(816)
    torch.set_num_threads(2)
    config = dict(vocab_size=259, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=16, rms_norm_eps=1e-5, hidden_act="silu", tie_word_embeddings=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    part = BlockPart(config, 0, 4, 64)
    parameters = dict(part.named_parameters())
    weights = {}
    for assignment in part.get_source_tensor_assignments():
        value = parameters[assignment["destination"]].detach().clone()
        if value.ndim == 4:
            value = value.reshape(value.shape[:2])
        weights[assignment["source"]] = value
    save_file(weights, str(source / "model.safetensors"))
    verification, _ = pin_test_source(source, monkeypatch)
    args = argparse.Namespace(source=source, input=None, output=tmp_path / "fp16.mlpackage",
                              source_verification=verification, kind="block", start=0, end=4,
                              context=64, threads=2, bits=request.param)
    exporter.convert(args)
    args.input, args.output = args.output, tmp_path / f"lut{args.bits}.mlpackage"
    # Every tiny projection exceeds 256 values; production keeps its 65536 limit.
    monkeypatch.setattr(exporter, "COMPRESSION_WEIGHT_THRESHOLD", 256)
    captured = []
    save_model = exporter.save_model

    def capture_and_save(model, path):
        captured.append(model)
        save_model(model, path)

    monkeypatch.setattr(exporter, "save_model", capture_and_save)
    exporter.compress(args)
    return args, captured[0]


def test_saved_stage_reports_and_state_contract(exported_block):
    args, model = exported_block
    for path in (args.input, args.output):
        report = json.loads(path.with_suffix(".mlpackage.json").read_text())
        assert (report["context_length"], report["layer_start"], report["layer_end"]) == (64, 0, 4)
        assert report["states"] == {"keyCache": [4, 16, 1, 64], "valueCache": [4, 16, 1, 64]}
        assert set(report["input_dtypes"].values()) == {"FLOAT16"}
        assert report["output_dtypes"] == {"outputHiddenStates": "FLOAT16"}
        assert report["operation_counts"]["read_state"] >= 2
        assert report["operation_counts"]["write_state"] >= 2
        assert not report["apple_runtime_verified"] and not report["ane_hardware_verified"]
    report = json.loads(args.output.with_suffix(".mlpackage.json").read_text())
    assert report["compression"]["large_weight_tensors_expected"] == 28
    assert report["compression"]["index_bits"] == args.bits
    assert report["stage"] == f"lut{args.bits}_compression"
    assert report["weight_format"] == f"LUT{args.bits} grouped-channel-16"
    assert report["compression"]["large_weight_tensors_compressed"] == 28
    assert len(report["weights"]) == 28
    assert {weight["parameter_name"] for weight in report["weights"]} == exporter.expected_weight_parameters(args)
    assert model.__proxy__ is None  # skip_model_load survived compression.
    saved = ct.models.MLModel(str(args.output), skip_model_load=True)
    assert exporter.spec_report(saved, "block")["states"] == report["states"]
    assert saved.user_defined_metadata["weight_format"] == report["weight_format"]
    assert saved.user_defined_metadata["weight_index_bits"] == str(args.bits)


def test_streamed_hash_matches_coreml_materialized_weights(exported_block):
    args, model = exported_block
    # Reload from the serialized blobs: uint6 must survive save/read boundaries.
    from coremltools.converters.mil.frontend.milproto.load import load
    saved = ct.models.MLModel(str(args.output), skip_model_load=True)
    spec = saved.get_spec()
    program = load(spec, specification_version=spec.specificationVersion,
                   file_weights_dir=saved.weights_dir)
    for op in exporter.walk_ops(program.functions["main"]):
        if op.op_type == "constexpr_lut_to_dense":
            evidence = exporter.dense_lut_hash(op, expected_bits=args.bits)
            dense = np.ascontiguousarray(op.materialized_val_inference(), dtype="<f2")
            assert evidence["dense_fp16_sha256"] == hashlib.sha256(dense.tobytes()).hexdigest()
            assert evidence["shape"] == list(dense.shape)
            assert evidence["index_dtype"] == f"uint{args.bits}"
            assert evidence["index_bits"] == args.bits
            with pytest.raises(ValueError, match="Expected LUT"):
                exporter.dense_lut_hash(op, expected_bits=6 if args.bits == 4 else 4)


def test_bad_state_shape_is_rejected(exported_block):
    _, model = exported_block
    with pytest.raises(ValueError, match="cache shapes"):
        exporter.spec_report(model, "block", {"keyCache": [1], "valueCache": [1]})


def test_wrong_group_axis_is_rejected():
    op = SimpleNamespace(name="invalid", indices=SimpleNamespace(
        val=np.zeros((16, 32), np.uint8), dtype=SimpleNamespace(__name__="uint4")),
        lut=SimpleNamespace(val=np.zeros((1, 2, 16, 1), np.float16)))
    with pytest.raises(ValueError, match="axis 0"):
        exporter.dense_lut_hash(op)


def test_compression_rejects_mismatched_context_before_saving(exported_block, tmp_path):
    args, _ = exported_block
    wrong = argparse.Namespace(**vars(args))
    wrong.context = 1024
    wrong.output = tmp_path / "must_not_exist.mlpackage"
    with pytest.raises(ValueError, match="context_length"):
        exporter.compress(wrong)
    assert not wrong.output.exists()


def test_conversion_checks_integrity_before_publishing_package(exported_block, tmp_path, monkeypatch):
    original_args, _ = exported_block
    args = argparse.Namespace(**vars(original_args))
    args.output = tmp_path / "must_not_exist.mlpackage"
    shard = args.source / "model.safetensors"
    original_bytes, before = shard.read_bytes(), shard.stat()
    convert = exporter.ct.convert

    def mutate_during_conversion(*a, **kw):
        model = convert(*a, **kw)
        shard.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return model

    monkeypatch.setattr(exporter.ct, "convert", mutate_during_conversion)
    try:
        with pytest.raises(RuntimeError, match="changed during"):
            exporter.convert(args)
        assert not args.output.exists()
        assert not Path(str(args.output) + ".json").exists()
    finally:
        shard.write_bytes(original_bytes)


@pytest.mark.parametrize("changed_input", ["report", "source"])
def test_compression_checks_integrity_before_publishing_package(
        exported_block, tmp_path, monkeypatch, changed_input):
    original_args, _ = exported_block
    args = argparse.Namespace(**vars(original_args))
    args.output = tmp_path / "must_not_exist.mlpackage"
    changed_path = (Path(str(args.input) + ".json") if changed_input == "report"
                    else args.source / "model.safetensors")
    original_bytes, before = changed_path.read_bytes(), changed_path.stat()
    annotate = exporter.annotate

    def mutate_before_integrity_check(*a, **kw):
        annotate(*a, **kw)
        changed_path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
        os.utime(changed_path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))

    monkeypatch.setattr(exporter, "annotate", mutate_before_integrity_check)
    try:
        with pytest.raises(RuntimeError, match="changed during"):
            exporter.compress(args)
        assert not args.output.exists()
        assert not Path(str(args.output) + ".json").exists()
    finally:
        changed_path.write_bytes(original_bytes)


@pytest.mark.parametrize("name,expected", [
    ("layers_0_self_attn_q_proj_weight_promoted_to_fp16_palettized", "layers.0.self_attn.q_proj.weight"),
    ("layers_3_mlp_down_proj_weight_to_fp16_palettized", "layers.3.mlp.down_proj.weight"),
    ("lm_heads_7_weight_promoted_to_fp16_palettized", "lm_heads.7.weight"),
    ("embed_tokens_weight_to_fp16_palettized", "embed_tokens.weight"),
])
def test_parameter_names(name, expected):
    assert exporter.parameter_name_for_compressed_op(name) == expected


@pytest.mark.parametrize("kind,part_type,count", [("embedding", EmbeddingPart, 1), ("head", HeadPart, 8)])
@pytest.mark.parametrize("bits", [4, 6])
def test_embedding_and_head_compression_names_and_contracts(kind, part_type, count, bits, tmp_path, monkeypatch):
    config = dict(vocab_size=259, hidden_size=32, rms_norm_eps=1e-5, tie_word_embeddings=True)
    part = part_type(config).eval()
    examples, inputs, outputs, states = exporter.io_description(part, kind)
    with torch.no_grad():
        traced = torch.jit.trace(part, examples, check_trace=False)
    model = ct.convert(traced, inputs=inputs, outputs=outputs, states=states,
                       convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    args = argparse.Namespace(kind=kind, start=0, end=4, context=64,
                              input=tmp_path / "fp16.mlpackage", output=tmp_path / f"lut{bits}.mlpackage", bits=bits)
    exporter.annotate(model, args)
    exporter.save_model(model, args.input)
    bind_compression_input(args, tmp_path, monkeypatch)
    monkeypatch.setattr(exporter, "COMPRESSION_WEIGHT_THRESHOLD", 256)
    exporter.compress(args)
    report = json.loads(args.output.with_suffix(".mlpackage.json").read_text())
    assert len(report["weights"]) == count
    assert all(w["index_bits"] == bits and w["index_dtype"] == f"uint{bits}" for w in report["weights"])
    assert {weight["parameter_name"] for weight in report["weights"]} == exporter.expected_weight_parameters(args)
    assert report["states"] == {}
    if kind == "embedding":
        assert report["input_dtypes"] == {"inputIds": "INT32"}
    else:
        assert sum(shape[1] for shape in report["outputs"].values()) == 259


def test_threaded_cache_matches_serial_saved_weights_with_duplicate_groups(tmp_path, monkeypatch):
    """Repeated identical source rows require multiple callback uses per key."""
    torch.manual_seed(709)
    part = EmbeddingPart(dict(vocab_size=259, hidden_size=256)).eval()
    with torch.no_grad():
        group = torch.randn(16, 256, dtype=torch.float16) * .02
        part.embed_tokens.weight.copy_(group.repeat(17, 1))
        part.embed_tokens.weight[259:].zero_()
    examples, inputs, outputs, states = exporter.io_description(part, "embedding")
    with torch.no_grad():
        traced = torch.jit.trace(part, examples, check_trace=False)
    model = ct.convert(traced, inputs=inputs, outputs=outputs, states=states,
                       convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    args = argparse.Namespace(kind="embedding", start=0, end=0, context=64, bits=6,
                              input=tmp_path / "fp16.mlpackage")
    exporter.annotate(model, args)
    exporter.save_model(model, args.input)
    bind_compression_input(args, tmp_path, monkeypatch)
    reports = []
    for workers in (1, 4):
        args.palette_workers = workers
        args.output = tmp_path / f"workers{workers}.mlpackage"
        exporter.compress(args)
        reports.append(json.loads(Path(str(args.output) + ".json").read_text()))
    assert reports[0]["weights"] == reports[1]["weights"]
    # The serialized weight payload, including packed uint6 indices, is identical.
    blobs = [{record["path"]: record["sha256"] for record in report["files"]
              if record["path"].endswith(".bin")} for report in reports]
    assert blobs[0] and blobs[0] == blobs[1]
    execution = reports[1]["palette_execution"]
    assert execution["workers"] == 4 and execution["batch_size"] == 8
    assert execution["group_count"] == execution["cache_hits"] == 17
    assert execution["unique_groups"] == 2
    assert execution["cache_misses"] == execution["remaining_groups"] == 0
    assert execution["peak_cache_bytes"] == 2 * (4096 + 128)
