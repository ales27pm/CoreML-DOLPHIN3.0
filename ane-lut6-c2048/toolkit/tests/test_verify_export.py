"""Saved-logit, precision metadata, sidecar, and RoPE integrity boundaries."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

import download_source
from dolphin_ane import reference as ref
from dolphin_ane.source import make_rope_tables
import verify_export as verify


@pytest.mark.parametrize("bits", [4, 6])
def test_manifest_precision_is_consistent_and_explicit(bits):
    manifest = {"format_version": 2, "weights": {"format": f"LUT{bits}", "index_bits": bits,
        "group_size": 16, "channel_axis": 0, "lut_dtype": "float16"}}
    assert verify.compression_bits(manifest) == bits
    with pytest.raises(ValueError, match="precision differs"):
        verify.compression_bits(manifest, expected_bits=6 if bits == 4 else 4)
    manifest["weights"]["group_size"] = 32
    with pytest.raises(ValueError, match="compression format"):
        verify.compression_bits(manifest)


def reference_files(directory, collapsed=False):
    bits, vocab = 6, 16
    cases = [{**copy.deepcopy(item), "token_ids": [1, 2, 3], "prompt_token_count": 3,
              "rendered_prompt": item["messages"][0]["content"]} for item in ref.DEFAULT_FIXTURES]
    source = np.tile(np.arange(vocab, dtype=np.float16) / 100, (3, 1))
    for index, winner in enumerate((3, 4, 5)):
        source[index, winner] = 10
    candidate = source.copy()
    if collapsed:
        candidate[:, 7] = 20
    comparisons = [{"fixture_id": case["id"], **ref._compare(torch.from_numpy(s), torch.from_numpy(q), bits=bits)}
                   for case, s, q in zip(cases, source, candidate)]
    gate = ref.numerical_quality_gate(cases, comparisons)
    summary = lambda row, case: {"fixture_id": case["id"], "finite": True, "vocabulary_size": vocab,
                                 "greedy_token_id": int(row.argmax()), "top_10": [
                                     {"token_id": int(i), "logit": float(row[i])} for i in np.argsort(row)[-10:][::-1]]}
    report = {"schema_version": 2, "fixtures": cases, "results": {
        "source_fp16": [summary(row, case) for row, case in zip(source, cases)],
        "lut6_reference": [summary(row, case) for row, case in zip(candidate, cases)]},
        "source_vs_lut6": comparisons, "numerical_quality_gate": gate, "numerical_quality_gate_passed": gate["passed"]}
    sparse = {"schema_version": 2, "index_bits": bits, "context_length": 8, "backend": "PyTorch CPU eager test",
              "numerical_quality_gate_passed": gate["passed"], "fixtures": [
                  {**case, "logits_npz_key": "lut6_reference", "logits_npz_row": index,
                   "greedy_token_id": int(row.argmax()), "probe_logits": [
                       {"token_id": i, "logit": float(row[i])} for i in range(vocab)]}
                  for index, (case, row) in enumerate(zip(cases, candidate))]}
    (directory / "reference-report.json").write_text(json.dumps(report))
    (directory / "reference-fixtures.json").write_text(json.dumps(sparse))
    np.savez_compressed(directory / "reference-logits.npz", source_fp16=source, lut6_reference=candidate)
    return report, source, candidate


def test_recompute_gate_from_saved_logits_and_keep_device_tolerance_provisional(tmp_path):
    report, _, _ = reference_files(tmp_path)
    result = verify.verify_fixtures(tmp_path, 8, 16, bits=6, report=report)
    assert result["numerical_quality_gate_passed"]
    assert not result["coreml_runtime_validated"] and not result["ane_hardware_validated"]
    assert "not measured or calibrated" in result["tolerance_policy"]["status"]
    assert [item["expected_next_token"] for item in result["cases"]] == [3, 4, 5]


def test_collapsed_wrong_logits_cannot_pass_even_with_finite_arrays(tmp_path):
    report, _, _ = reference_files(tmp_path, collapsed=True)
    result = verify.verify_fixtures(tmp_path, 8, 16, bits=6, report=report)
    assert not result["numerical_quality_gate_passed"]
    assert result["numerical_quality_gate"]["degenerate_same_wrong_next_token"]
    report["numerical_quality_gate_passed"] = True
    with pytest.raises(ValueError, match="gate differs"):
        verify.verify_fixtures(tmp_path, 8, 16, bits=6, report=report)


@pytest.mark.parametrize("tamper", ["precision", "logits", "row"])
def test_sparse_fixture_or_saved_logits_tampering_rejected(tmp_path, tamper):
    report, source, candidate = reference_files(tmp_path)
    sparse_path = tmp_path / "reference-fixtures.json"
    sparse = json.loads(sparse_path.read_text())
    if tamper == "precision":
        sparse["index_bits"] = 4
    elif tamper == "row":
        sparse["fixtures"][0]["logits_npz_row"] = 1
    else:
        candidate[0, 0] += 1
        np.savez_compressed(tmp_path / "reference-logits.npz", source_fp16=source, lut6_reference=candidate)
    sparse_path.write_text(json.dumps(sparse))
    with pytest.raises(ValueError):
        verify.verify_fixtures(tmp_path, 8, 16, bits=6, report=report)


@pytest.mark.parametrize("tamper", [None, "tokenizer", "rope", "source_identity"])
def test_runtime_sidecar_bytes_and_rope_values_are_verified(tmp_path, monkeypatch, tamper):
    config = {"hidden_size": 3072, "num_attention_heads": 24, "head_dim": 128,
              "rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
              "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}
    files = {name: b"{}\n" for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                                        "special_tokens_map.json")}
    files["config.json"] = json.dumps(config).encode()
    files["SOURCE_MODEL_CARD.md"] = b"Pinned test model card\n"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = {"all_verified": True, "revision": verify.REVISION, "repo": "dphn/Dolphin3.0-Llama3.2-3B", "files": {}}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
        original = "README.md" if name == "SOURCE_MODEL_CARD.md" else name
        (source_dir / original).write_bytes(content)
        source["files"][original] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "verified": True}
    shard_values = {"a.safetensors": b"pinned-a", "b.safetensors": b"pinned-b"}
    index = json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}).encode()
    (source_dir / "model.safetensors.index.json").write_bytes(index)
    source["files"]["model.safetensors.index.json"] = {
        "size": len(index), "sha256": hashlib.sha256(index).hexdigest(), "verified": True}
    shards = {}
    for name, content in shard_values.items():
        (source_dir / name).write_bytes(content)
        shards[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    metadata = {name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name in ("config.json", "generation_config.json", "model.safetensors.index.json",
                             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md")
                for path in [source_dir / name]}
    monkeypatch.setattr(download_source, "SHARDS", shards)
    monkeypatch.setattr(download_source, "PINNED_METADATA", metadata)
    for name, expected in shards.items():
        source["files"][name] = {**expected, "verified": True}
    source["source_weight_identity"] = download_source.pinned_source_weight_identity(source)
    source_path = tmp_path / "source-verification.json"
    source_path.write_text(json.dumps(source))
    report = {"schema_version": 2, "source_identity": {"config": config,
              "config_file_sha256": hashlib.sha256(files["config.json"]).hexdigest(), "tokenizer_files": [],
              "source_weight_identity": source["source_weight_identity"]}}
    cos, sin = make_rope_tables(config, 8)
    for name, table in (("rope_cos.fp16", cos), ("rope_sin.fp16", sin)):
        (tmp_path / name).write_bytes(table.numpy().astype("<f2").tobytes())
    if tamper == "tokenizer":
        (tmp_path / "tokenizer.json").write_bytes(b"[]\n")
    elif tamper == "rope":
        path = tmp_path / "rope_sin.fp16"
        contents = bytearray(path.read_bytes())
        contents[3] ^= 1
        path.write_bytes(contents)
    elif tamper == "source_identity":
        report["source_identity"]["source_weight_identity"] = {
            **source["source_weight_identity"], "identity_sha256": "0" * 64}
    if tamper:
        with pytest.raises(ValueError, match="SHA256 mismatch|RoPE values differ|source shard identity differs"):
            verify.verify_sidecars(tmp_path, report, 8, source_dir=source_dir, source_verification=source_path)
    else:
        records, _ = verify.verify_sidecars(tmp_path, report, 8, source_dir=source_dir, source_verification=source_path)
        assert len(records) == 8 and {item["path"] for item in records} == set(files) | {"rope_cos.fp16", "rope_sin.fp16"}
