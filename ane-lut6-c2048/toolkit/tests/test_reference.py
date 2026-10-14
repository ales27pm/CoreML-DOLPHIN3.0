"""Precision, baseline provenance, and fixed-prompt failure-gate checks."""

import copy
import hashlib
import json
import threading

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

import download_source
from dolphin_ane import reference as ref
from dolphin_ane.palette import palette


@pytest.mark.parametrize("bits", [4, 6])
def test_quantized_hashes_match_actual_group_reconstruction(bits):
    torch.manual_seed(22)
    part = nn.Sequential(nn.Conv2d(16, 32, 1, bias=False, dtype=torch.float16),
                         nn.LayerNorm(32, dtype=torch.float16))
    original = part[0].weight.detach().clone()
    expected = original.clone()
    for start in (0, 16):
        group = original[start:start + 16].numpy()
        lut, labels = palette(group, bits)
        expected[start:start + 16].copy_(torch.from_numpy(lut[labels].reshape(group.shape)))
    records = ref.quantize_and_hash_part(part, bits=bits, min_elements=100)
    torch.testing.assert_close(part[0].weight, expected, rtol=0, atol=0)
    assert records["0.weight"]["dense_reference_sha256"] == ref._dense_hash(expected)
    assert records["0.weight"]["source_fp16_sha256"] == ref._dense_hash(original)
    assert records["0.weight"]["index_bits"] == bits
    assert not records["1.weight"]["quantized"]


@pytest.mark.parametrize("bits", [4, 6])
def test_parallel_and_serial_full_part_hashes_are_identical_across_batches(bits, monkeypatch):
    torch.manual_seed(103)
    serial = nn.ModuleDict({"embedding": nn.Embedding(272, 32, dtype=torch.float16),
        "projection": nn.Conv2d(16, 144, 1, bias=False, dtype=torch.float16),
        "norm": nn.LayerNorm(16, dtype=torch.float16)})
    parallel = copy.deepcopy(serial)
    serial_hashes = ref.quantize_and_hash_part(serial, bits=bits, min_elements=100, palette_workers=1)
    main_thread = threading.get_ident()
    worker_threads = set()
    original_palette = ref.palette

    def read_only_palette(group, bits=6):
        assert not group.flags.writeable
        assert threading.get_ident() != main_thread
        worker_threads.add(threading.get_ident())
        return original_palette(group, bits=bits)

    monkeypatch.setattr(ref, "palette", read_only_palette)
    parallel_hashes = ref.quantize_and_hash_part(parallel, bits=bits, min_elements=100, palette_workers=4)
    assert parallel_hashes == serial_hashes
    assert worker_threads
    for name, parameter in serial.named_parameters():
        torch.testing.assert_close(parameter, parallel.get_parameter(name), rtol=0, atol=0)


def test_failed_parallel_batch_does_not_partially_mutate_parameter(monkeypatch):
    part = nn.Embedding(128, 32, dtype=torch.float16)
    original = part.weight.detach().clone()
    original_palette = ref.palette

    def reject_selected_group(group, bits=6):
        if float(group[0, 0]) == float(original[32, 0]):
            raise RuntimeError("Test fit failure")
        return original_palette(group, bits=bits)

    monkeypatch.setattr(ref, "palette", reject_selected_group)
    with pytest.raises(RuntimeError, match="Test fit failure"):
        ref.quantize_and_hash_part(part, bits=6, min_elements=100, palette_workers=4)
    torch.testing.assert_close(part.weight, original, rtol=0, atol=0)


def test_changed_source_parameter_rejected_before_candidate_forward():
    part = nn.Embedding(16, 32, dtype=torch.float16)
    expected = ref.quantize_and_hash_part(part, quantize=False)
    with torch.no_grad():
        part.weight[0, 0] += 1
    with pytest.raises(ValueError, match="FP16 parameter identity mismatch"):
        ref.quantize_and_hash_part(part, bits=6, expected_source=expected)


def test_strict_threshold_does_not_quantize_equal_size():
    part = nn.Embedding(16, 32, dtype=torch.float16)
    record = ref.quantize_and_hash_part(part, min_elements=512)["weight"]
    assert not record["quantized"] and record["index_bits"] is None


def test_fixed_prompt_gate_rejects_collapsed_wrong_outputs():
    measurements = [{"fixture_id": item["id"], "greedy_token_equal": False,
                     "source_greedy_token_id": source, "candidate_greedy_token_id": 95024}
                    for item, source in zip(ref.DEFAULT_FIXTURES, (9906, 19, 82681))]
    failed = ref.numerical_quality_gate(ref.DEFAULT_FIXTURES, measurements)
    assert not failed["passed"] and failed["degenerate_same_wrong_next_token"]
    for item in measurements:
        item.update(greedy_token_equal=True, candidate_greedy_token_id=item["source_greedy_token_id"])
    passed = ref.numerical_quality_gate(ref.DEFAULT_FIXTURES, measurements)
    assert passed["passed"] and passed["greedy_agreement_count"] == 3
    assert not ref.numerical_quality_gate(ref.DEFAULT_FIXTURES[:1], measurements[:1])["passed"]


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        index = next(i for i, fixture in enumerate(ref.DEFAULT_FIXTURES) if fixture["messages"] == messages)
        return [1, 2, 3 + index] if tokenize else json.dumps(messages, ensure_ascii=False)

    def decode(self, ids, skip_special_tokens=False):
        return str(ids[0])


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=259, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-5, tie_word_embeddings=True)
    source = tmp_path / "source"
    source.mkdir()
    model = LlamaForCausalLM(config).eval()
    save_file({name: value.detach().bfloat16().contiguous() for name, value in model.state_dict().items()
               if name != "lm_head.weight"}, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    shard = source / "model.safetensors"
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    expected = {"size": shard.stat().st_size, "sha256": digest}
    monkeypatch.setattr(download_source, "SHARDS", {shard.name: expected})
    monkeypatch.setattr(download_source, "PINNED_METADATA", {})
    (source.parent / "source-verification.json").write_text(json.dumps({
        "repo": download_source.REPO, "revision": download_source.REVISION, "all_verified": True,
        "files": {shard.name: {**expected, "verified": True}},
    }))
    monkeypatch.setattr(ref.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    prior = tmp_path / "baseline"
    ref.run_reference(source, prior, context_length=8, mode="source", threads=2)
    return source, prior


def test_reference_rehashes_source_shards_before_forward(baseline, tmp_path, monkeypatch):
    source, _ = baseline
    calls = []
    original_embed = ref._embed

    def recording_embed(*args):
        calls.append(1)
        return original_embed(*args)

    monkeypatch.setattr(ref, "_embed", recording_embed)
    shard = source / "model.safetensors"
    original_size = shard.stat().st_size
    contents = bytearray(shard.read_bytes())
    contents[-1] ^= 1
    shard.write_bytes(contents)
    assert shard.stat().st_size == original_size
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        ref.run_reference(source, tmp_path / "candidate", context_length=8, threads=2)
    assert calls == []


def test_baseline_reuse_skips_source_forward_and_preserves_all_logits(baseline, tmp_path, monkeypatch):
    source, prior = baseline
    calls = []
    original_embed = ref._embed

    def recording_embed(*args):
        calls.append(1)
        return original_embed(*args)

    monkeypatch.setattr(ref, "_embed", recording_embed)
    output = tmp_path / "candidate"
    report = ref.run_reference(source, output, context_length=8, threads=2, bits=6, source_reference=prior)
    assert len(calls) == 1  # Only the candidate embedding path executed.
    assert report["numerical_quality_gate_passed"]
    assert report["source_reference_reuse"]["all_source_fp16_parameter_hashes_verified"]
    assert report["source_reference_reuse"]["source_parameter_count_verified"] == 28
    assert report["source_identity"]["source_weight_identity"]["identity_sha256"]
    with np.load(prior / "reference-logits.npz") as old, np.load(output / "reference-logits.npz") as new:
        np.testing.assert_array_equal(old["source_fp16"], new["source_fp16"])
        np.testing.assert_array_equal(new["source_fp16"], new["lut6_reference"])


@pytest.mark.parametrize("mismatch", ["config", "fixture"])
def test_baseline_identity_mismatch_rejected(baseline, tmp_path, mismatch):
    source, prior = baseline
    path = prior / "reference-report.json"
    report = json.loads(path.read_text())
    if mismatch == "config":
        report["source_identity"]["config"]["rms_norm_eps"] = 0.25
    else:
        report["fixtures"][0]["token_ids"] = [123]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity differs"):
        ref.run_reference(source, tmp_path / "candidate", context_length=8, threads=2, source_reference=prior)


@pytest.mark.parametrize("schema", [True, False, 1.0, 2.0, 3.0, "3", 4])
@pytest.mark.parametrize("artifact", ["both", "reference-report.json", "reference-weight-hashes.json"])
def test_baseline_schemas_require_exact_supported_integers(baseline, tmp_path, schema, artifact):
    source, prior = baseline
    names = ("reference-report.json", "reference-weight-hashes.json") if artifact == "both" else (artifact,)
    for name in names:
        path = prior / name
        report = json.loads(path.read_text())
        report["schema_version"] = schema
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="schemas must match"):
        ref.run_reference(source, tmp_path / "candidate", context_length=8, threads=2, source_reference=prior)


@pytest.mark.parametrize("schema", [2, 3])
@pytest.mark.parametrize("missing", [True, False])
def test_nonlegacy_reference_cannot_fall_back_to_companion_config(baseline, tmp_path, schema, missing):
    source, prior = baseline
    for name in ("reference-report.json", "reference-weight-hashes.json"):
        path = prior / name
        report = json.loads(path.read_text())
        report["schema_version"] = schema
        if name == "reference-report.json":
            if missing:
                report.pop("source_identity")
            else:
                report["source_identity"] = None
        path.write_text(json.dumps(report))
    (source / "manifest.json").write_text(json.dumps({
        "source_revision": ref.SOURCE_REVISION, "context_length": 8}))
    with pytest.raises(ValueError, match="missing its required source identity"):
        ref.run_reference(source, tmp_path / "candidate", context_length=8, threads=2,
                          source_reference=prior, source_reference_config=source / "config.json")
