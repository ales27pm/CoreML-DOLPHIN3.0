"""Loader integrity and independent Transformers 4.47.1 RoPE parity checks."""

import hashlib
import json

import pytest
from safetensors.torch import save_file
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from dolphin_ane.model import BlockPart, EmbeddingPart, HeadPart
from dolphin_ane.source import load_config, load_part, make_rope_tables
import dolphin_ane.source as source_module


@pytest.fixture
def source_data(tmp_path):
    torch.manual_seed(117)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=259, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rms_norm_eps=1e-5, tie_word_embeddings=True)
    reference = LlamaForCausalLM(config).float().eval()
    tensors = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
               if name != "lm_head.weight"}
    write_source(tmp_path, tensors, config)
    return tmp_path, config, tensors


def write_source(path, tensors, config):
    groups = [{}, {}]
    weights = {}
    for index, name in enumerate(sorted(tensors)):
        shard = index % 2
        groups[shard][name] = tensors[name]
        weights[name] = f"model-{shard + 1:05d}-of-00002.safetensors"
    for index, values in enumerate(groups):
        save_file(values, str(path / f"model-{index + 1:05d}-of-00002.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    (path / "config.json").write_text(json.dumps(config.to_dict()))


@pytest.mark.parametrize("kind", ["embedding", "block", "head"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_meta_loading_exact_rows_padding_and_frozen_parameters(source_data, monkeypatch, kind, dtype):
    path, config, tensors = source_data
    # Force many source reads to exercise row boundaries in the bounded loader.
    monkeypatch.setattr(source_module, "MAX_READ_ELEMENTS", 128)
    with torch.device("meta"):
        if kind == "embedding":
            part = EmbeddingPart(config, dtype=dtype)
        elif kind == "block":
            part = BlockPart(config, 1, 2, context_length=8, dtype=dtype)
        else:
            part = HeadPart(config, dtype=dtype)
    if kind == "block":
        part.keyCache.fill_(1)
        part.valueCache.fill_(2)
    assert load_part(part, path) is part
    assert not part.training
    for item in part.get_source_tensor_assignments():
        expected = tensors[item["source"]].to(dtype)
        if item["row_start"] is not None:
            expected = expected[item["row_start"]:item["row_stop"]]
        if item["pad_rows"]:
            expected = torch.cat((expected, expected.new_zeros(item["pad_rows"], *expected.shape[1:])))
        actual = part.get_parameter(item["destination"])
        assert actual.device.type == "cpu" and not actual.requires_grad
        torch.testing.assert_close(actual, expected.reshape(item["reshape"]), rtol=0, atol=0)
    if kind == "block":
        assert part.keyCache.device.type == "cpu"
        assert torch.count_nonzero(part.keyCache) == 0
        assert torch.count_nonzero(part.valueCache) == 0
    assert load_config(path)["vocab_size"] == 259


def test_fp16_overflow_rejected_without_partially_assigning_parameters(source_data):
    path, config, tensors = source_data
    # The original value is finite FP32/BF16, but its FP16 representation is not.
    tensors["model.embed_tokens.weight"][-1, 0] = 100000.0
    write_source(path, tensors, config)
    with torch.device("meta"):
        part = HeadPart(config)
    with pytest.raises(ValueError, match="Nonfinite value after conversion"):
        load_part(part, path)
    assert all(parameter.is_meta for parameter in part.parameters())


def test_missing_source_key_rejected_before_assignment(source_data):
    path, config, _ = source_data
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.1.mlp.down_proj.weight"]
    index_path.write_text(json.dumps(index))
    with torch.device("meta"):
        part = BlockPart(config, 1, 2, context_length=8)
    part.keyCache.fill_(3)
    with pytest.raises(KeyError, match="Source tensor missing"):
        load_part(part, path)
    assert all(parameter.is_meta for parameter in part.parameters())
    assert (part.keyCache == 3).all()


@pytest.mark.parametrize("rope_scaling", [None, {
    "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
    "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
}])
@torch.no_grad()
def test_rope_tables_exactly_match_transformers_fp16(rope_scaling):
    config = LlamaConfig(hidden_size=3072, num_attention_heads=24, head_dim=128,
        max_position_embeddings=131072, rope_theta=500000.0, rope_scaling=rope_scaling)
    caller_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        reference = LlamaRotaryEmbedding(config=config, device="cpu")
        expected_cos, expected_sin = reference(torch.zeros(1, 1, 3072), torch.arange(2048)[None])
    finally:
        torch.set_num_threads(caller_threads)
    actual_cos, actual_sin = make_rope_tables(config.to_dict(), 2048)
    assert torch.get_num_threads() == caller_threads
    assert actual_cos.shape == actual_sin.shape == (2048, 128)
    assert actual_cos.dtype == actual_sin.dtype == torch.float16
    torch.testing.assert_close(actual_cos, expected_cos[0].half(), atol=0, rtol=0)
    torch.testing.assert_close(actual_sin, expected_sin[0].half(), atol=0, rtol=0)


def test_dolphin_rope_hashes_are_canonical_and_thread_count_is_unchanged():
    config = LlamaConfig(hidden_size=3072, num_attention_heads=24, head_dim=128,
        max_position_embeddings=131072, rope_theta=500000.0, rope_scaling={
            "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        })
    caller_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        cos, sin = make_rope_tables(config.to_dict(), 2048)
        tail_cos, tail_sin = make_rope_tables(config.to_dict(), 2049)
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(caller_threads)
    assert hashlib.sha256(cos.numpy().astype("<f2", copy=False).tobytes()).hexdigest() == \
        "7a4e770a06b615795822b57e02f3eafd2c60c3859bd9d110ed7d68162c10794e"
    assert hashlib.sha256(sin.numpy().astype("<f2", copy=False).tobytes()).hexdigest() == \
        "ecc3cecff33021f825e995e163302e61dc66b96a860ce29964337091ae848d40"
    assert tail_cos.shape == tail_sin.shape == (2049, 128)
    torch.testing.assert_close(tail_cos[:2048], cos, atol=0, rtol=0)
    torch.testing.assert_close(tail_sin[:2048], sin, atol=0, rtol=0)
    try:
        torch.set_num_threads(1)
        reference = LlamaRotaryEmbedding(config=config, device="cpu")
        expected_cos, expected_sin = reference(torch.zeros(1, 1, 3072), torch.tensor([[2048]]))
    finally:
        torch.set_num_threads(caller_threads)
    torch.testing.assert_close(tail_cos[-1], expected_cos[0, 0].half(), atol=0, rtol=0)
    torch.testing.assert_close(tail_sin[-1], expected_sin[0, 0].half(), atol=0, rtol=0)


def test_unsupported_rope_is_rejected(source_data):
    _, config, _ = source_data
    config.rope_scaling = {"rope_type": "linear", "factor": 32.0}
    with pytest.raises(ValueError, match="Unsupported RoPE"):
        make_rope_tables(config, 8)
