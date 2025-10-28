from __future__ import annotations

from collections import Counter

import numpy as np

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantization import (
    SUPPORTED_MIXED_PRECISION_KEYS,
    _cosine_similarity,
    _parse_mixed_precision_overrides,
    _resolve_mixed_precision_plan,
)


pytestmark = pytest.mark.slow


def _build_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        vocab_size=128,
    )
    return LlamaForCausalLM(config)


def _classify_weight_for_test(weight_name: str) -> str | None:
    lowered = weight_name.lower()
    if "bias" in lowered:
        return None
    attention_tokens = (
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    mlp_tokens = (
        "mlp",
        "feed_forward",
        "ffn",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    if any(token in lowered for token in attention_tokens):
        return "attention"
    if any(token in lowered for token in mlp_tokens):
        return "mlp"
    return None


def _expected_category_counts(weight_names: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in weight_names:
        category = _classify_weight_for_test(name)
        if category is not None:
            counts[category] += 1
    return counts


def test_resolve_mixed_precision_plan_matches_llama_weights() -> None:
    model = _build_tiny_llama()
    weight_names = list(model.state_dict().keys())
    overrides = _parse_mixed_precision_overrides(
        ",".join(f"{key}=4" for key in SUPPORTED_MIXED_PRECISION_KEYS)
    )
    plan, counts = _resolve_mixed_precision_plan(weight_names, overrides)

    expected_counts = _expected_category_counts(weight_names)
    assert counts == expected_counts
    assert all(name in weight_names for name in plan)
    for name, bits in plan.items():
        category = _classify_weight_for_test(name)
        assert category is not None
        assert overrides[category] == bits


def test_cosine_similarity_aligns_with_torch() -> None:
    torch.manual_seed(0)
    lhs = torch.randn(128, dtype=torch.float64)
    rhs = lhs.clone() * 0.5
    numpy_similarity = _cosine_similarity(lhs.numpy(), rhs.numpy())
    torch_similarity = torch.nn.functional.cosine_similarity(lhs, rhs, dim=0)
    assert float(torch_similarity) == pytest.approx(numpy_similarity)


def test_cosine_similarity_near_zero_for_orthogonal_vectors() -> None:
    lhs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rhs = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert _cosine_similarity(lhs, rhs) == pytest.approx(0.0, abs=1e-7)
