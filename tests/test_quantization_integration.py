from __future__ import annotations

from collections import Counter

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dolphin_quantization import (
    SUPPORTED_MIXED_PRECISION_KEYS,
    _classify_weight_for_mixed_precision,
    _cosine_similarity,
    _parse_mixed_precision_overrides,
    _resolve_mixed_precision_plan,
)


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


def _expected_category_counts(weight_names: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in weight_names:
        category = _classify_weight_for_mixed_precision(name)
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
        category = _classify_weight_for_mixed_precision(name)
        assert category is not None
        assert overrides[category] == bits


def test_cosine_similarity_aligns_with_torch() -> None:
    torch.manual_seed(0)
    lhs = torch.randn(128, dtype=torch.float64)
    rhs = lhs.clone() * 0.5
    numpy_similarity = _cosine_similarity(lhs.numpy(), rhs.numpy())
    torch_similarity = torch.nn.functional.cosine_similarity(lhs, rhs, dim=0)
    assert float(torch_similarity) == pytest.approx(numpy_similarity)
