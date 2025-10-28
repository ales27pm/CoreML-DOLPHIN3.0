from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dolphin_quantization import (
    NEURAL_ENGINE_GROUP_SIZES,
    SUPPORTED_MIXED_PRECISION_KEYS,
    SUPPORTED_WBITS,
    _classify_weight_for_mixed_precision,
    _cosine_similarity,
    _parse_mixed_precision_overrides,
    _resolve_mixed_precision_plan,
    _validate_group_size_for_backend,
)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ],
)
def test_cosine_similarity_matches_expected(vector: np.ndarray) -> None:
    assert _cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        _cosine_similarity(np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32))


def test_parse_mixed_precision_overrides() -> None:
    overrides = _parse_mixed_precision_overrides("attention=6,mlp=4")
    assert overrides == {"attention": 6, "mlp": 4}


@pytest.mark.parametrize("invalid_spec", ["foo=4", "attention=5", "attention:"])
def test_parse_mixed_precision_overrides_rejects_invalid_entries(invalid_spec: str) -> None:
    with pytest.raises(ValueError):
        _parse_mixed_precision_overrides(invalid_spec)


@pytest.mark.parametrize("compute_units", ["ALL", "CPU_AND_GPU"])
@pytest.mark.parametrize("group_size", NEURAL_ENGINE_GROUP_SIZES)
def test_validate_group_size_for_backend_accepts_ne_group_sizes(group_size: int, compute_units: str) -> None:
    _validate_group_size_for_backend(group_size, compute_units)


@pytest.mark.parametrize(
    "group_size",
    [0, -8, 7],
)
def test_validate_group_size_for_backend_rejects_invalid_sizes(group_size: int) -> None:
    with pytest.raises(ValueError):
        _validate_group_size_for_backend(group_size, "ALL")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("layers.0.self_attn.q_proj.weight", "attention"),
        ("layers.0.self_attn.k_proj.weight", "attention"),
        ("layers.0.mlp.down_proj.weight", "mlp"),
        ("norm.weight", None),
        ("layers.0.self_attn.q_proj.bias", None),
    ],
)
def test_classify_weight_for_mixed_precision(name: str, expected: str | None) -> None:
    assert _classify_weight_for_mixed_precision(name) == expected


def test_resolve_mixed_precision_plan_counts() -> None:
    overrides = {"attention": 6, "mlp": 4}
    names = [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.bias",
    ]
    plan, counts = _resolve_mixed_precision_plan(names, overrides)
    assert plan == {
        "layers.0.self_attn.q_proj.weight": 6,
        "layers.0.self_attn.k_proj.weight": 6,
        "layers.0.mlp.down_proj.weight": 4,
    }
    assert counts["attention"] == 2
    assert counts["mlp"] == 1


@pytest.mark.parametrize("category", SUPPORTED_MIXED_PRECISION_KEYS.keys())
def test_mixed_precision_overrides_allow_supported_bits(category: str) -> None:
    spec = f"{category}=4"
    overrides = _parse_mixed_precision_overrides(spec)
    assert overrides[category] in SUPPORTED_WBITS
