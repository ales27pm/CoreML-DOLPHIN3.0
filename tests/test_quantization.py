from __future__ import annotations

import argparse

import pytest

try:  # pragma: no cover - exercised during import
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "NumPy is required for quantization unit tests. Install the project "
        "dependencies via 'python -m pip install -r requirements-dev.txt' before "
        "running pytest."
    ) from exc

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantization import (
    NEURAL_ENGINE_GROUP_SIZES,
    SUPPORTED_MIXED_PRECISION_KEYS,
    SUPPORTED_WBITS,
    _cosine_similarity,
    _parse_mixed_precision_overrides,
    _resolve_mixed_precision_plan,
    _validate_group_size_for_backend,
    sweep_group_size_arg,
    sweep_wbits_arg,
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
        _cosine_similarity(
            np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)
        )


def test_parse_mixed_precision_overrides() -> None:
    overrides = _parse_mixed_precision_overrides("attention=6,mlp=4")
    assert overrides == {"attention": 6, "mlp": 4}


@pytest.mark.parametrize("invalid_spec", ["foo=4", "attention=5", "attention:"])
def test_parse_mixed_precision_overrides_rejects_invalid_entries(
    invalid_spec: str,
) -> None:
    with pytest.raises(ValueError):
        _parse_mixed_precision_overrides(invalid_spec)


def test_parse_mixed_precision_overrides_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_mixed_precision_overrides("attention=4, attention=6")


@pytest.mark.parametrize("compute_units", ["ALL", "CPU_AND_GPU"])
@pytest.mark.parametrize("group_size", NEURAL_ENGINE_GROUP_SIZES)
def test_validate_group_size_for_backend_accepts_ne_group_sizes(
    group_size: int, compute_units: str
) -> None:
    _validate_group_size_for_backend(group_size, compute_units)


def test_validate_group_size_for_backend_accepts_cpu_only_arbitrary_positive() -> None:
    for group_size in (1, 7, 13, 128):
        _validate_group_size_for_backend(group_size, "CPU_ONLY")


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
        ("layers.0.self_attn.rotary_emb.inv_freq", "attention"),
        ("layers.0.mlp.down_proj.weight", "mlp"),
        ("norm.weight", None),
        ("layers.0.self_attn.q_proj.bias", None),
        ("LAYERS.0.SELF_ATTN.Q_PROJ.WEIGHT", None),
        ("layers_0__self_attn__q_proj__weight", None),
        ("layers.0.self_attn.q_proj_weight", None),
        ("layers.0.self_attn.q_proj.weight_extra", None),
        ("layers.0.SELF_ATTN.q_proj.weight", None),
        ("layers.0.self_attn.k_proj__weight", None),
        ("layers.0.mlp.down_proj.weight_", None),
        ("layers.0.mlp.down_proj.WEIGHT", None),
    ],
)
def test_resolve_mixed_precision_plan_matches_patterns(
    name: str, expected: str | None
) -> None:
    overrides = {"attention": 6, "mlp": 4}
    plan, counts = _resolve_mixed_precision_plan([name], overrides)
    if expected is None:
        assert plan == {}
        assert not counts
    else:
        assert plan == {name: overrides[expected]}
        assert counts[expected] == 1


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


@pytest.mark.parametrize("bits", SUPPORTED_WBITS)
@pytest.mark.parametrize("category", SUPPORTED_MIXED_PRECISION_KEYS.keys())
def test_mixed_precision_overrides_allow_supported_bits(
    category: str, bits: int
) -> None:
    spec = f"{category}={bits}"
    overrides = _parse_mixed_precision_overrides(spec)
    assert overrides[category] == bits


def test_sweep_wbits_arg_parses_unique_ordered_values() -> None:
    assert sweep_wbits_arg("4,2,4,8") == (4, 2, 8)


@pytest.mark.parametrize("invalid", ["", "foo", "3", "2,seven"])
def test_sweep_wbits_arg_rejects_invalid_entries(invalid: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        sweep_wbits_arg(invalid)


def test_sweep_group_size_arg_accepts_positive_integers() -> None:
    assert sweep_group_size_arg("16,32,16") == (16, 32)


@pytest.mark.parametrize("invalid", ["0", "-2", "", "one"])
def test_sweep_group_size_arg_rejects_invalid_entries(invalid: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        sweep_group_size_arg(invalid)
