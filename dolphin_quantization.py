"""Quantization helpers for the Dolphin Core ML export pipeline."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

SUPPORTED_MIXED_PRECISION_KEYS: Mapping[str, str] = {
    "attention": "Self-attention projection weights",
    "mlp": "Feed-forward projection weights",
}

SUPPORTED_WBITS: Tuple[int, ...] = (2, 4, 6, 8)

NEURAL_ENGINE_GROUP_SIZES: Sequence[int] = (8, 16, 32, 64)

_MIXED_PRECISION_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "attention": (
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    "mlp": (
        "mlp",
        "feed_forward",
        "ffn",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}

_MIXED_PRECISION_PRIORITY: Tuple[str, ...] = tuple(_MIXED_PRECISION_PATTERNS.keys())


def _validate_group_size_for_backend(group_size: int, compute_units: str) -> None:
    """Validate that the requested group size is supported by the backend."""

    if group_size <= 0:
        raise ValueError("Palettization group size must be positive.")

    if compute_units in {"ALL", "CPU_AND_GPU"} and group_size not in NEURAL_ENGINE_GROUP_SIZES:
        raise ValueError(
            "Neural Engine / GPU palettization requires group sizes in {8, 16, 32, 64}."
        )


def _parse_mixed_precision_overrides(raw: Optional[str]) -> Dict[str, int]:
    """Parse CLI overrides for mixed-precision palettization."""

    if raw is None or raw.strip() == "":
        return {}

    overrides: Dict[str, int] = {}
    for entry in raw.split(","):
        if "=" not in entry:
            raise ValueError(
                "Mixed precision overrides must use key=value format (e.g., attention=6)."
            )
        key, value = entry.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in SUPPORTED_MIXED_PRECISION_KEYS:
            supported = ", ".join(sorted(SUPPORTED_MIXED_PRECISION_KEYS))
            raise ValueError(f"Unsupported mixed precision key '{key}'. Use one of: {supported}.")
        try:
            nbits = int(value)
        except ValueError as exc:  # pragma: no cover - argparse guards help text
            raise ValueError(f"Bit-width for '{key}' must be an integer.") from exc
        if nbits not in SUPPORTED_WBITS:
            supported_bits = ", ".join(str(bit) for bit in SUPPORTED_WBITS)
            raise ValueError(
                f"Bit-width {nbits} is not supported. Choose from: {supported_bits}."
            )
        overrides[key] = nbits
    return overrides


def _classify_weight_for_mixed_precision(weight_name: str) -> Optional[str]:
    """Return the mixed-precision bucket for a given weight name if matched."""

    lowered = weight_name.lower()
    if "bias" in lowered:
        return None
    for category in _MIXED_PRECISION_PRIORITY:
        patterns = _MIXED_PRECISION_PATTERNS[category]
        if any(token in lowered for token in patterns):
            return category
    return None


def _resolve_mixed_precision_plan(
    weight_names: Iterable[str], overrides: Mapping[str, int]
) -> Tuple[Dict[str, int], Counter[str]]:
    """Derive per-weight overrides from requested mixed-precision categories."""

    plan: Dict[str, int] = {}
    counts: Counter[str] = Counter()

    if not overrides:
        return plan, counts

    for name in weight_names:
        category = _classify_weight_for_mixed_precision(name)
        if category is None:
            continue
        if category not in overrides:
            continue
        plan[name] = overrides[category]
        counts[category] += 1
    return plan, counts


def _cosine_similarity(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Compute cosine similarity between two vectors."""

    lhs_flat = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("Cosine similarity undefined for zero-norm embeddings.")
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


__all__ = [
    "SUPPORTED_MIXED_PRECISION_KEYS",
    "SUPPORTED_WBITS",
    "NEURAL_ENGINE_GROUP_SIZES",
    "_validate_group_size_for_backend",
    "_parse_mixed_precision_overrides",
    "_classify_weight_for_mixed_precision",
    "_resolve_mixed_precision_plan",
    "_cosine_similarity",
]
