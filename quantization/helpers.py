"""Helper utilities for quantization configuration and validation."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

SUPPORTED_MIXED_PRECISION_KEYS: Mapping[str, str] = OrderedDict(
    [
        ("attention", "Self-attention projection weights"),
        ("mlp", "Feed-forward projection weights"),
    ]
)

SUPPORTED_WBITS: Tuple[int, ...] = (2, 4, 6, 8)

NEURAL_ENGINE_GROUP_SIZES: Sequence[int] = (8, 16, 32, 64)

_MIXED_PRECISION_PATTERNS: OrderedDict[str, Tuple[str, ...]] = OrderedDict(
    [
        (
            "attention",
            (
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ),
        ),
        (
            "mlp",
            (
                "mlp",
                "feed_forward",
                "ffn",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        ),
    ]
)


def _validate_group_size_for_backend(group_size: int, compute_units: str) -> None:
    """Validate that the requested group size is supported by the backend."""

    if group_size <= 0:
        raise ValueError("Palettization group size must be positive.")  # noqa: TRY003

    if (
        compute_units in {"ALL", "CPU_AND_GPU"}
        and group_size not in NEURAL_ENGINE_GROUP_SIZES
    ):
        raise ValueError(
            "Neural Engine / GPU palettization requires group sizes in {8, 16, 32, 64}."
        )  # noqa: TRY003


def _parse_mixed_precision_overrides(raw: Optional[str]) -> Dict[str, int]:
    """Parse CLI overrides for mixed-precision palettization."""

    if raw is None:
        return {}

    overrides: Dict[str, int] = {}
    entries = [segment.strip() for segment in raw.split(",") if segment.strip()]
    if not entries:
        return overrides

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                "Mixed precision overrides must use key=value format (e.g., attention=6)."
            )  # noqa: TRY003
        key, value = entry.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in SUPPORTED_MIXED_PRECISION_KEYS:
            supported = ", ".join(SUPPORTED_MIXED_PRECISION_KEYS.keys())
            raise ValueError(
                f"Unsupported mixed precision key '{key}'. Use one of: {supported}."
            )  # noqa: TRY003
        if key in overrides:
            raise ValueError(f"Duplicate mixed precision override for '{key}'.")
        try:
            nbits = int(value)
        except ValueError as exc:  # pragma: no cover - argparse guards help text
            raise ValueError(f"Bit-width for '{key}' must be an integer.") from exc
        if nbits not in SUPPORTED_WBITS:
            supported_bits = ", ".join(str(bit) for bit in SUPPORTED_WBITS)
            raise ValueError(
                f"Bit-width {nbits} is not supported. Choose from: {supported_bits}."
            )  # noqa: TRY003
        overrides[key] = nbits
    return overrides


def _resolve_mixed_precision_plan(
    weight_names: Iterable[str], overrides: Mapping[str, int]
) -> Tuple[Dict[str, int], Counter[str]]:
    """Derive per-weight overrides from requested mixed-precision categories."""

    plan: Dict[str, int] = {}
    counts: Counter[str] = Counter()

    if not overrides:
        return plan, counts

    for name in weight_names:
        lowered = name.lower()
        if "bias" in lowered:
            continue
        if name != lowered:
            continue
        if "__" in name:
            continue
        segments = tuple(segment.strip() for segment in name.split("."))
        if not segments or segments[-1] != "weight":
            continue
        trunk = segments[:-1]
        if not trunk:
            continue
        for category, patterns in _MIXED_PRECISION_PATTERNS.items():
            if category not in overrides:
                continue
            if any(token in trunk for token in patterns):
                plan[name] = overrides[category]
                counts[category] += 1
                break

    return plan, counts


def _mixed_precision_arg(value: str) -> Dict[str, int]:
    """argparse helper that validates mixed-precision override specifications."""

    try:
        return _parse_mixed_precision_overrides(value)
    except ValueError as exc:  # pragma: no cover - argparse converts to CLI error
        raise argparse.ArgumentTypeError(str(exc)) from exc


__all__ = [
    "NEURAL_ENGINE_GROUP_SIZES",
    "SUPPORTED_MIXED_PRECISION_KEYS",
    "SUPPORTED_WBITS",
    "_mixed_precision_arg",
    "_parse_mixed_precision_overrides",
    "_resolve_mixed_precision_plan",
    "_validate_group_size_for_backend",
]
