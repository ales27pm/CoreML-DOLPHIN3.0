"""Quantization sweep regression guard for CI."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Result of comparing a sweep report against a baseline."""

    variant_label: str
    size_delta_pct: float | None
    latency_delta_pct: float | None
    passed: bool
    reasons: tuple[str, ...]


def _load_report(path: Path) -> Mapping[str, Any]:
    """Load and validate a quantization sweep report."""

    if not path.exists():
        raise FileNotFoundError(f"Sweep report not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sweep report {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Sweep report must be a JSON object")
    if "variants" not in data or not isinstance(data["variants"], Sequence):
        raise ValueError("Sweep report is missing a 'variants' array")
    return data


def _variant_label(entry: Mapping[str, Any]) -> str:
    wbits = entry.get("wbits")
    group_size = entry.get("group_size")
    return (
        f"W{wbits}/G{group_size}"
        if wbits is not None and group_size is not None
        else "unknown"
    )


def _select_variant(
    report: Mapping[str, Any], label: Optional[str]
) -> Optional[Mapping[str, Any]]:
    variants = report.get("variants")
    if not isinstance(variants, Sequence):
        return None
    if label is None:
        return variants[0] if variants else None
    label = label.upper()
    for entry in variants:
        entry_label = _variant_label(entry).upper()
        if entry_label == label:
            return entry
    return None


def _percent_delta(new: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return ((new - baseline) / baseline) * 100.0


def compare_reports(
    *,
    report: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    variant_label: Optional[str],
    max_size_regression_pct: float,
    latency_metric: str,
    max_latency_regression_pct: float,
) -> GuardResult:
    """Compare the active sweep report against a baseline variant."""

    candidate = _select_variant(report, variant_label)
    if candidate is None:
        raise ValueError("Unable to locate the requested variant in the sweep report")

    label = _variant_label(candidate)
    reasons: list[str] = []

    size_delta_pct: float | None = None
    latency_delta_pct: float | None = None

    if baseline is not None:
        baseline_variant = _select_variant(baseline, variant_label)
        if baseline_variant is None:
            reasons.append(
                "Baseline report is missing the requested variant; skipping comparisons"
            )
        else:
            size_new = candidate.get("size_bytes")
            size_old = baseline_variant.get("size_bytes")
            if (
                isinstance(size_new, (int, float))
                and isinstance(size_old, (int, float))
                and size_old
            ):
                size_delta_pct = _percent_delta(float(size_new), float(size_old))
                if (
                    size_delta_pct is not None
                    and size_delta_pct > max_size_regression_pct
                ):
                    reasons.append(
                        f"Package size regression {size_delta_pct:.2f}% exceeds {max_size_regression_pct:.2f}% limit"
                    )
            else:
                reasons.append("Missing size_bytes metrics; skipping size comparison")

            perf_new = candidate.get("performance") or {}
            perf_old = baseline_variant.get("performance") or {}
            if isinstance(perf_new, Mapping) and isinstance(perf_old, Mapping):
                agg_new = perf_new.get("aggregate")
                agg_old = perf_old.get("aggregate")
                if isinstance(agg_new, Mapping) and isinstance(agg_old, Mapping):
                    metric_new = agg_new.get(latency_metric)
                    metric_old = agg_old.get(latency_metric)
                    if (
                        isinstance(metric_new, (int, float))
                        and isinstance(metric_old, (int, float))
                        and metric_old
                    ):
                        latency_delta_pct = _percent_delta(
                            float(metric_new), float(metric_old)
                        )
                        if (
                            latency_delta_pct is not None
                            and latency_delta_pct > max_latency_regression_pct
                        ):
                            reasons.append(
                                f"Latency regression on {latency_metric} {latency_delta_pct:.2f}% exceeds {max_latency_regression_pct:.2f}% limit"
                            )
                    else:
                        reasons.append(
                            f"Missing latency metric '{latency_metric}' in report or baseline; skipping latency comparison"
                        )
                else:
                    reasons.append(
                        "Aggregate latency metrics unavailable; skipping latency comparison"
                    )
            else:
                reasons.append(
                    "Performance metrics unavailable; skipping latency comparison"
                )
    else:
        reasons.append("No baseline provided; skipping regression checks")

    passed = not any(reason for reason in reasons if "exceeds" in reason.lower())
    return GuardResult(
        variant_label=label,
        size_delta_pct=size_delta_pct,
        latency_delta_pct=latency_delta_pct,
        passed=passed,
        reasons=tuple(reasons),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization sweep regression guard")
    parser.add_argument(
        "--report", required=True, help="Path to the latest sweep report JSON"
    )
    parser.add_argument("--baseline", help="Baseline sweep report JSON for comparison")
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant label to compare (e.g., W4/G16). Defaults to the first variant in the report",
    )
    parser.add_argument(
        "--max-size-regression",
        type=float,
        default=5.0,
        help="Maximum allowed package size regression percentage",
    )
    parser.add_argument(
        "--latency-metric",
        default="decode_p90_ms",
        help="Aggregate latency metric key to evaluate (default: decode_p90_ms)",
    )
    parser.add_argument(
        "--max-latency-regression",
        type=float,
        default=5.0,
        help="Maximum allowed latency regression percentage for the chosen metric",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report_path = Path(args.report).expanduser()
    try:
        report = _load_report(report_path)
    except Exception as exc:
        parser.error(str(exc))

    baseline_data: Mapping[str, Any] | None = None
    if args.baseline:
        baseline_path = Path(args.baseline).expanduser()
        if baseline_path.exists():
            baseline_data = _load_report(baseline_path)
        else:
            print(
                f"[sweep-guard] Baseline report not found at {baseline_path}; skipping regression comparisons"
            )

    result = compare_reports(
        report=report,
        baseline=baseline_data,
        variant_label=args.variant,
        max_size_regression_pct=args.max_size_regression,
        latency_metric=args.latency_metric,
        max_latency_regression_pct=args.max_latency_regression,
    )

    print(f"[sweep-guard] Evaluated variant {result.variant_label}")
    if result.size_delta_pct is not None:
        print(f"  Size delta: {result.size_delta_pct:.2f}%")
    else:
        print("  Size delta: unavailable")
    if result.latency_delta_pct is not None:
        print(f"  {args.latency_metric} delta: {result.latency_delta_pct:.2f}%")
    else:
        print(f"  {args.latency_metric} delta: unavailable")

    for reason in result.reasons:
        print(f"  note: {reason}")

    if not result.passed:
        print("[sweep-guard] Regression guard failed", flush=True)
        return 1

    print("[sweep-guard] Regression guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
