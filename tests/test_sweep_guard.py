"""Tests for the quantization sweep guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.sweep_guard import (
    compare_reports,
    main,
)  # noqa: E402  -- imported after sys.path mutation


def _write_report(path: Path, *, size_bytes: float, latency: float) -> None:
    payload = {
        "variants": [
            {
                "wbits": 4,
                "group_size": 16,
                "size_bytes": size_bytes,
                "performance": {
                    "aggregate": {
                        "decode_p50_ms": latency / 2,
                        "decode_p90_ms": latency,
                        "decode_p99_ms": latency * 1.2,
                    }
                },
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_reports_passes_within_threshold(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    _write_report(report_path, size_bytes=100.0, latency=10.0)
    _write_report(baseline_path, size_bytes=101.0, latency=10.5)

    result = compare_reports(
        report=json.loads(report_path.read_text(encoding="utf-8")),
        baseline=json.loads(baseline_path.read_text(encoding="utf-8")),
        variant_label=None,
        max_size_regression_pct=5.0,
        latency_metric="decode_p90_ms",
        max_latency_regression_pct=5.0,
    )

    assert result.passed
    assert result.size_delta_pct is not None and result.size_delta_pct < 0
    assert result.latency_delta_pct is not None and result.latency_delta_pct < 0


def test_compare_reports_fails_on_regression(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    baseline_path = tmp_path / "baseline.json"
    _write_report(report_path, size_bytes=120.0, latency=12.0)
    _write_report(baseline_path, size_bytes=100.0, latency=10.0)

    result = compare_reports(
        report=json.loads(report_path.read_text(encoding="utf-8")),
        baseline=json.loads(baseline_path.read_text(encoding="utf-8")),
        variant_label=None,
        max_size_regression_pct=5.0,
        latency_metric="decode_p90_ms",
        max_latency_regression_pct=5.0,
    )

    assert not result.passed
    assert any("size regression" in reason.lower() for reason in result.reasons)
    assert any("latency regression" in reason.lower() for reason in result.reasons)


@pytest.mark.parametrize("missing_baseline", [None, "absent.json"])
def test_main_skips_when_baseline_missing(
    tmp_path: Path, missing_baseline: str | None
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, size_bytes=100.0, latency=10.0)

    argv = ["--report", str(report_path)]
    if missing_baseline is not None:
        argv.extend(["--baseline", str(tmp_path / missing_baseline)])

    exit_code = main(argv)

    assert exit_code == 0
