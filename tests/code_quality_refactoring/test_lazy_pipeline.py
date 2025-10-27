from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.code_quality_refactoring.lazy_pipeline import (
    MemoryComparison,
    MemoryEfficiencyError,
    compare_memory,
    eager_sum,
    lazy_sum,
)


@pytest.mark.parametrize("limit", [0, 1, 10, 1_000])
def test_eager_and_lazy_sums_match(limit: int) -> None:
    assert eager_sum(limit) == lazy_sum(limit)


def test_compare_memory_returns_expected_reduction() -> None:
    result = compare_memory(limit=200_000)

    assert isinstance(result, MemoryComparison)
    assert result.eager_peak >= result.lazy_peak
    assert result.reduction >= 0.8


def test_compare_memory_raises_on_insufficient_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_run_with_peak(func, limit: int) -> tuple[int, int]:
        return func(limit), 100

    from tasks.code_quality_refactoring import lazy_pipeline

    monkeypatch.setattr(lazy_pipeline, "_run_with_peak", _mock_run_with_peak)

    with pytest.raises(MemoryEfficiencyError):
        lazy_pipeline.compare_memory(limit=1)


def test_cli_json_output() -> None:
    script = (
        Path(__file__).parents[2]
        / "tasks"
        / "code_quality_refactoring"
        / "lazy_pipeline.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--limit", "1000", "--output-format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert set(payload) == {"eager_peak", "lazy_peak", "reduction"}
    assert payload["lazy_peak"] <= payload["eager_peak"]
