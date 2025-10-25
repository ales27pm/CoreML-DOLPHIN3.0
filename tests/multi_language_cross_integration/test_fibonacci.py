from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.multi_language_cross_integration.fibonacci import (
    FibonacciReport,
    fibonacci_parity,
    fibonacci_python,
    fibonacci_report,
    write_fibonacci_report,
)


def test_fibonacci_python_sequence_matches_expected() -> None:
    assert fibonacci_python(0) == []
    assert fibonacci_python(1) == [0]
    assert fibonacci_python(7) == [0, 1, 1, 2, 3, 5, 8]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, 1, 2], ["even", "odd", "odd", "even"]),
        ([5, 8, 13], ["odd", "even", "odd"]),
    ],
)
def test_fibonacci_parity(values: list[int], expected: list[str]) -> None:
    assert fibonacci_parity(values) == expected


def test_fibonacci_report_structure() -> None:
    report = fibonacci_report(10)
    assert isinstance(report, FibonacciReport)
    assert report.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert report.parity == [
        "even",
        "odd",
        "odd",
        "even",
        "odd",
        "odd",
        "even",
        "odd",
        "odd",
        "even",
    ]


def test_cli_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "fibonacci.json"
    path = write_fibonacci_report(5, target, indent=0)
    assert path == target
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {
        "sequence": [0, 1, 1, 2, 3],
        "parity": ["even", "odd", "odd", "even", "odd"],
    }


@pytest.mark.integration
def test_module_execution(tmp_path: Path) -> None:
    """Execute the module as a script to ensure CLI wiring functions."""

    output = tmp_path / "cli.json"
    subprocess.run(  # noqa: S603  # trusted input
        [
            sys.executable,
            "-m",
            "tasks.multi_language_cross_integration.fibonacci",
            "--count",
            "6",
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).parents[2],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sequence"] == [0, 1, 1, 2, 3, 5]
