from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.artificial_intelligence.quantization_study import (
    HeuristicQuantizedModel,
    QuantizationResult,
    QuantizationSample,
    load_quantization_dataset,
    plot_quantization_tradeoffs,
    run_quantization_study,
    write_results,
)


def _sample_dataset() -> list[QuantizationSample]:
    return [
        QuantizationSample(prompt="alpha", difficulty=0.2, tokens=320),
        QuantizationSample(prompt="beta", difficulty=0.35, tokens=480),
        QuantizationSample(prompt="gamma", difficulty=0.5, tokens=620),
    ]


def test_run_quantization_study_behaviour() -> None:
    dataset = _sample_dataset()
    model = HeuristicQuantizedModel()
    results = run_quantization_study(model, dataset, bit_depths=(8, 6, 4, 2))
    assert [result.bits for result in results] == [8, 6, 4, 2]
    assert all(isinstance(result, QuantizationResult) for result in results)

    accuracies = [result.accuracy for result in results]
    throughputs = [result.tokens_per_second for result in results]
    assert accuracies == sorted(accuracies, reverse=True)
    assert throughputs == sorted(throughputs)


def test_write_results(tmp_path: Path) -> None:
    dataset = _sample_dataset()
    model = HeuristicQuantizedModel()
    results = run_quantization_study(model, dataset)
    output = write_results(tmp_path / "study.json", results)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "results" in payload
    assert len(payload["results"]) == len(results)


def test_load_quantization_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"prompt": "alpha", "difficulty": 0.25, "tokens": 400},
                {"prompt": "beta", "difficulty": 0.5, "tokens": 520},
            ]
        ),
        encoding="utf-8",
    )
    samples = load_quantization_dataset(dataset_path)
    assert len(samples) == 2
    assert samples[0].prompt == "alpha"


def test_plot_quantization_tradeoffs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dataset = _sample_dataset()
    model = HeuristicQuantizedModel()
    results = run_quantization_study(model, dataset)
    target = tmp_path / "plot.png"
    plot_quantization_tradeoffs(results, target)
    assert target.exists()


@pytest.mark.integration
def test_quantization_cli(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"prompt": "alpha", "difficulty": 0.2, "tokens": 320},
                    {"prompt": "beta", "difficulty": 0.4, "tokens": 560},
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.artificial_intelligence.quantization_study",
            "--dataset",
            str(dataset_path),
            "--results-json",
            str(report_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["results"]
