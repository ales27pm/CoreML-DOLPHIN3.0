from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.artificial_intelligence.pipeline_validation import (
    BaselineNotFoundError,
    PipelineValidationResult,
    validate_pipeline,
)

BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks"
    / "artificial_intelligence"
    / "baselines"
    / "pipeline_logits.json"
)


def test_validate_pipeline_with_baseline() -> None:
    result = validate_pipeline("Hello", baseline_path=BASELINE_PATH)
    assert isinstance(result, PipelineValidationResult)
    assert result.passed
    assert result.cosine_similarity > 0.999


def test_validate_pipeline_detects_regression() -> None:
    with pytest.raises(ValueError):
        validate_pipeline(
            "regression",
            pytorch_logits=[0.2, 0.1, -0.6],
            coreml_logits=[-0.5, 0.3, 0.7],
            threshold=0.95,
        )


def test_missing_prompt_in_baseline_raises() -> None:
    with pytest.raises(BaselineNotFoundError):
        validate_pipeline("unknown", baseline_path=BASELINE_PATH)


@pytest.mark.integration
def test_pipeline_validation_cli(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.artificial_intelligence.pipeline_validation",
            "--prompt",
            "Hello",
            "--baseline",
            str(BASELINE_PATH),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["cosine_similarity"] >= 0.99
