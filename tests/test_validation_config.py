"""Tests for validation configuration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dolphin2coreml_full import (  # imported after sys.path mutation
    DEFAULT_EMBEDDING_BENCHMARKS,
    DEFAULT_GOLDEN_PROMPTS,
    _load_validation_config,
    _write_validation_report,
)


def test_load_validation_config_defaults() -> None:
    prompts, embeddings = _load_validation_config(None)
    assert prompts == tuple(DEFAULT_GOLDEN_PROMPTS)
    assert embeddings == tuple(DEFAULT_EMBEDDING_BENCHMARKS)


def test_load_validation_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.json"
    payload = {
        "prompts": [
            {"prompt": "Explain deterministic validation.", "max_new_tokens": 16},
            {"prompt": "Summarise the exporter."},
        ],
        "embedding_sentences": ["Sentence A", "Sentence B"],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    prompts, embeddings = _load_validation_config(str(config_path))

    assert [item.prompt for item in prompts] == [
        "Explain deterministic validation.",
        "Summarise the exporter.",
    ]
    assert prompts[0].max_new_tokens == 16
    assert prompts[1].max_new_tokens == 32  # default when omitted
    assert list(embeddings) == ["Sentence A", "Sentence B"]


def test_load_validation_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        """
        prompts:
          - prompt: First prompt
            max_new_tokens: 8
        embedding_sentences:
          - Example sentence
        """,
        encoding="utf-8",
    )

    prompts, embeddings = _load_validation_config(str(config_path))

    assert len(prompts) == 1
    assert prompts[0].prompt == "First prompt"
    assert prompts[0].max_new_tokens == 8
    assert embeddings == ("Example sentence",)


@pytest.mark.parametrize(
    "config_text, expected_message",
    [
        ("prompts: []", "At least one prompt"),
        (
            json.dumps(
                {
                    "prompts": [{"prompt": "", "max_new_tokens": 4}],
                    "embedding_sentences": ["ok"],
                }
            ),
            "Prompt text must be a non-empty string",
        ),
    ],
)
def test_load_validation_config_rejects_invalid(
    tmp_path: Path, config_text: str, expected_message: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        _load_validation_config(str(config_path))
    assert expected_message in str(excinfo.value)


def test_write_validation_report(tmp_path: Path) -> None:
    report_path = tmp_path / "validation.json"
    payload = {"status": "ok", "metrics": {"decode_p90_ms": 12.34}}

    _write_validation_report(report_path, payload)

    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written == payload
