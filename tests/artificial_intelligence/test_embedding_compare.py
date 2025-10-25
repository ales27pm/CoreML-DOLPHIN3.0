from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tasks.artificial_intelligence.embedding_compare import (
    EmbeddingComparisonResult,
    InMemoryEmbeddingModel,
    evaluate_embeddings,
    load_sts_dataset,
)


def _fixture_models() -> tuple[InMemoryEmbeddingModel, InMemoryEmbeddingModel]:
    model_a = InMemoryEmbeddingModel(
        {
            "hello world": [0.9, 0.3, 0.1],
            "core ml": [0.8, 0.5, 0.2],
            "quantization": [0.7, 0.1, 0.7],
        }
    )
    model_b = InMemoryEmbeddingModel(
        {
            "hola mundo": [0.91, 0.29, 0.05],
            "pipeline": [0.81, 0.48, 0.25],
            "quantizacion": [0.68, 0.12, 0.71],
        }
    )
    return model_a, model_b


def test_evaluate_embeddings_success() -> None:
    model_a, model_b = _fixture_models()
    dataset = [("hello world", "hola mundo"), ("core ml", "pipeline")]
    result = evaluate_embeddings(model_a, model_b, dataset, minimum_similarity=0.95)
    assert isinstance(result, EmbeddingComparisonResult)
    assert result.sample_count == 2
    assert result.average_similarity >= 0.95


def test_evaluate_embeddings_detects_regression() -> None:
    model_a, model_b = _fixture_models()
    dataset = [("hello world", "quantizacion")]
    with pytest.raises(ValueError):
        evaluate_embeddings(model_a, model_b, dataset, minimum_similarity=0.99)


def test_load_sts_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sts.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence_a", "sentence_b"])
        writer.writeheader()
        writer.writerow({"sentence_a": "hello world", "sentence_b": "hola mundo"})
        writer.writerow({"sentence_a": "core ml", "sentence_b": "pipeline"})
    pairs = load_sts_dataset(dataset_path)
    assert pairs == [("hello world", "hola mundo"), ("core ml", "pipeline")]


@pytest.mark.integration
def test_embedding_cli(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sts.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sentence_a", "sentence_b"])
        writer.writeheader()
        writer.writerow({"sentence_a": "hello world", "sentence_b": "hola mundo"})

    embeddings_a_path = tmp_path / "a.json"
    embeddings_b_path = tmp_path / "b.json"
    embeddings_a_path.write_text(
        json.dumps({"hello world": [0.9, 0.3, 0.1]}),
        encoding="utf-8",
    )
    embeddings_b_path.write_text(
        json.dumps({"hola mundo": [0.9, 0.29, 0.08]}),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.artificial_intelligence.embedding_compare",
            "--dataset",
            str(dataset_path),
            "--model-a",
            str(embeddings_a_path),
            "--model-b",
            str(embeddings_b_path),
            "--output",
            str(output_path),
            "--threshold",
            "0.9",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["average_similarity"] >= 0.9
