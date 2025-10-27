"""Embedding quality comparison utilities (Task 30)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol implemented by embedding providers."""

    def encode(self, text: str) -> Sequence[float]:
        """Encode ``text`` into a vector representation."""


@dataclass(frozen=True)
class EmbeddingComparisonResult:
    """Aggregate statistics returned by :func:`evaluate_embeddings`."""

    average_similarity: float
    sample_count: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "average_similarity": round(self.average_similarity, 6),
            "sample_count": self.sample_count,
        }


class InMemoryEmbeddingModel:
    """Simple embedding model backed by a dictionary of vectors."""

    def __init__(
        self, embeddings: Mapping[str, Sequence[float]], *, normalise: bool = True
    ) -> None:
        if not embeddings:
            raise ValueError("Embeddings mapping cannot be empty")
        self._embeddings = {
            key: _normalise_vector(value) if normalise else list(value)
            for key, value in embeddings.items()
        }
        self.normalise = normalise

    def encode(self, text: str) -> Sequence[float]:
        try:
            vector = self._embeddings[text]
        except KeyError as exc:
            raise KeyError(f"Embedding for sentence '{text}' not found") from exc
        return vector


def _normalise_vector(values: Sequence[float]) -> List[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Embedding vectors must be a sequence of numbers")
    vector = [float(item) for item in values]
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        raise ValueError("Embedding vector must have non-zero magnitude")
    return [component / norm for component in vector]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must be the same dimension")
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left))
    right_norm = math.sqrt(sum(r * r for r in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must not be zero vectors")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def evaluate_embeddings(
    model_a: EmbeddingModel,
    model_b: EmbeddingModel,
    dataset: Iterable[Tuple[str, str]],
    *,
    minimum_similarity: float = 0.98,
) -> EmbeddingComparisonResult:
    """Evaluate the cosine similarity between two embedding models."""

    similarities: List[float] = []
    for sentence_a, sentence_b in dataset:
        vector_a = model_a.encode(sentence_a)
        vector_b = model_b.encode(sentence_b)
        similarity = _cosine_similarity(vector_a, vector_b)
        similarities.append(similarity)
        logger.debug(
            "Sentence pair '%s'/'%s' similarity: %.6f",
            sentence_a,
            sentence_b,
            similarity,
        )

    if not similarities:
        raise ValueError("Dataset must contain at least one pair")

    average_similarity = sum(similarities) / len(similarities)
    if average_similarity < minimum_similarity:
        raise ValueError(
            f"Average similarity {average_similarity:.6f} below threshold {minimum_similarity:.2f}"
        )
    return EmbeddingComparisonResult(
        average_similarity=average_similarity,
        sample_count=len(similarities),
    )


def load_sts_dataset(path: Path) -> List[Tuple[str, str]]:
    """Load an STS-style dataset containing sentence pairs from ``path``."""

    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or "sentence_a" not in reader.fieldnames
            or "sentence_b" not in reader.fieldnames
        ):
            raise ValueError(
                "Dataset must contain 'sentence_a' and 'sentence_b' columns"
            )
        for row in reader:
            sentence_a = row.get("sentence_a", "").strip()
            sentence_b = row.get("sentence_b", "").strip()
            if not sentence_a or not sentence_b:
                continue
            pairs.append((sentence_a, sentence_b))
    if not pairs:
        raise ValueError("Dataset must include at least one valid pair")
    return pairs


def _load_embeddings(path: Path) -> Mapping[str, Sequence[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, MutableMapping):
        raise ValueError(
            "Embeddings JSON must be an object mapping sentences to vectors"
        )
    embeddings: dict[str, Sequence[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, Sequence):
            raise ValueError(f"Embedding for '{key}' must be a sequence of floats")
        embeddings[str(key)] = [float(component) for component in value]
    return embeddings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare embedding models via cosine similarity"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to STS dataset (CSV)"
    )
    parser.add_argument(
        "--model-a", type=Path, required=True, help="JSON embeddings for model A"
    )
    parser.add_argument(
        "--model-b", type=Path, required=True, help="JSON embeddings for model B"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Minimum average similarity expected (default: 0.98)",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON file for summary statistics"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    try:
        dataset = load_sts_dataset(args.dataset)
        embeddings_a = _load_embeddings(args.model_a)
        embeddings_b = _load_embeddings(args.model_b)
        model_a = InMemoryEmbeddingModel(embeddings_a)
        model_b = InMemoryEmbeddingModel(embeddings_b)
        result = evaluate_embeddings(
            model_a, model_b, dataset, minimum_similarity=args.threshold
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding comparison failed: %s", exc)
        return 2

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
        logger.info("Summary written to %s", args.output)
    else:
        logger.info(
            "Average similarity %.6f across %d samples",
            result.average_similarity,
            result.sample_count,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
