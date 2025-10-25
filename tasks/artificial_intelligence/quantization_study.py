"""Quantization accuracy and throughput study helpers (Task 29)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, MutableMapping, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BIT_DEPTHS: tuple[int, ...] = (8, 6, 4, 2)


@dataclass(frozen=True)
class QuantizationSample:
    """Metadata describing a dataset example used during quantisation tests."""

    prompt: str
    difficulty: float
    tokens: int


@dataclass(frozen=True)
class QuantizationResult:
    """Result row describing accuracy and throughput for a quantisation level."""

    bits: int
    accuracy: float
    tokens_per_second: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "bits": self.bits,
            "accuracy": round(self.accuracy, 6),
            "tokens_per_second": round(self.tokens_per_second, 3),
        }


class QuantizedModel(Protocol):
    """Protocol describing the interface required by :func:`run_quantization_study`."""

    def evaluate(
        self, dataset: Sequence[QuantizationSample], quantization_bits: int
    ) -> float:
        """Return the evaluation accuracy for *dataset* at ``quantization_bits``."""

    def benchmark(
        self, dataset: Sequence[QuantizationSample], quantization_bits: int
    ) -> float:
        """Return tokens per second for *dataset* at ``quantization_bits``."""


class HeuristicQuantizedModel:
    """Heuristic model approximating accuracy/speed trade-offs.

    The implementation models common behaviour observed when palettizing
    transformer weights:

    * accuracy gently decreases as bit depth shrinks,
    * throughput increases sub-linearly with more aggressive quantisation, and
    * particularly challenging prompts penalise both metrics.
    """

    def __init__(
        self,
        *,
        baseline_accuracy: float = 0.985,
        baseline_throughput: float = 22.5,
        max_bits: int = 8,
    ) -> None:
        if baseline_accuracy <= 0 or baseline_accuracy > 1:
            raise ValueError("baseline_accuracy must be in (0, 1]")
        if baseline_throughput <= 0:
            raise ValueError("baseline_throughput must be positive")
        if max_bits <= 0:
            raise ValueError("max_bits must be positive")
        self.baseline_accuracy = baseline_accuracy
        self.baseline_throughput = baseline_throughput
        self.max_bits = max_bits

    def _difficulty_factor(self, dataset: Sequence[QuantizationSample]) -> float:
        if not dataset:
            raise ValueError("Dataset must contain at least one sample")
        return min(1.0, mean(sample.difficulty for sample in dataset))

    def evaluate(
        self, dataset: Sequence[QuantizationSample], quantization_bits: int
    ) -> float:
        difficulty = self._difficulty_factor(dataset)
        bit_penalty = (self.max_bits - quantization_bits) * (0.012 + 0.01 * difficulty)
        accuracy = self.baseline_accuracy - bit_penalty
        return max(0.0, min(1.0, accuracy))

    def benchmark(
        self, dataset: Sequence[QuantizationSample], quantization_bits: int
    ) -> float:
        difficulty = self._difficulty_factor(dataset)
        token_mean = mean(sample.tokens for sample in dataset)
        improvement = math.pow(self.max_bits / quantization_bits, 0.75)
        difficulty_penalty = 1.0 - min(0.25, difficulty * 0.1)
        throughput = self.baseline_throughput * improvement * difficulty_penalty
        # Encourage realistic throughput by accounting for prompt length.
        length_scale = max(0.5, min(1.5, 512 / (token_mean + 1)))
        return throughput * length_scale


def load_quantization_dataset(path: Path) -> List[QuantizationSample]:
    """Load quantisation samples from ``path``.

    The JSON structure accepts either a list of samples or an object containing a
    ``"samples"`` key. Each sample must provide ``prompt``, ``difficulty``, and
    ``tokens`` fields.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, MutableMapping) and "samples" in payload:
        payload = payload["samples"]
    if not isinstance(payload, list):
        raise ValueError("Quantisation dataset must be a list of samples")

    samples: List[QuantizationSample] = []
    for entry in payload:
        if not isinstance(entry, MutableMapping):
            raise ValueError("Each sample must be a mapping of attributes")
        try:
            prompt = str(entry["prompt"])
            difficulty = float(entry["difficulty"])
            tokens = int(entry["tokens"])
        except KeyError as exc:  # pragma: no cover - programmer error
            raise ValueError(f"Sample missing required field: {exc}") from exc
        samples.append(QuantizationSample(prompt=prompt, difficulty=difficulty, tokens=tokens))
    if not samples:
        raise ValueError("Quantisation dataset cannot be empty")
    return samples


def run_quantization_study(
    model: QuantizedModel,
    dataset: Sequence[QuantizationSample],
    bit_depths: Iterable[int] = DEFAULT_BIT_DEPTHS,
) -> List[QuantizationResult]:
    results: List[QuantizationResult] = []
    for bits in bit_depths:
        if bits <= 0:
            raise ValueError("Bit depth must be positive")
        accuracy = model.evaluate(dataset, bits)
        throughput = model.benchmark(dataset, bits)
        results.append(QuantizationResult(bits=bits, accuracy=accuracy, tokens_per_second=throughput))
    return results


def write_results(path: Path, results: Sequence[QuantizationResult]) -> Path:
    payload = {"results": [result.to_json() for result in results]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def plot_quantization_tradeoffs(
    results: Sequence[QuantizationResult],
    output: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to generate plots") from exc

    bits = [result.bits for result in results]
    accuracy = [result.accuracy for result in results]
    throughput = [result.tokens_per_second for result in results]

    figure, axis1 = plt.subplots()
    axis2 = axis1.twinx()
    axis1.plot(bits, accuracy, marker="o", color="tab:blue", label="Accuracy")
    axis2.plot(bits, throughput, marker="s", color="tab:green", label="Tokens/s")

    axis1.set_xlabel("Quantisation bits")
    axis1.set_ylabel("Accuracy", color="tab:blue")
    axis2.set_ylabel("Tokens/s", color="tab:green")
    axis1.grid(True, linestyle="--", alpha=0.4)
    axis1.invert_xaxis()
    axis1.set_ylim(0, 1.05)
    axis1.set_title("Quantisation Accuracy Trade-offs")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quantisation accuracy/throughput analysis",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSON")
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=list(DEFAULT_BIT_DEPTHS),
        help="Bit depths to evaluate (default: 8 6 4 2)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Optional JSON file to persist the quantisation study results",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional PNG output path for the trade-off visualisation",
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
        dataset = load_quantization_dataset(args.dataset)
        model = HeuristicQuantizedModel(max_bits=max(args.bits))
        results = run_quantization_study(model, dataset, args.bits)
    except Exception as exc:  # noqa: BLE001 - surface detailed error to CLI
        logger.error("Failed to execute quantisation study: %s", exc)
        return 2

    if args.results_json:
        write_results(args.results_json, results)
        logger.info("Results written to %s", args.results_json)

    if args.plot:
        try:
            plot_quantization_tradeoffs(results, args.plot)
            logger.info("Plot exported to %s", args.plot)
        except RuntimeError as exc:  # pragma: no cover - depends on matplotlib
            logger.warning("Plot generation skipped: %s", exc)

    for result in results:
        logger.info(
            "bits=%d accuracy=%.4f tokens_per_second=%.2f",
            result.bits,
            result.accuracy,
            result.tokens_per_second,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
