"""Validation utilities for the Dolphin3.0 → Core ML pipeline.

The module follows Task 28 from ``Codex_Master_Task_Results.md`` by comparing
logits generated from two runtimes (typically PyTorch and Core ML) using cosine
similarity. The public API is intentionally flexible: callers can supply logits
as in-memory sequences, JSON files, or rely on repository-provided baselines.

The default CLI mirrors the workflow exercised by production operators:

* load baseline logits recorded from previous validation runs,
* compute cosine similarity with a configurable acceptance threshold, and
* persist the validation summary as JSON for regression tracking.

No heavyweight numerical dependencies are required which keeps the module
portable for CI environments where only the exported logits are available.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)

_BASELINE_PROMPTS_KEY = "prompts"
_PYTORCH_KEY = "pytorch"
_COREML_KEY = "coreml"


@dataclass(frozen=True)
class PipelineValidationResult:
    """Summary describing the comparison between two sets of logits."""

    prompt: str
    cosine_similarity: float
    threshold: float
    pytorch_source: Path | None
    coreml_source: Path | None
    package_path: Path | None

    @property
    def passed(self) -> bool:
        """Return ``True`` when the similarity satisfies the threshold."""

        return self.cosine_similarity >= self.threshold

    def to_json(self) -> dict[str, object]:
        """Serialise the result to a JSON-compatible dictionary."""

        return {
            "prompt": self.prompt,
            "cosine_similarity": round(self.cosine_similarity, 6),
            "threshold": self.threshold,
            "passed": self.passed,
            "pytorch_source": str(self.pytorch_source) if self.pytorch_source else None,
            "coreml_source": str(self.coreml_source) if self.coreml_source else None,
            "package_path": str(self.package_path) if self.package_path else None,
        }


class BaselineNotFoundError(FileNotFoundError):
    """Raised when a requested baseline prompt is missing."""


def _load_json(path: Path) -> MutableMapping[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - handled by caller
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc


def load_baselines(path: Path) -> Mapping[str, Mapping[str, Sequence[float]]]:
    """Load baseline logits from ``path``.

    The JSON format must mirror ``tasks/artificial_intelligence/baselines``.
    Each prompt contains ``"pytorch"`` and ``"coreml"`` arrays.
    """

    payload = _load_json(path)
    if _BASELINE_PROMPTS_KEY not in payload:
        raise ValueError(
            f"Baseline file {path} is missing '{_BASELINE_PROMPTS_KEY}' key"
        )

    prompts = payload[_BASELINE_PROMPTS_KEY]
    if not isinstance(prompts, Mapping):
        raise ValueError(
            f"Expected '{_BASELINE_PROMPTS_KEY}' to map prompts to logits"
        )

    normalized: dict[str, Mapping[str, Sequence[float]]] = {}
    for prompt, entries in prompts.items():
        if not isinstance(entries, Mapping):
            raise ValueError(f"Prompt '{prompt}' payload must be a mapping")
        if _PYTORCH_KEY not in entries or _COREML_KEY not in entries:
            raise ValueError(
                f"Prompt '{prompt}' requires both '{_PYTORCH_KEY}' and '{_COREML_KEY}' entries"
            )
        normalized[prompt] = {
            _PYTORCH_KEY: _ensure_sequence(entries[_PYTORCH_KEY], f"{prompt}::pytorch"),
            _COREML_KEY: _ensure_sequence(entries[_COREML_KEY], f"{prompt}::coreml"),
        }
    return normalized


def _ensure_sequence(values: object, label: str) -> Sequence[float]:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        try:
            return [float(item) for item in values]  # type: ignore[return-value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} logits contain non-numeric values") from exc
    raise TypeError(f"{label} logits must be a sequence of numbers")


def _load_logits_from_file(path: Path, label: str) -> Sequence[float]:
    payload = _load_json(path)
    if "logits" in payload:
        return _ensure_sequence(payload["logits"], f"{label}::logits")
    raise ValueError(
        f"File {path} does not contain a 'logits' entry; received keys: {sorted(payload.keys())}"
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Logit vectors must be the same length for cosine similarity calculation"
        )
    dot = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left))
    right_norm = math.sqrt(sum(r * r for r in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Logit vectors must be non-zero to compute cosine similarity")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _resolve_logits(
    prompt: str,
    direct: Sequence[float] | None,
    file_path: Path | None,
    baselines: Mapping[str, Mapping[str, Sequence[float]]] | None,
    source_key: str,
) -> tuple[Sequence[float], Path | None]:
    if direct is not None:
        return _ensure_sequence(direct, f"{prompt}::{source_key}"), None
    if file_path is not None:
        return _load_logits_from_file(file_path, source_key), file_path
    if baselines is not None:
        if prompt not in baselines:
            raise BaselineNotFoundError(
                f"Prompt '{prompt}' not present in baseline {list(baselines)}"
            )
        return baselines[prompt][source_key], None
    raise ValueError(
        "Logits not provided. Supply direct values, file paths, or a baseline file."
    )


def validate_pipeline(
    prompt: str,
    *,
    pytorch_logits: Sequence[float] | None = None,
    coreml_logits: Sequence[float] | None = None,
    pytorch_file: Path | None = None,
    coreml_file: Path | None = None,
    baseline_path: Path | None = None,
    threshold: float = 0.99,
    exporter: Callable[[], Path] | None = None,
) -> PipelineValidationResult:
    """Validate that two runtime outputs agree within ``threshold``.

    Args:
        prompt: Prompt used to produce logits.
        pytorch_logits: Optional in-memory PyTorch logits.
        coreml_logits: Optional in-memory Core ML logits.
        pytorch_file: JSON file containing a ``{"logits": [...]}`` payload.
        coreml_file: JSON file containing a ``{"logits": [...]}`` payload.
        baseline_path: Optional baseline file storing historical logits.
        threshold: Minimum cosine similarity accepted.
        exporter: Optional callback returning the exported ``.mlpackage`` path.

    Raises:
        ValueError: When cosine similarity falls below ``threshold`` or inputs are
            invalid.
        BaselineNotFoundError: When the prompt is missing from the baseline file.
    """

    baselines = load_baselines(baseline_path) if baseline_path else None

    pytorch_values, pytorch_source = _resolve_logits(
        prompt, pytorch_logits, pytorch_file, baselines, _PYTORCH_KEY
    )
    coreml_values, coreml_source = _resolve_logits(
        prompt, coreml_logits, coreml_file, baselines, _COREML_KEY
    )

    similarity = _cosine_similarity(pytorch_values, coreml_values)
    logger.debug(
        "Cosine similarity for prompt '%s': %.6f (threshold %.2f)",
        prompt,
        similarity,
        threshold,
    )

    if similarity < threshold:
        raise ValueError(
            f"Cosine similarity below threshold for prompt '{prompt}': {similarity:.6f}"
        )

    package_path = exporter() if exporter else None

    return PipelineValidationResult(
        prompt=prompt,
        cosine_similarity=similarity,
        threshold=threshold,
        pytorch_source=pytorch_source,
        coreml_source=coreml_source,
        package_path=package_path,
    )


def _write_result(output: Path, result: PipelineValidationResult) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Dolphin3.0 Core ML exports against PyTorch baselines",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to validate")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).with_name("baselines").joinpath("pipeline_logits.json"),
        help="Baseline JSON containing recorded logits (default: bundled baseline)",
    )
    parser.add_argument("--pytorch-logits", type=Path, help="Optional PyTorch logits JSON")
    parser.add_argument("--coreml-logits", type=Path, help="Optional Core ML logits JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold required for success (default: 0.99)",
    )
    parser.add_argument(
        "--package-path",
        type=Path,
        help="Path to the exported .mlpackage when validation succeeds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report destination",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level, logging.INFO))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    exporter: Callable[[], Path] | None = None
    if args.package_path:
        package_path = args.package_path
        exporter = lambda: package_path

    try:
        result = validate_pipeline(
            prompt=args.prompt,
            pytorch_file=args.pytorch_logits,
            coreml_file=args.coreml_logits,
            baseline_path=args.baseline,
            threshold=args.threshold,
            exporter=exporter,
        )
    except (ValueError, BaselineNotFoundError) as exc:
        logger.error("Validation failed: %s", exc)
        return 2

    if args.output:
        _write_result(args.output, result)
        logger.info("Validation report written to %s", args.output)
    else:
        logger.info(
            "Prompt '%s' validated with cosine similarity %.6f (threshold %.2f)",
            result.prompt,
            result.cosine_similarity,
            result.threshold,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
