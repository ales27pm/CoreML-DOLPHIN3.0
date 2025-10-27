"""Lazy evaluation pipeline utilities for Task 7.

This module exposes eager and lazy summation helpers alongside a
memory-comparison routine that demonstrates the allocation benefits of
switching from a list comprehension to a generator expression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, asdict
import argparse
import json
import logging
import sys
import tracemalloc

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10_000_000
MINIMUM_REDUCTION = 0.8


class MemoryEfficiencyError(RuntimeError):
    """Raised when the lazy pipeline fails to achieve the expected savings."""


@dataclass(frozen=True)
class MemoryComparison:
    """Container describing the peak memory measurements for both strategies."""

    eager_peak: int
    lazy_peak: int
    reduction: float

    def to_dict(self) -> dict[str, float]:
        """Expose a JSON-serialisable mapping of the captured metrics."""

        return asdict(self)


def eager_sum(limit: int) -> int:
    """Compute the sum of doubled integers using a materialised list."""

    _validate_limit(limit)
    return sum([value * 2 for value in range(limit)])


def lazy_sum(limit: int) -> int:
    """Compute the sum of doubled integers using a generator pipeline."""

    _validate_limit(limit)
    return sum(value * 2 for value in range(limit))


def compare_memory(limit: int = DEFAULT_LIMIT) -> MemoryComparison:
    """Profile peak allocations for the eager and lazy implementations."""

    _validate_limit(limit)
    tracemalloc.start()
    try:
        eager_result, eager_peak = _run_with_peak(eager_sum, limit)
        tracemalloc.reset_peak()
        lazy_result, lazy_peak = _run_with_peak(lazy_sum, limit)
    finally:
        tracemalloc.stop()

    if eager_result != lazy_result:
        raise AssertionError(
            "Lazy pipeline altered the numeric result: "
            f"{eager_result} != {lazy_result}"
        )

    reduction = _calculate_reduction(eager_peak, lazy_peak)
    if reduction < MINIMUM_REDUCTION:
        raise MemoryEfficiencyError(
            "Lazy pipeline failed to reduce peak memory by at least "
            f"{MINIMUM_REDUCTION:.0%}. Observed reduction: {reduction:.2%}"
        )

    logger.debug(
        "Eager peak: %s bytes, lazy peak: %s bytes, reduction: %.2f%%",
        eager_peak,
        lazy_peak,
        reduction * 100,
    )

    return MemoryComparison(
        eager_peak=eager_peak,
        lazy_peak=lazy_peak,
        reduction=reduction,
    )


def _run_with_peak(func: Callable[[int], int], limit: int) -> tuple[int, int]:
    result = func(limit)
    _current, peak = tracemalloc.get_traced_memory()
    return result, peak


def _calculate_reduction(eager_peak: int, lazy_peak: int) -> float:
    if eager_peak <= 0:
        return 0.0
    return 1 - (lazy_peak / eager_peak)


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int):
        raise TypeError("Limit must be an integer")
    if limit < 0:
        raise ValueError("Limit must be non-negative")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:  # pragma: no cover - argparse converts
        raise argparse.ArgumentTypeError("Limit must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Limit must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare peak memory usage between eager and lazy doubling pipelines.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        help=(
            "Upper bound (exclusive) for the range of integers to process. "
            "Must be a positive integer."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices={"json", "text"},
        default="text",
        help="Select whether to print metrics as human-readable text or JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        help="Configure logging verbosity for troubleshooting.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.WARNING))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        comparison = compare_memory(args.limit)
    except MemoryEfficiencyError as error:
        logger.error("%s", error)
        return 2
    except Exception as error:  # pragma: no cover - defensive guard for CLI usage
        logger.exception("Unexpected error while comparing memory usage")
        return 1

    if args.output_format == "json":
        print(json.dumps(comparison.to_dict()))
    else:
        print(
            "Eager peak: {eager} bytes\nLazy peak: {lazy} bytes\nReduction: {reduction:.2%}".format(
                eager=comparison.eager_peak,
                lazy=comparison.lazy_peak,
                reduction=comparison.reduction,
            )
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    sys.exit(main())
