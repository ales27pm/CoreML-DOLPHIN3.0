"""0/1 knapsack optimizers with profiling helpers.

This module provides the production-ready implementation for
``Codex_Master_Task_Results`` Task 4.  It implements both memoised top-down and
iterative bottom-up dynamic programming strategies and exposes instrumentation
for comparing their runtime and memory characteristics.

The public API covers the following capabilities:

* ``knapsack_top_down`` – canonical recursive solver with LRU caching.
* ``knapsack_bottom_up`` – memory-optimised iterative solver using a 1-D DP
  buffer.
* ``profile_algorithms`` – executes both implementations under a
  ``tracemalloc`` profiler and asserts result parity and memory targets.
* ``write_profiles_to_csv`` – convenience helper for persisting collected
  metrics for regression analysis.
* ``main`` – CLI entry point used by the verification workflow to generate
  reproducible metrics.

All functions include thorough input validation and raise descriptive errors to
simplify downstream debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import argparse
import csv
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


class KnapsackInputError(ValueError):
    """Raised when the provided knapsack inputs are invalid."""


@dataclass(frozen=True)
class AlgorithmProfile:
    """Profiling information captured for a knapsack algorithm run."""

    name: str
    result: int
    time_seconds: float
    peak_bytes: int

    def to_row(self) -> List[str]:
        """Serialise the profile for CSV persistence."""

        return [
            self.name,
            str(self.result),
            f"{self.time_seconds:.9f}",
            str(self.peak_bytes),
        ]


def _validate_inputs(
    capacity: int, weights: Sequence[int], values: Sequence[int]
) -> None:
    """Validate knapsack inputs raising :class:`KnapsackInputError` when invalid."""

    if not isinstance(capacity, int):
        raise KnapsackInputError("capacity must be an integer")
    if capacity < 0:
        raise KnapsackInputError("capacity must be non-negative")
    if len(weights) != len(values):
        raise KnapsackInputError("weights and values must be the same length")
    if not weights:
        return
    for sequence, label in ((weights, "weights"), (values, "values")):
        for item in sequence:
            if not isinstance(item, int):
                raise KnapsackInputError(f"{label} must contain integers")
            if isinstance(item, bool):
                raise KnapsackInputError(f"{label} must contain integers, not booleans")
            if item < 0:
                raise KnapsackInputError(f"{label} must contain non-negative integers")


def knapsack_top_down(
    capacity: int, weights: Sequence[int], values: Sequence[int]
) -> int:
    """Solve the 0/1 knapsack problem using memoised recursion."""

    _validate_inputs(capacity, weights, values)
    n = len(weights)

    @lru_cache(maxsize=None)
    def solve(index: int, remaining: int) -> int:
        if index == n or remaining <= 0:
            return 0
        best = solve(index + 1, remaining)
        weight = weights[index]
        if weight <= remaining:
            candidate = values[index] + solve(index + 1, remaining - weight)
            if candidate > best:
                best = candidate
        return best

    return solve(0, capacity)


def knapsack_bottom_up(
    capacity: int, weights: Sequence[int], values: Sequence[int]
) -> int:
    """Solve the 0/1 knapsack problem using an iterative dynamic program."""

    _validate_inputs(capacity, weights, values)
    dp: List[int] = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for remaining in range(capacity, weight - 1, -1):
            candidate = value + dp[remaining - weight]
            if candidate > dp[remaining]:
                dp[remaining] = candidate
    return dp[capacity]


def profile_algorithms(
    capacity: int,
    weights: Sequence[int],
    values: Sequence[int],
    *,
    enforce_memory_ratio: bool = True,
) -> Tuple[AlgorithmProfile, AlgorithmProfile]:
    """Profile both implementations and return their metrics.

    Parameters
    ----------
    capacity, weights, values:
        Input describing the knapsack instance to evaluate.
    enforce_memory_ratio:
        When ``True`` (default) ensure that the bottom-up solver's peak memory
        consumption is at most 70% of the top-down variant.  This reflects the
        optimisation target called out in the historical specification.
    """

    _validate_inputs(capacity, weights, values)

    def _run(
        name: str, func: Callable[[int, Sequence[int], Sequence[int]], int]
    ) -> Tuple[int, float, int]:
        tracemalloc.start()
        try:
            start = time.perf_counter()
            result = func(capacity, weights, values)
            elapsed = time.perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
            logger.debug("%s traced memory: current=%d peak=%d", name, current, peak)
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
        return result, elapsed, peak

    top_result, top_time, top_peak = _run("top_down", knapsack_top_down)
    bottom_result, bottom_time, bottom_peak = _run("bottom_up", knapsack_bottom_up)

    if top_result != bottom_result:
        raise AssertionError(
            "Knapsack implementations produced divergent results: "
            f"top_down={top_result}, bottom_up={bottom_result}"
        )

    if enforce_memory_ratio and top_peak > 0:
        ratio = bottom_peak / top_peak
        logger.debug("Memory ratio bottom/top: %.3f", ratio)
        if ratio > 0.70:
            raise AssertionError(
                "Bottom-up knapsack solver exceeded memory target: "
                f"ratio={ratio:.3f} (top_peak={top_peak}, bottom_peak={bottom_peak})"
            )

    top_profile = AlgorithmProfile(
        name="top_down",
        result=top_result,
        time_seconds=top_time,
        peak_bytes=top_peak,
    )
    bottom_profile = AlgorithmProfile(
        name="bottom_up",
        result=bottom_result,
        time_seconds=bottom_time,
        peak_bytes=bottom_peak,
    )
    return top_profile, bottom_profile


def write_profiles_to_csv(
    path: Path, profiles: Iterable[AlgorithmProfile], *, newline: str = ""
) -> None:
    """Persist profiling results to ``path`` using a deterministic header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "result", "time_seconds", "peak_bytes"])
        for profile in profiles:
            writer.writerow(profile.to_row())


def _default_dataset() -> Tuple[int, List[int], List[int]]:
    """Return the deterministic dataset used by the CLI entry point."""

    capacity = 50
    weights = [3, 4, 7, 8, 9, 11, 13, 15, 19, 21]
    values = [4, 5, 10, 11, 13, 17, 19, 23, 29, 31]
    return capacity, weights, values


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for profiling the knapsack implementations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Override the default capacity used for profiling",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Comma separated list of weights matching the number of values. "
            "Defaults to the built-in benchmarking dataset."
        ),
    )
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma separated list of values matching the number of weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("knapsack_profiles.csv"),
        help="Destination CSV file for profiling results.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for diagnostic output.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    default_capacity, default_weights, default_values = _default_dataset()
    capacity = args.capacity if args.capacity is not None else default_capacity
    if args.weights is not None and args.values is None:
        parser.error("--weights requires --values to also be provided")
    if args.values is not None and args.weights is None:
        parser.error("--values requires --weights to also be provided")

    if args.weights is not None and args.values is not None:
        try:
            weights = [int(item) for item in args.weights.split(",") if item]
            values = [int(item) for item in args.values.split(",") if item]
        except ValueError as exc:  # pragma: no cover - CLI guard
            parser.error(f"Failed to parse integer payloads: {exc}")
        if len(weights) != len(values):  # pragma: no cover - CLI guard
            parser.error(
                "--weights and --values must contain the same number of entries"
            )
    else:
        weights = default_weights
        values = default_values

    try:
        top_profile, bottom_profile = profile_algorithms(capacity, weights, values)
    except (KnapsackInputError, AssertionError) as exc:  # pragma: no cover - CLI guard
        logger.error("Failed to profile algorithms: %s", exc)
        return 1

    write_profiles_to_csv(args.output, (top_profile, bottom_profile))
    logger.info("Profiles written to %s", args.output)
    return 0


__all__ = [
    "AlgorithmProfile",
    "KnapsackInputError",
    "knapsack_bottom_up",
    "knapsack_top_down",
    "main",
    "profile_algorithms",
    "write_profiles_to_csv",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
