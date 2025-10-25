"""Memoized Fibonacci utilities with parity reporting.

This module provides a production-ready implementation of the Fibonacci
sequence tailored for cross-language parity validation with the matching
TypeScript implementation. The functions are intentionally iterative to
avoid recursion depth limits while leveraging memoization so that repeated
invocations remain O(1) once the target sequence length has been computed.

Running the module as a script writes a JSON report containing the sequence
and its parity classification, which can be diffed against the TypeScript
export to confirm behavioural equivalence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

Parity = str


@dataclass(frozen=True)
class FibonacciReport:
    """Structured result containing the sequence and parity labels."""

    sequence: List[int]
    parity: List[Parity]

    def to_dict(self) -> Dict[str, List[object]]:
        """Return a serialisable representation of the report."""

        return {"sequence": list(self.sequence), "parity": list(self.parity)}


def _validate_count(n: int) -> None:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")


@lru_cache(maxsize=None)
def _fibonacci_tuple(n: int) -> tuple[int, ...]:
    """Return the first *n* Fibonacci numbers as an immutable tuple."""

    _validate_count(n)
    if n == 0:
        return ()
    if n == 1:
        return (0,)
    sequence = [0, 1]
    for _ in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return tuple(sequence[:n])


def fibonacci_python(n: int) -> List[int]:
    """Return the first *n* Fibonacci numbers as a list."""

    return list(_fibonacci_tuple(n))


def fibonacci_parity(sequence: Iterable[int]) -> List[Parity]:
    """Classify each Fibonacci number as ``"even"`` or ``"odd"``."""

    parity_labels: List[Parity] = []
    for value in sequence:
        parity_labels.append("even" if value % 2 == 0 else "odd")
    return parity_labels


def fibonacci_report(n: int) -> FibonacciReport:
    """Generate a parity report for the first *n* Fibonacci numbers."""

    sequence = fibonacci_python(n)
    parity = fibonacci_parity(sequence)
    return FibonacciReport(sequence=sequence, parity=parity)


def write_fibonacci_report(count: int, output: Path, indent: int = 2) -> Path:
    """Write the Fibonacci parity report to ``output`` as JSON."""

    report = fibonacci_report(count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the first N Fibonacci numbers with parity labels to a JSON file"
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25,
        help="Number of Fibonacci numbers to generate (default: 25)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON file that will be written",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation to use when serialising JSON (default: 2)",
    )
    return parser


def _run_cli(arguments: Sequence[str] | None = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        return write_fibonacci_report(args.count, args.output, indent=args.indent)
    except (TypeError, ValueError) as exc:  # pragma: no cover - argparse handles exit
        parser.error(str(exc))
        raise  # Unreachable but preserves typing for linters


def main() -> None:  # pragma: no cover - exercised via integration tests
    """Entry point when executing as ``python -m ...`` or ``python file.py``."""

    output_path = _run_cli()
    print(f"Fibonacci report written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - integration behaviour
    main()
