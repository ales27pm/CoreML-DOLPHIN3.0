"""Command line tool for Task 2 – Balanced Binary Tree Validator.

This script exposes a small demonstration harness around
``tasks.core_algorithmic_foundations.balanced_binary_tree`` so the behaviour
outlined in ``Codex_Master_Task_Results.md`` can be exercised directly from the
command line.  Running the module prints balance checks for both a perfectly
balanced tree and an intentionally skewed tree together with their level-order
ASCII renderings.

The script remains intentionally dependency-free: the heavy lifting occurs in
``balanced_binary_tree`` where comprehensive validation and helpers such as
``render_tree`` live.  Here we simply orchestrate pre-defined demo inputs and
emit human-readable status lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from tasks.core_algorithmic_foundations.balanced_binary_tree import (
    TreeNode,
    build_tree_from_level_order,
    is_balanced,
    render_tree,
)


@dataclass(frozen=True)
class DemoCase:
    """Container describing a tree example and its expected balance status."""

    name: str
    values: Iterable[Optional[int]]
    expected_balance: bool

    def build(self) -> Optional[TreeNode]:
        """Materialise the tree associated with this demo case."""

        return build_tree_from_level_order(self.values)


def _iter_demo_cases() -> Iterator[DemoCase]:
    """Yield the built-in demonstration cases."""

    yield DemoCase(
        name="Balanced",
        values=[1, 2, 3, 4, 5, 6, 7],
        expected_balance=True,
    )
    yield DemoCase(
        name="Skewed",
        values=[1, 2, None, 3, None, 4],
        expected_balance=False,
    )


def _format_report(case: DemoCase, tree: Optional[TreeNode]) -> List[str]:
    """Return formatted output lines for *case* and its *tree*."""

    if tree is None:
        return [f"{case.name} tree: <empty>"]

    actual_balance = is_balanced(tree)
    if actual_balance != case.expected_balance:
        raise RuntimeError(
            "Demo case expectation mismatch:"
            f" {case.name} expected {case.expected_balance}"
            f" but received {actual_balance}"
        )

    status = "Yes" if actual_balance else "No"
    expected = "Yes" if case.expected_balance else "No"
    header = f"{case.name} tree balanced? {status} (expected: {expected})"

    lines = [
        header,
        render_tree(tree),
    ]
    return lines


def main() -> None:
    """Execute the demonstration flow for all configured cases."""

    for case in _iter_demo_cases():
        tree = case.build()
        output_lines = _format_report(case, tree)
        for line in output_lines:
            print(line)
        print()  # Spacer between cases


if __name__ == "__main__":
    main()
