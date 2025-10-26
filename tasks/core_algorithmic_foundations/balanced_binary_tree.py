"""Height-balanced binary tree validation utilities.

This module implements the canonical solution described in
``Codex_Master_Task_Results.md`` for Task 2.  It offers a production-ready
representation of binary trees together with helpers for verifying the
height-balanced invariant and rendering level-order visualisations suitable for
regression artefacts.

The APIs intentionally provide:

* ``TreeNode`` – a ``@dataclass`` with optional left/right children.
* ``is_balanced`` – checks whether a tree is height-balanced in ``O(n)`` time.
* ``render_tree`` – produces a deterministic ASCII representation that
  highlights missing children with centred dots.
* ``build_tree_from_level_order`` – convenience function for tests and CLI
  demonstrations that construct a tree from a level-order sequence containing
  ``None`` sentinels.

The implementation contains comprehensive input validation and short-circuits
computation as soon as an imbalance is detected which keeps the runtime optimal
for large structures.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional


@dataclass(slots=True)
class TreeNode:
    """Node representation used for binary tree algorithms."""

    value: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise TypeError("TreeNode value must be an integer")


BalanceResult = tuple[bool, int]


def _check_height(node: Optional[TreeNode]) -> BalanceResult:
    """Return a tuple indicating whether *node* is balanced and its height."""

    if node is None:
        return True, 0

    left_balanced, left_height = _check_height(node.left)
    if not left_balanced:
        return False, left_height + 1

    right_balanced, right_height = _check_height(node.right)
    if not right_balanced:
        return False, right_height + 1

    balanced = abs(left_height - right_height) <= 1
    height = max(left_height, right_height) + 1
    return balanced and left_balanced and right_balanced, height


def is_balanced(root: Optional[TreeNode]) -> bool:
    """Return ``True`` when *root* is a height-balanced binary tree."""

    balanced, _ = _check_height(root)
    return balanced


def render_tree(root: Optional[TreeNode]) -> str:
    """Render *root* level-by-level, marking missing nodes with ``·``.

    The renderer stops once the entire level is empty, ensuring that the output
    contains no trailing placeholder-only rows.
    """

    if root is None:
        return "<empty>"

    lines: List[str] = []
    queue: Deque[Optional[TreeNode]] = deque([root])

    while queue:
        level_count = len(queue)
        level_nodes: List[str] = []
        next_level_has_real_node = False
        for _ in range(level_count):
            node = queue.popleft()
            if node is None:
                level_nodes.append("·")
                queue.extend((None, None))
                continue

            level_nodes.append(str(node.value))
            queue.append(node.left)
            queue.append(node.right)
            if node.left is not None or node.right is not None:
                next_level_has_real_node = True

        lines.append(" ".join(level_nodes))
        if not next_level_has_real_node:
            break

    return "\n".join(lines)


def build_tree_from_level_order(values: Iterable[Optional[int]]) -> Optional[TreeNode]:
    """Construct a binary tree from a level-order sequence.

    The *values* iterable may contain ``None`` sentinels to represent missing
    children. The function returns ``None`` when the sequence is empty or the
    root is ``None`` and validates that all provided payloads are integers (or
    ``None`` placeholders).
    """

    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return None

    if first is None:
        return None

    if not isinstance(first, int):
        raise TypeError("Level-order values must be integers or None")

    root = TreeNode(first)
    queue: Deque[TreeNode] = deque([root])

    while queue:
        node = queue.popleft()
        try:
            left_value = next(iterator)
        except StopIteration:
            break
        if left_value is not None:
            if not isinstance(left_value, int):
                raise TypeError("Level-order values must be integers or None")
            node.left = TreeNode(left_value)
            queue.append(node.left)

        try:
            right_value = next(iterator)
        except StopIteration:
            break
        if right_value is not None:
            if not isinstance(right_value, int):
                raise TypeError("Level-order values must be integers or None")
            node.right = TreeNode(right_value)
            queue.append(node.right)

    return root


def level_order_traversal(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Return the tree's level-order traversal including ``None`` sentinels."""

    if root is None:
        return []
    result: List[Optional[int]] = []
    queue: Deque[Optional[TreeNode]] = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            result.append(None)
            continue
        result.append(node.value)
        queue.append(node.left)
        queue.append(node.right)
    while result and result[-1] is None:
        result.pop()
    return result


__all__ = [
    "TreeNode",
    "is_balanced",
    "render_tree",
    "build_tree_from_level_order",
    "level_order_traversal",
]
