from __future__ import annotations

import pytest

from tasks.core_algorithmic_foundations.balanced_binary_tree import (
    TreeNode,
    build_tree_from_level_order,
    is_balanced,
    level_order_traversal,
    render_tree,
)


def test_is_balanced_detects_balanced_tree() -> None:
    root = TreeNode(
        1,
        left=TreeNode(2, TreeNode(4), TreeNode(5)),
        right=TreeNode(3, TreeNode(6), TreeNode(7)),
    )
    assert is_balanced(root)


def test_is_balanced_detects_unbalanced_tree() -> None:
    root = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4))))
    assert not is_balanced(root)


def test_render_tree_renders_structure_with_placeholders() -> None:
    root = TreeNode(1, TreeNode(2, right=TreeNode(4)), TreeNode(3))
    expected = "\n".join(["1", "2 3", "· 4 · ·"])
    assert render_tree(root) == expected


def test_render_tree_empty_tree() -> None:
    assert render_tree(None) == "<empty>"


def test_render_tree_trims_placeholder_only_levels() -> None:
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    rendered = render_tree(root)
    assert rendered == "\n".join(["1", "2 3"])


def test_build_tree_from_level_order_roundtrip() -> None:
    balanced_values = [1, 2, 3, None, 5, None, 7]
    balanced_root = build_tree_from_level_order(balanced_values)
    assert level_order_traversal(balanced_root) == balanced_values
    assert is_balanced(balanced_root) is True

    unbalanced_values = [1, 2, None, 3, None, 4]
    unbalanced_root = build_tree_from_level_order(unbalanced_values)
    assert level_order_traversal(unbalanced_root) == unbalanced_values
    assert is_balanced(unbalanced_root) is False


def test_tree_node_rejects_non_integer_values() -> None:
    with pytest.raises(TypeError):
        TreeNode("invalid")  # type: ignore[arg-type]


def test_build_tree_from_level_order_rejects_non_integer_payloads() -> None:
    with pytest.raises(TypeError):
        build_tree_from_level_order([1, "two"])  # type: ignore[list-item]
