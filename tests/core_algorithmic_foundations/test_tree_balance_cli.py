"""Tests for the ``tree_balance`` CLI demonstration script."""

from __future__ import annotations

import importlib

import tree_balance


def test_cli_outputs_expected_demo_lines(capsys) -> None:
    """Ensure the CLI emits the documented demonstration output."""

    importlib.reload(tree_balance)
    tree_balance.main()
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert lines[0] == "Balanced tree balanced? Yes (expected: Yes)"
    assert lines[1:5] == [
        "1",
        "2 3",
        "4 5 6 7",
        "",
    ]
    assert lines[5] == "Skewed tree balanced? No (expected: No)"
    assert lines[6:10] == [
        "1",
        "2 ·",
        "3 · · ·",
        "4 · · · · · · ·",
    ]
