"""Tests for the trie autocomplete implementation (Task 3)."""

from __future__ import annotations

import json

import pytest

from tasks.core_algorithmic_foundations.trie_autocomplete import (
    Trie,
    TrieNode,
    benchmark,
)


def test_insert_and_search_round_trip() -> None:
    trie = Trie()
    words = ["hello", "helium", "hero", "help"]
    for word in words:
        trie.insert(word)

    for word in words:
        assert trie.search(word)

    assert not trie.search("heron")
    assert len(trie) == len(words)


def test_prefix_detection() -> None:
    trie = Trie(["core", "coral", "corner"])
    assert trie.starts_with("cor")
    assert not trie.starts_with("xyz")


def test_serialization_round_trip_preserves_structure() -> None:
    trie = Trie(["apple", "app", "apply"])
    payload = trie.to_json()
    restored = Trie.from_json(payload)

    assert all(restored.search(word) for word in ("apple", "app", "apply"))
    assert restored.node_count == trie.node_count
    assert len(restored) == len(trie)


def test_benchmark_handles_empty_iterable() -> None:
    trie = Trie(["network"])
    assert benchmark(trie, []) == 0.0


def test_benchmark_measures_average_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    trie = Trie(["alpha", "beta", "gamma"])

    timings: list[float] = [10.0, 11.0]

    def fake_perf_counter() -> float:
        return timings.pop(0)

    monkeypatch.setattr("tasks.core_algorithmic_foundations.trie_autocomplete.time.perf_counter", fake_perf_counter)

    duration = benchmark(trie, ["alpha", "beta"])
    assert duration == pytest.approx(0.5)


def test_insert_rejects_non_string_inputs() -> None:
    trie = Trie()
    with pytest.raises(TypeError):
        trie.insert(123)  # type: ignore[arg-type]


def test_from_json_validates_payload() -> None:
    with pytest.raises(TypeError):
        Trie.from_json(123)  # type: ignore[arg-type]

    invalid_payload = json.dumps({"terminal": "yes", "children": {}})
    with pytest.raises(TypeError):
        Trie.from_json(invalid_payload)


def test_trie_node_validates_child_keys() -> None:
    with pytest.raises(TypeError):
        TrieNode(children={1: TrieNode()})  # type: ignore[arg-type]

