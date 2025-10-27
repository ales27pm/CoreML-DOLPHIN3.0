"""Trie-based autocomplete implementation.

This module provides the production-ready implementation for
``Codex_Master_Task_Results`` Task 3.  It exposes a ``Trie`` structure with
deterministic JSON serialisation, ergonomic helpers for prefix search, and a
benchmark utility for instrumentation during regression testing.

The implementation favours correctness and debuggability:

* Inputs are validated to guard against silent coercion of non-string payloads.
* Serialisation produces a compact representation that can be persisted or
  diffed in tests.
* Metadata (word count and node count) is tracked eagerly so integrations can
  surface diagnostics without re-traversing the structure.

The API mirrors the historical specification exactly and is accompanied by unit
tests located in ``tests/core_algorithmic_foundations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence
import time

__all__ = [
    "TrieNode",
    "Trie",
    "benchmark",
]


@dataclass(slots=True)
class TrieNode:
    """A node inside the trie data structure."""

    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False

    def __post_init__(self) -> None:
        for key, child in self.children.items():
            if not isinstance(key, str) or len(key) != 1:
                raise TypeError(
                    "TrieNode children must be keyed by single-character strings"
                )
            if not isinstance(child, TrieNode):
                raise TypeError("TrieNode children must be TrieNode instances")
        if not isinstance(self.is_terminal, bool):
            raise TypeError("TrieNode.is_terminal must be a boolean")


class Trie:
    """Trie supporting insertion, lookup and JSON round-tripping."""

    __slots__ = ("root", "_size", "_node_count")

    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        self.root = TrieNode()
        self._size = 0
        self._node_count = 1
        if words is not None:
            self.bulk_insert(words)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def insert(self, word: str) -> None:
        """Insert *word* into the trie.

        Duplicate insertions are ignored. Non-string inputs raise ``TypeError``
        to surface invalid usage as early as possible.
        """

        normalized = self._normalize_word(word)
        node = self.root
        for char in normalized:
            next_node = node.children.get(char)
            if next_node is None:
                next_node = TrieNode()
                node.children[char] = next_node
                self._node_count += 1
            node = next_node
        if not node.is_terminal:
            node.is_terminal = True
            self._size += 1

    def bulk_insert(self, words: Iterable[str]) -> None:
        """Insert multiple *words*.

        The iterable is eagerly consumed to surface TypeErrors deterministically
        even when the caller provides a generator that would otherwise short
        circuit on the first invalid element.
        """

        for word in list(words):
            self.insert(word)

    def search(self, word: str) -> bool:
        """Return ``True`` if *word* exists in the trie."""

        node = self._traverse(self._normalize_word(word))
        return bool(node and node.is_terminal)

    def starts_with(self, prefix: str) -> bool:
        """Return ``True`` when any word shares the supplied *prefix*."""

        return self._traverse(self._normalize_word(prefix)) is not None

    def __contains__(self, word: object) -> bool:  # pragma: no cover - thin wrapper
        return isinstance(word, str) and self.search(word)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    @property
    def node_count(self) -> int:
        """Total number of nodes currently allocated."""

        return self._node_count

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Return a deterministic JSON representation of the trie."""

        serialised = self._serialize_node(self.root)
        return json.dumps(serialised, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "Trie":
        """Restore a trie from *payload* produced by :meth:`to_json`."""

        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("payload is not valid JSON") from exc

        trie = cls()
        trie.root = cls._deserialize_node(data)
        trie._size, trie._node_count = trie._calculate_metadata(trie.root)
        return trie

    @staticmethod
    def _serialize_node(node: TrieNode) -> Dict[str, object]:
        return {
            "terminal": node.is_terminal,
            "children": {
                char: Trie._serialize_node(child)
                for char, child in sorted(node.children.items())
            },
        }

    @staticmethod
    def _deserialize_node(data: MutableMapping[str, object]) -> TrieNode:
        if not isinstance(data, MutableMapping):
            raise TypeError("payload structure is invalid")
        terminal = data.get("terminal")
        if not isinstance(terminal, bool):
            raise TypeError("payload structure is invalid")
        children_payload = data.get("children", {})
        if not isinstance(children_payload, MutableMapping):
            raise TypeError("payload structure is invalid")
        children: Dict[str, TrieNode] = {}
        for char, child_payload in children_payload.items():
            if not isinstance(char, str) or len(char) != 1:
                raise TypeError(
                    "payload structure is invalid: child keys must be single-character strings"
                )
            child_node = Trie._deserialize_node(child_payload)
            if not isinstance(child_node, TrieNode):  # pragma: no cover - defensive
                raise TypeError(
                    "payload structure is invalid: child values must describe TrieNodes"
                )
            children[char] = child_node
        return TrieNode(children=children, is_terminal=terminal)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_word(word: str) -> str:
        if not isinstance(word, str):
            raise TypeError("word must be a string")
        return word

    def _traverse(self, fragment: str) -> Optional[TrieNode]:
        current: Optional[TrieNode] = self.root
        for char in fragment:
            if current is None:
                return None
            current = current.children.get(char)
        return current

    def _calculate_metadata(self, node: TrieNode) -> tuple[int, int]:
        size = 1 if node.is_terminal else 0
        nodes = 1
        for child in node.children.values():
            child_size, child_nodes = self._calculate_metadata(child)
            size += child_size
            nodes += child_nodes
        return size, nodes


def benchmark(trie: Trie, words: Iterable[str]) -> float:
    """Return the average lookup latency for *words* in seconds.

    The function consumes the iterable exactly once and returns ``0.0`` when no
    words are supplied. Time measurement uses :func:`time.perf_counter` for
    high-resolution timing.
    """

    word_list = list(words) if not isinstance(words, Sequence) else list(words)
    if not word_list:
        return 0.0

    start = time.perf_counter()
    for word in word_list:
        trie.search(word)
    elapsed = time.perf_counter() - start
    return elapsed / len(word_list)


def iter_words(trie: Trie) -> Iterator[str]:  # pragma: no cover - convenience utility
    """Yield all words stored in *trie* in lexicographical order."""

    def _walk(node: TrieNode, prefix: List[str]) -> Iterator[str]:
        if node.is_terminal:
            yield "".join(prefix)
        for char in sorted(node.children):
            prefix.append(char)
            yield from _walk(node.children[char], prefix)
            prefix.pop()

    yield from _walk(trie.root, [])


if __name__ == "__main__":  # pragma: no cover - CLI demonstration
    SAMPLE_WORDS = ("hello", "helium", "help", "hero", "her")
    trie = Trie(SAMPLE_WORDS)
    encoded = trie.to_json()
    restored = Trie.from_json(encoded)
    assert restored.search("hero")
    assert restored.starts_with("he")

    import random
    import string

    rng = random.Random(13)
    dataset = ["".join(rng.choices(string.ascii_lowercase, k=7)) for _ in range(10_000)]
    trie.bulk_insert(dataset)
    sample = rng.sample(dataset, 1_000)
    median_time = benchmark(trie, sample)
    print(f"Average lookup latency across 1,000 samples: {median_time * 1_000:.3f} ms")
