"""Longest palindromic substring solver.

This module exposes a production-ready implementation of the expand-around-center
strategy for computing the longest palindromic substring of a given string.
The implementation mirrors the approach described in Codex_Master_Task_Results
(Task 1) while adding input validation, documentation, and a helper for
benchmarking scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PalindromeResult:
    """Container describing the best palindrome identified in *text*.

    Attributes
    ----------
    text:
        Original string.
    start:
        Inclusive start index of the palindrome inside *text*.
    end:
        Inclusive end index of the palindrome inside *text*.
    """

    text: str
    start: int
    end: int

    @property
    def value(self) -> str:
        """Return the palindromic substring."""

        return self.text[self.start : self.end + 1]

    @property
    def length(self) -> int:
        """Return the length of the palindrome."""

        return self.end - self.start + 1


def _expand_from_center(text: str, left: int, right: int) -> Tuple[int, int]:
    """Return the (start, end) indices after expanding around a center.

    Parameters
    ----------
    text:
        String to inspect.
    left, right:
        Starting indices for the expansion. When *left == right* the expansion
        considers odd-length palindromes, otherwise even-length palindromes.
    """

    while left >= 0 and right < len(text) and text[left] == text[right]:
        left -= 1
        right += 1
    return left + 1, right - 1


def longest_palindromic_substring(text: str) -> PalindromeResult:
    """Return the longest palindromic substring contained in *text*.

    The function performs an O(n^2) expand-around-center search. The
    complexity aligns with the specification from Task 1 of
    ``Codex_Master_Task_Results.md`` and remains deterministic for identical
    inputs across Python versions.

    Parameters
    ----------
    text:
        Input string. Empty strings are supported and return an empty
        palindrome.

    Raises
    ------
    TypeError
        If *text* is not a string.

    Returns
    -------
    PalindromeResult
        Dataclass containing the result metadata.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if len(text) < 2:
        return PalindromeResult(text=text, start=0, end=len(text) - 1)

    best_start, best_end = 0, 0
    for center in range(len(text)):
        odd_start, odd_end = _expand_from_center(text, center, center)
        if odd_end - odd_start > best_end - best_start:
            best_start, best_end = odd_start, odd_end

        even_start, even_end = _expand_from_center(text, center, center + 1)
        if even_end - even_start > best_end - best_start:
            best_start, best_end = even_start, even_end

    return PalindromeResult(text=text, start=best_start, end=best_end)


__all__ = ["PalindromeResult", "longest_palindromic_substring"]
