"""Unit tests for :mod:`tasks.core_algorithmic_foundations.longest_palindromic_substring`."""

from __future__ import annotations

import unittest

from tasks.core_algorithmic_foundations.longest_palindromic_substring import (
    PalindromeResult,
    longest_palindromic_substring,
)


class LongestPalindromicSubstringTests(unittest.TestCase):
    """Verify the expand-around-center palindrome implementation."""

    def test_examples(self) -> None:
        result = longest_palindromic_substring("babad")
        self.assertIn(result.value, {"bab", "aba"})
        self.assertEqual(result.length, len(result.value))

        self.assertEqual(longest_palindromic_substring("cbbd").value, "bb")
        self.assertEqual(
            longest_palindromic_substring("forgeeksskeegfor").value,
            "geeksskeeg",
        )

    def test_single_character_and_empty(self) -> None:
        self.assertEqual(longest_palindromic_substring("a").value, "a")
        self.assertEqual(longest_palindromic_substring("").value, "")

    def test_all_unique_characters(self) -> None:
        self.assertEqual(longest_palindromic_substring("abcd").value, "a")

    def test_full_string_palindrome(self) -> None:
        self.assertEqual(longest_palindromic_substring("racecar").value, "racecar")

    def test_multiple_centers(self) -> None:
        self.assertEqual(
            longest_palindromic_substring("abacdfgdcaba").value,
            "aba",
        )

    def test_invalid_input_type(self) -> None:
        with self.assertRaises(TypeError):
            longest_palindromic_substring(123)  # type: ignore[arg-type]

    def test_result_metadata(self) -> None:
        text = "anana"
        result = longest_palindromic_substring(text)
        self.assertIsInstance(result, PalindromeResult)
        self.assertEqual(result.start, 0)
        self.assertEqual(result.end, len(text) - 1)
        self.assertEqual(result.value, text)


if __name__ == "__main__":  # pragma: no cover - manual test execution
    unittest.main()
