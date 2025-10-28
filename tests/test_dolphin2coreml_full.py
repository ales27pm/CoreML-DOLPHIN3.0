from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from dolphin2coreml_full import _cosine_similarity


def test_cosine_similarity_matches_expected() -> None:
    vector = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert _cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_norm() -> None:
    with pytest.raises(ValueError):
        _cosine_similarity(
            np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)
        )
