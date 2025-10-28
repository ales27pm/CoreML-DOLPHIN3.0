"""Mathematical helpers used across quantization tests."""
from __future__ import annotations

import numpy as np


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""

    lhs_flat = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lhs_norm = np.linalg.norm(lhs_flat)
    rhs_norm = np.linalg.norm(rhs_flat)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        raise ValueError("Cosine similarity undefined for zero-norm embeddings.")
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


__all__ = ["_cosine_similarity"]
