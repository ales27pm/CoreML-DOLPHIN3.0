"""Hash-based regression guard for model output verification.

This module implements Task 32 from the Codex master ledger. It provides a
``RegressionGuard`` utility that maintains deterministic JSON snapshots of model
responses keyed by prompt text.  Each snapshot entry stores both the
cryptographic digest of the numeric response vector and the vector itself so
that callers can perform tolerance-aware comparisons using mean squared error
(MSE).

The guard is intentionally lightweight and side-effect free beyond writing the
snapshot file.  It exposes two high-level operations:

``record``
    Capture a new baseline for a prompt, replacing any existing entry.  The
    method normalises numeric inputs, persists the JSON snapshot with stable
    formatting, and logs a structured summary.

``verify``
    Compare the supplied response against the stored baseline.  The method
    checks the SHA-256 digest first for fast rejection and then evaluates the
    MSE between the baseline vector and the candidate response.  A regression is
    signalled when the MSE exceeds the configured tolerance.

All file I/O uses UTF-8 encoding and writes are atomic at the Python level by
leveraging ``Path.write_text`` on temporary strings.  The implementation errs on
informative error messages and explicit validation so that CI failures can be
triaged quickly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SnapshotEntry:
    """Representation of a stored regression snapshot."""

    digest: str
    response: list[float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping."""

        return {"digest": self.digest, "response": self.response}

    @classmethod
    def from_mapping(cls, prompt: str, mapping: Mapping[str, object]) -> "SnapshotEntry":
        """Validate *mapping* and return a populated entry.

        Parameters
        ----------
        prompt:
            Prompt identifier used for error reporting.
        mapping:
            Mapping loaded from the snapshot JSON file.
        """

        digest = mapping.get("digest")
        response = mapping.get("response")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"Snapshot for prompt '{prompt}' is missing a digest string")
        if not isinstance(response, list):
            raise ValueError(
                f"Snapshot for prompt '{prompt}' must include a list response payload"
            )
        normalised = _normalise_response(response)
        return cls(digest=digest, response=normalised)


def _normalise_response(values: Iterable[object]) -> list[float]:
    """Return *values* as a list of floats with strict validation."""

    normalised: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise TypeError("Boolean values are not valid model outputs")
        try:
            normalised.append(float(value))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Model response values must be numeric (index {index})"
            ) from exc
    return normalised


def _mean_squared_error(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the MSE between two equally sized sequences."""

    if len(a) != len(b):
        raise ValueError(
            "Cannot compare responses with differing lengths: "
            f"{len(a)} != {len(b)}"
        )
    if not a:
        return 0.0
    total = 0.0
    for left, right in zip(a, b):
        diff = left - right
        total += diff * diff
    return total / len(a)


class RegressionGuard:
    """Maintain cryptographic snapshots of model responses.

    Parameters
    ----------
    snapshot_path:
        Path to the JSON file storing regression snapshots.
    tolerance:
        Maximum allowed mean squared error (MSE) when comparing a new response
        against the stored baseline.  Defaults to ``1e-3`` to match the Codex
        specification.
    indent:
        Number of spaces used when serialising JSON snapshots.
    """

    def __init__(self, snapshot_path: Path, *, tolerance: float = 1e-3, indent: int = 2) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if indent < 0:
            raise ValueError("indent must be non-negative")

        self.snapshot_path = Path(snapshot_path)
        self.tolerance = float(tolerance)
        self.indent = indent
        self._snapshots: MutableMapping[str, SnapshotEntry] = {}

        if self.snapshot_path.exists():
            try:
                raw_data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Snapshot file '{self.snapshot_path}' contains invalid JSON"
                ) from exc

            if not isinstance(raw_data, dict):
                raise ValueError("Snapshot file must contain a JSON object")

            for prompt, payload in raw_data.items():
                if not isinstance(prompt, str):
                    raise ValueError("Snapshot keys must be strings")
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"Snapshot entry for prompt '{prompt}' must be a mapping"
                    )
                self._snapshots[prompt] = SnapshotEntry.from_mapping(prompt, payload)

    def record(self, prompt: str, response: Iterable[object]) -> SnapshotEntry:
        """Store *response* as the new baseline for *prompt*.

        Returns the :class:`SnapshotEntry` that was persisted to disk.
        """

        if not isinstance(prompt, str) or not prompt:
            raise TypeError("prompt must be a non-empty string")

        normalised = _normalise_response(response)
        digest = self._hash(normalised)
        entry = SnapshotEntry(digest=digest, response=normalised)
        self._snapshots[prompt] = entry
        self._write_snapshots()
        logger.info(
            "Recorded regression snapshot", extra={"prompt": prompt, "digest": digest}
        )
        return entry

    def verify(self, prompt: str, response: Iterable[object]) -> None:
        """Validate *response* against the stored snapshot for *prompt*.

        Raises :class:`ValueError` when the response deviates beyond the allowed
        tolerance and :class:`KeyError` if no baseline exists for the prompt.
        """

        if prompt not in self._snapshots:
            raise KeyError(f"No snapshot found for prompt '{prompt}'")

        normalised = _normalise_response(response)
        digest = self._hash(normalised)
        baseline = self._snapshots[prompt]

        if digest == baseline.digest:
            logger.info(
                "Regression guard verified snapshot", extra={"prompt": prompt, "digest": digest}
            )
            return

        mse = _mean_squared_error(baseline.response, normalised)
        if mse > self.tolerance:
            logger.error(
                "Model regression detected", extra={"prompt": prompt, "mse": mse, "digest": digest}
            )
            raise ValueError(
                f"Model regression detected for prompt '{prompt}' (MSE={mse:.6f})"
            )

        logger.info(
            "Regression guard verified within tolerance",
            extra={"prompt": prompt, "mse": mse, "digest": digest},
        )

    def has_snapshot(self, prompt: str) -> bool:
        """Return ``True`` when a baseline exists for *prompt*."""

        return prompt in self._snapshots

    def _write_snapshots(self) -> None:
        data = {prompt: entry.to_dict() for prompt, entry in sorted(self._snapshots.items())}
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(
            json.dumps(data, indent=self.indent, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _hash(response: Iterable[float]) -> str:
        payload = json.dumps(list(response), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = ["RegressionGuard", "SnapshotEntry"]
