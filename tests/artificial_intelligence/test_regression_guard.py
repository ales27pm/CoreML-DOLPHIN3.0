from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tasks.artificial_intelligence.regression_guard import RegressionGuard


def test_record_persists_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshots.json"
    guard = RegressionGuard(snapshot_path)

    entry = guard.record("hello", [0.1, 0.2, 0.3])

    assert entry.digest
    loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert loaded == {
        "hello": {
            "digest": entry.digest,
            "response": [0.1, 0.2, 0.3],
        }
    }


def test_verify_accepts_identical_response(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshots.json"
    guard = RegressionGuard(snapshot_path)
    guard.record("prompt", [0.1, 0.2])

    guard.verify("prompt", [0.1, 0.2])


def test_verify_allows_small_mse(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshots.json"
    guard = RegressionGuard(snapshot_path, tolerance=1e-3)
    guard.record("prompt", [0.5, 0.5, 0.5])

    guard.verify("prompt", [0.5, 0.501, 0.499])


def test_verify_rejects_large_mse(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshots.json"
    guard = RegressionGuard(snapshot_path, tolerance=1e-4)
    guard.record("prompt", [0.1, 0.1, 0.1])

    with pytest.raises(ValueError):
        guard.verify("prompt", [0.2, 0.2, 0.2])


def test_verify_requires_existing_snapshot(tmp_path: Path) -> None:
    guard = RegressionGuard(tmp_path / "snapshots.json")

    with pytest.raises(KeyError):
        guard.verify("missing", [0.1])


def test_verify_detects_length_mismatch(tmp_path: Path) -> None:
    guard = RegressionGuard(tmp_path / "snapshots.json")
    guard.record("prompt", [0.0, 0.0])

    with pytest.raises(ValueError):
        guard.verify("prompt", [0.0])


def test_verify_emits_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    snapshot_path = tmp_path / "snapshots.json"
    guard = RegressionGuard(snapshot_path)
    guard.record("prompt", [0.25, 0.75])

    with caplog.at_level(logging.INFO):
        guard.verify("prompt", [0.25, 0.75])

    assert any(
        record.levelno == logging.INFO
        and "Regression guard verified" in record.getMessage()
        for record in caplog.records
    )
