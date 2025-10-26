from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tools.session_finalize import (
    RoadmapMaintainer,
    SessionRecord,
    discover_readmes,
    slugify,
    update_markdown_log,
)


def test_slugify_strips_invalid_characters() -> None:
    assert slugify("Session 1: Ready!") == "session-1-ready"


def test_update_markdown_adds_header_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    record = SessionRecord(
        session_name="Session 42",
        summary="Wrapped work",
        notes=("Synced agents",),
        timestamp=datetime(2024, 5, 25, tzinfo=timezone.utc),
    )

    assert update_markdown_log(path, record, "## Session Updates") is True
    content = path.read_text(encoding="utf-8")
    assert "## Session Updates" in content
    assert "Synced agents" in content


def test_update_markdown_is_idempotent(tmp_path: Path) -> None:
    record = SessionRecord(
        session_name="Session Alpha",
        summary="Did things",
        notes=(),
        timestamp=datetime(2024, 5, 25, tzinfo=timezone.utc),
    )
    path = tmp_path / "README.md"
    header = "## Session Updates"
    assert update_markdown_log(path, record, header) is True
    first = path.read_text(encoding="utf-8")
    assert update_markdown_log(path, record, header) is False
    assert path.read_text(encoding="utf-8") == first


def test_discover_readmes_skips_vendor(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "README.md").write_text("ignore", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "README.md").write_text("ignore", encoding="utf-8")
    target = tmp_path / "README.md"
    target.write_text("visible", encoding="utf-8")
    found = discover_readmes(tmp_path)
    assert found == [target]


def test_roadmap_maintainer_renders_dashboard(tmp_path: Path) -> None:
    codex = tmp_path / "Codex_Master_Task_Results.md"
    codex.write_text(
        "\n".join(
            [
                "# Codex",
                "",
                "## Status Dashboard",
                "",
                "| Task | Status |",
                "| ---- | ------ |",
                "| 1 | ✅ Done |",
                "",
                "## Footer",
            ]
        ),
        encoding="utf-8",
    )
    roadmap = tmp_path / "docs" / "ROADMAP.md"
    record = SessionRecord(
        session_name="Session 99",
        summary="Validated roadmap automation",
        notes=("Synced",),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    maintainer = RoadmapMaintainer(codex_path=codex, roadmap_path=roadmap)
    maintainer.refresh(record)

    content = roadmap.read_text(encoding="utf-8")
    assert "Session 99" in content
    assert "| Task | Status |" in content
