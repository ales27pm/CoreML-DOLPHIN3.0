from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from tools.session_finalize import (
    DocumentPlan,
    FinalizationReport,
    MarkdownUpdateError,
    RoadmapMaintainer,
    SessionFinalizer,
    SessionRecord,
    discover_readmes,
    load_documentation_manifest,
    main,
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
    assert maintainer.refresh(record) is True

    content = roadmap.read_text(encoding="utf-8")
    assert "# Delivery Roadmap" in content
    assert "_Refreshed automatically: 2025-01-01T00:00:00+00:00_" in content
    assert "## Session Pulse" in content
    assert "- **Key Notes:**" in content
    assert "## Automation Footnotes" in content
    assert "| Task | Status |" in content


def test_update_markdown_log_dry_run(tmp_path: Path) -> None:
    record = SessionRecord(
        session_name="Dry Run",
        summary="Verification",
        notes=(),
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    path = tmp_path / "README.md"
    header = "## Session Updates"

    assert update_markdown_log(path, record, header, dry_run=True) is True
    assert not path.exists()


def test_update_markdown_log_enforces_limit(tmp_path: Path) -> None:
    header = "## Session Timeline"
    path = tmp_path / "README.md"
    record1 = SessionRecord(
        session_name="Alpha",
        summary="Did alpha",
        notes=(),
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    record2 = SessionRecord(
        session_name="Beta",
        summary="Did beta",
        notes=(),
        timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    record3 = SessionRecord(
        session_name="Gamma",
        summary="Did gamma",
        notes=(),
        timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc),
    )

    assert update_markdown_log(path, record1, header, limit=2) is True
    assert update_markdown_log(path, record2, header, limit=2) is True
    assert update_markdown_log(path, record3, header, limit=2) is True

    content = path.read_text(encoding="utf-8")
    assert content.count("<!-- session-log:") == 2
    assert "Gamma" in content and "Beta" in content and "Alpha" not in content


def test_session_finalizer_reports_changes(tmp_path: Path, monkeypatch) -> None:
    record = SessionRecord(
        session_name="Session 101",
        summary="Expanded coverage",
        notes=("Checked report",),
        timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    codex = tmp_path / "Codex_Master_Task_Results.md"
    codex.write_text(
        "\n".join([
            "# Codex",
            "",
            "## Status Dashboard",
            "",
            "| Task | Status |",
            "| ---- | ------ |",
            "| Hardening | 🔄 In Progress |",
        ]),
        encoding="utf-8",
    )

    roadmap = RoadmapMaintainer(codex_path=codex, roadmap_path=tmp_path / "docs" / "ROADMAP.md")

    calls: list[tuple[Path, bool]] = []

    def fake_sync(*, repo_root: Path, dry_run: bool) -> bool:
        calls.append((repo_root, dry_run))
        return True

    monkeypatch.setattr("tools.session_finalize.synchronize_agents", fake_sync)

    document = DocumentPlan(path=tmp_path / "README.md", header="## Session Updates")
    finalizer = SessionFinalizer(
        repo_root=tmp_path,
        record=record,
        documents=[document],
        roadmap=roadmap,
        sync_agents=True,
        dry_run=False,
    )

    report = finalizer.finalize()

    assert isinstance(report, FinalizationReport)
    assert report.agents_changed is True
    assert report.roadmap_changed is True
    assert report.documents_changed == (document.path,)
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "docs" / "ROADMAP.md").exists()
    assert calls == [(tmp_path, False)]


def test_roadmap_maintainer_reports_no_change(tmp_path: Path) -> None:
    codex = tmp_path / "Codex_Master_Task_Results.md"
    codex.write_text(
        "\n".join([
            "# Codex",
            "",
            "## Status Dashboard",
            "",
            "| Task | Status |",
        ]),
        encoding="utf-8",
    )

    roadmap_path = tmp_path / "docs" / "ROADMAP.md"
    record = SessionRecord(
        session_name="Repeat",
        summary="Initial",
        notes=(),
        timestamp=datetime(2025, 1, 3, tzinfo=timezone.utc),
    )

    maintainer = RoadmapMaintainer(codex_path=codex, roadmap_path=roadmap_path)
    assert maintainer.refresh(record) is True
    assert maintainer.refresh(record) is False


def test_load_documentation_manifest_reads_templates(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs" / "documentation_manifest.json"
    templates_dir = tmp_path / "docs" / "templates"
    templates_dir.mkdir(parents=True)
    template_file = templates_dir / "example.md"
    template_file.write_text("# Template\n", encoding="utf-8")

    manifest_path.write_text(
        (
            "{\n"
            "  \"version\": 1,\n"
            "  \"documents\": [\n"
            "    {\n"
            "      \"path\": \"README.md\",\n"
            "      \"header\": \"## Session Timeline\",\n"
            "      \"limit\": 2\n"
            "    },\n"
            "    {\n"
            "      \"path\": \"docs/history/SESSION_LOG.md\",\n"
            "      \"header\": \"## Session Log\",\n"
            "      \"ensure_exists\": true,\n"
            "      \"template_path\": \"docs/templates/example.md\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    manifest = load_documentation_manifest(tmp_path, manifest_path)
    assert manifest is not None
    assert manifest.version == 1
    assert len(manifest.documents) == 2
    readme_plan, log_plan = manifest.documents
    assert readme_plan.limit == 2
    assert readme_plan.ensure_exists is False
    assert log_plan.ensure_exists is True
    assert log_plan.template == "# Template\n"


def test_load_documentation_manifest_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs" / "documentation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{ invalid", encoding="utf-8")

    with pytest.raises(MarkdownUpdateError):
        load_documentation_manifest(tmp_path, manifest_path)


def test_load_documentation_manifest_missing_template(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs" / "documentation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": "docs/history/SESSION_LOG.md",
                        "header": "## Session Log",
                        "ensure_exists": True,
                        "template_path": "docs/templates/missing.md",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MarkdownUpdateError):
        load_documentation_manifest(tmp_path, manifest_path)


def test_manifest_codex_override_updates_plan(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs" / "documentation_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "codex_path": "Codex_Master_Task_Results.md",
                "roadmap_path": "docs/ROADMAP.md",
                "documents": [
                    {
                        "path": "Codex_Master_Task_Results.md",
                        "header": "## Session Journal",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    default_codex = tmp_path / "Codex_Master_Task_Results.md"
    default_codex.write_text(
        "# Codex\n\n## Status Dashboard\n\n| Task | Status |\n",
        encoding="utf-8",
    )

    alternate_codex = tmp_path / "AltCodex.md"
    alternate_codex.write_text(
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
                "## Session Journal",
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            "--session-name",
            "Override Session",
            "--summary",
            "Testing manifest override",
            "--codex-path",
            "AltCodex.md",
            "--manifest-path",
            str(manifest_path),
            "--repo-root",
            str(tmp_path),
            "--skip-agent-sync",
        ]
    )

    default_content = default_codex.read_text(encoding="utf-8")
    override_content = alternate_codex.read_text(encoding="utf-8")

    assert "Override Session" not in default_content
    assert "Override Session" in override_content
    assert "<!-- session-log:" in override_content
