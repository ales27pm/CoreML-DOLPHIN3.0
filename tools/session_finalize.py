"""Automate end-of-session synchronization for documentation and agent files."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.manage_agents import synchronize_agents

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionRecord:
    """Represents a single session summary recorded in repository documents."""

    session_name: str
    summary: str
    notes: Sequence[str]
    timestamp: datetime

    @property
    def marker(self) -> str:
        """Return a stable marker for detecting existing log entries."""

        session_id = slugify(self.session_name)
        return f"<!-- session-log:{session_id}:{self.timestamp.isoformat()} -->"


@dataclass(frozen=True)
class DocumentPlan:
    """Describe a markdown document and header target for session updates."""

    path: Path
    header: str
    limit: int | None = None
    ensure_exists: bool = False
    template: str | None = None
    tags: frozenset[str] = frozenset()

    @property
    def is_readme(self) -> bool:
        """Return ``True`` when the plan targets a README file."""

        return self.path.name.lower() == "readme.md"


@dataclass(frozen=True)
class DocumentUpdateResult:
    """Represents the outcome of a document update during finalization."""

    path: Path
    changed: bool


@dataclass(frozen=True)
class FinalizationReport:
    """Aggregate report emitted after a session finalization run."""

    agents_changed: bool | None
    documents: Sequence[DocumentUpdateResult]
    roadmap_changed: bool | None

    @property
    def documents_changed(self) -> Sequence[Path]:
        """Return paths for documents that required mutation."""

        return tuple(result.path for result in self.documents if result.changed)


class MarkdownUpdateError(RuntimeError):
    """Raised when markdown content cannot be updated as requested."""


@dataclass(frozen=True)
class DocumentationManifest:
    """Configuration describing documentation update behavior."""

    version: int
    documents: Sequence[DocumentPlan]
    codex_path: Path
    roadmap_path: Path
    manifest_path: Path


def slugify(text: str) -> str:
    """Generate a filesystem-safe slug from ``text`` for use in markers."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "session"


def discover_readmes(root: Path) -> List[Path]:
    """Return README files beneath ``root`` while skipping vendor directories."""

    excluded = {".git", "node_modules", "__pycache__", "build", "dist", ".venv", ".pytest_cache"}
    paths: List[Path] = []
    for readme in root.rglob("README.md"):
        if any(part in excluded for part in readme.parts):
            continue
        paths.append(readme)
    return sorted(paths)


def _render_entry(record: SessionRecord) -> str:
    notes_block = "\n".join(f"- {note}" for note in record.notes) if record.notes else "- (none)"
    entry_lines = [
        record.marker,
        f"### {record.session_name} ({record.timestamp.isoformat()})",
        "",
        f"**Summary:** {record.summary}",
        "",
        "**Notes:**",
        notes_block,
        "",
    ]
    return "\n".join(entry_lines)


def _update_markdown_content(
    content: str,
    header: str,
    entry: str,
    marker: str,
    *,
    limit: int | None = None,
) -> tuple[str, bool]:
    """Insert ``entry`` below ``header`` when ``marker`` is not already present."""

    if marker in content:
        return content, False

    header_pattern = re.compile(rf"(?m)^\s*{re.escape(header)}\s*$")
    match = header_pattern.search(content)

    entry_block = f"\n\n{entry.strip()}\n"

    if not match:
        base = content.rstrip()
        prefix = "\n\n" if base else ""
        combined = base + f"{prefix}{header}{entry_block}\n"
        updated = combined
        if limit is not None and limit >= 0:
            updated = _limit_entries_within_header(updated, header, limit)
        return updated, True

    insert_pos = match.end()
    subsequent = content[insert_pos:]
    next_header_match = re.search(r"(?m)^##\s+", subsequent)
    if next_header_match:
        absolute_pos = insert_pos + next_header_match.start()
        updated = content[:absolute_pos] + entry_block + content[absolute_pos:]
    else:
        updated = content.rstrip() + entry_block + "\n"

    if limit is not None and limit >= 0:
        updated = _limit_entries_within_header(updated, header, limit)
    return updated, True


def _limit_entries_within_header(content: str, header: str, limit: int) -> str:
    """Ensure only the ``limit`` newest session entries remain under ``header``."""

    if limit <= 0:
        return content

    header_pattern = re.compile(rf"(?m)^\s*{re.escape(header)}\s*$")
    match = header_pattern.search(content)
    if not match:
        return content

    insert_pos = match.end()
    subsequent = content[insert_pos:]
    next_header_match = re.search(r"(?m)^##\s+", subsequent)
    section_end = (
        insert_pos + next_header_match.start()
        if next_header_match
        else len(content)
    )

    section = content[insert_pos:section_end]
    trimmed = _trim_section_entries(section, limit)
    if trimmed == section:
        return content
    return content[:insert_pos] + trimmed + content[section_end:]


def _trim_section_entries(section: str, limit: int) -> str:
    """Trim session log entries within ``section`` to ``limit`` occurrences."""

    marker = "<!-- session-log:"
    first_idx = section.find(marker)
    if first_idx == -1:
        return section

    prefix = section[:first_idx]
    body = section[first_idx:]
    entry_pattern = re.compile(
        r"(<!-- session-log:[^>]+-->.*?)(?=(?:\n<!-- session-log:)|\Z)", re.DOTALL
    )
    entries = list(entry_pattern.finditer(body))
    if len(entries) <= limit:
        return section

    keep_entries = entries[-limit:]
    keep_start = keep_entries[0].start()
    trimmed_body = body[keep_start:]
    trimmed_body = trimmed_body.lstrip("\n")
    if trimmed_body and not trimmed_body.endswith("\n"):
        trimmed_body += "\n"

    if prefix and not prefix.endswith("\n"):
        prefix += "\n"

    return prefix + trimmed_body


def update_markdown_log(
    path: Path,
    record: SessionRecord,
    header: str,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> bool:
    """Insert ``record`` under ``header`` in ``path`` if not already present."""

    content = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = _render_entry(record)
    updated, changed = _update_markdown_content(
        content, header, entry, record.marker, limit=limit
    )
    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
    return changed


def write_notes(path: Path, record: SessionRecord, *, dry_run: bool = False) -> bool:
    """Append an extended session entry to the notes file located at ``path``."""

    header = "# Session Notes"
    changed = update_markdown_log(path, record, header, dry_run=dry_run)
    return changed


def collect_git_status(repo_root: Path) -> Sequence[str]:
    """Return a summary of ``git status --short`` for logging purposes."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        LOGGER.debug("Unable to collect git status; skipping auto-notes", exc_info=True)
        return ()

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return ()

    if len(lines) == 1:
        return (f"git status: {lines[0]}",)

    return ("git status changes:", *lines)


class RoadmapMaintainer:
    """Synchronize the roadmap document with the Codex status dashboard."""

    def __init__(self, codex_path: Path, roadmap_path: Path) -> None:
        """Initialize maintainer with Codex source and roadmap target paths.

        :param codex_path: Path to the Codex dashboard markdown document.
        :param roadmap_path: Path to the roadmap snapshot markdown output.
        """

        self.codex_path = codex_path
        self.roadmap_path = roadmap_path

    def refresh(self, record: SessionRecord, *, dry_run: bool = False) -> bool:
        """Update the roadmap document using the latest Codex status.

        Returns ``True`` when the rendered content differs from the previous snapshot.
        """

        dashboard = self._extract_status_dashboard()
        content = self._render_content(record, dashboard)
        previous = (
            self.roadmap_path.read_text(encoding="utf-8")
            if self.roadmap_path.exists()
            else None
        )
        changed = previous != content
        if dry_run:
            LOGGER.info("[dry-run] Would update roadmap at %s", self.roadmap_path)
            return changed
        if changed:
            self.roadmap_path.parent.mkdir(parents=True, exist_ok=True)
            self.roadmap_path.write_text(content, encoding="utf-8")
        return changed

    def _extract_status_dashboard(self) -> str:
        if not self.codex_path.exists():
            raise MarkdownUpdateError(f"Codex ledger not found at {self.codex_path}")
        content = self.codex_path.read_text(encoding="utf-8")
        header_pattern = re.compile(r"(?m)^##\s+Status Dashboard\s*$")
        header_match = header_pattern.search(content)
        if not header_match:
            raise MarkdownUpdateError("Status Dashboard section missing from Codex ledger")

        start = header_match.end()
        remainder = content[start:]
        next_header = re.search(r"(?m)^##\s+", remainder)
        section = remainder[: next_header.start()] if next_header else remainder
        return section.strip()

    def _render_content(self, record: SessionRecord, dashboard: str) -> str:
        notes_section = "\n".join(f"- {note}" for note in record.notes) if record.notes else "- (none)"
        lines = [
            "# Delivery Roadmap",
            "",
            f"_Refreshed automatically: {record.timestamp.isoformat()}_",
            "",
            "## Session Pulse",
            "",
            f"- **Session:** {record.session_name}",
            f"- **Summary:** {record.summary}",
            "- **Key Notes:**",
            notes_section,
            "",
            "## Snapshot Details",
            "",
            "This roadmap snapshot is rendered from the `## Status Dashboard` section",
            "in `Codex_Master_Task_Results.md`. Update the ledger first, then rerun",
            "`tools/session_finalize.py` to propagate the refresh across documentation.",
            "",
            "### Status Dashboard Snapshot",
            "",
            dashboard,
            "",
            "## Automation Footnotes",
            "",
            "- Maintained by `tools/session_finalize.py` using `docs/documentation_manifest.json`.",
            "- `docs/history/SESSION_LOG.md` retains the full session narrative; the README",
            "  timeline is intentionally pruned to the latest updates.",
            "",
        ]
        return "\n".join(lines)


class SessionFinalizer:
    """Coordinate end-of-session metadata updates for the repository."""

    def __init__(
        self,
        repo_root: Path,
        record: SessionRecord,
        documents: Iterable[DocumentPlan],
        roadmap: RoadmapMaintainer | None,
        *,
        sync_agents: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.record = record
        self.documents = list(documents)
        self.roadmap = roadmap
        self.sync_agents = sync_agents
        self.dry_run = dry_run

    def finalize(self) -> FinalizationReport:
        """Execute the session finalization workflow."""

        agents_changed: bool | None = None
        if self.sync_agents:
            LOGGER.info("Synchronizing AGENTS.md manifest")
            agents_changed = synchronize_agents(
                repo_root=self.repo_root, dry_run=self.dry_run
            )

        document_results = [self._update_document(plan) for plan in self.documents]

        roadmap_changed: bool | None = None
        if self.roadmap is not None:
            LOGGER.info("Refreshing roadmap snapshot")
            roadmap_changed = self.roadmap.refresh(self.record, dry_run=self.dry_run)

        return FinalizationReport(
            agents_changed=agents_changed,
            documents=tuple(document_results),
            roadmap_changed=roadmap_changed,
        )

    def _update_document(self, plan: DocumentPlan) -> DocumentUpdateResult:
        LOGGER.info("Updating %s", plan.path)

        created = False
        if plan.ensure_exists and not plan.path.exists():
            if self.dry_run:
                LOGGER.info("[dry-run] Would create %s", plan.path)
            else:
                LOGGER.info("Creating %s", plan.path)
                plan.path.parent.mkdir(parents=True, exist_ok=True)
                initial_content = plan.template or f"{plan.header}\n\n"
                plan.path.write_text(initial_content, encoding="utf-8")
            created = True

        changed = update_markdown_log(
            plan.path,
            self.record,
            plan.header,
            dry_run=self.dry_run,
            limit=plan.limit,
        )
        if changed:
            LOGGER.info(
                "%s %s",
                "Would update" if self.dry_run else "Updated",
                plan.path,
            )
        else:
            LOGGER.info("No changes required for %s", plan.path)
        return DocumentUpdateResult(path=plan.path, changed=changed or created)


def load_documentation_manifest(
    repo_root: Path, manifest_path: Path
) -> DocumentationManifest | None:
    """Load documentation manifest configuration if present."""

    resolved = manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    if not resolved.exists():
        return None

    data = json.loads(resolved.read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    documents_cfg = data.get("documents", [])
    documents: list[DocumentPlan] = []
    for entry in documents_cfg:
        doc_path = Path(entry["path"])
        if not doc_path.is_absolute():
            doc_path = repo_root / doc_path
        header = entry["header"]
        limit = entry.get("limit")
        ensure_exists = bool(entry.get("ensure_exists", False))
        template_path_value = entry.get("template_path")
        template: str | None = None
        if template_path_value:
            template_path = Path(template_path_value)
            if not template_path.is_absolute():
                template_path = repo_root / template_path
            template = template_path.read_text(encoding="utf-8")
        tags = frozenset(entry.get("tags", []))
        documents.append(
            DocumentPlan(
                path=doc_path,
                header=header,
                limit=limit,
                ensure_exists=ensure_exists,
                template=template,
                tags=tags,
            )
        )

    codex_path_cfg = data.get("codex_path", "Codex_Master_Task_Results.md")
    roadmap_path_cfg = data.get("roadmap_path", "docs/ROADMAP.md")
    codex_path = Path(codex_path_cfg)
    roadmap_path = Path(roadmap_path_cfg)
    if not codex_path.is_absolute():
        codex_path = repo_root / codex_path
    if not roadmap_path.is_absolute():
        roadmap_path = repo_root / roadmap_path

    return DocumentationManifest(
        version=version,
        documents=tuple(documents),
        codex_path=codex_path,
        roadmap_path=roadmap_path,
        manifest_path=resolved,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments for the session finalizer utility."""

    parser = argparse.ArgumentParser(description="Finalize repository session metadata")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--timestamp", help="ISO timestamp; defaults to current UTC time")
    parser.add_argument("--readme", action="append", dest="readmes")
    parser.add_argument("--codex-path")
    parser.add_argument("--notes-path")
    parser.add_argument("--roadmap-path")
    parser.add_argument("--manifest-path", default="docs/documentation_manifest.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--skip-agent-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-git-status",
        action="store_true",
        help="Automatically append git status summary to session notes.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _build_record(args: argparse.Namespace, repo_root: Path) -> SessionRecord:
    timestamp = (
        datetime.fromisoformat(args.timestamp)
        if args.timestamp
        else datetime.now(timezone.utc).replace(microsecond=0)
    )
    notes: List[str] = list(args.note)
    if args.include_git_status:
        notes.extend(collect_git_status(repo_root))
    return SessionRecord(
        session_name=args.session_name,
        summary=args.summary,
        notes=tuple(notes),
        timestamp=timestamp,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the session finalizer CLI."""

    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    repo_root = Path(args.repo_root).resolve()
    record = _build_record(args, repo_root)

    manifest = load_documentation_manifest(repo_root, Path(args.manifest_path))

    documents: list[DocumentPlan]
    if manifest:
        documents = list(manifest.documents)
        codex_path = repo_root / Path(args.codex_path) if args.codex_path else manifest.codex_path
        roadmap_path = (
            repo_root / Path(args.roadmap_path)
            if args.roadmap_path
            else manifest.roadmap_path
        )

        if args.readmes:
            requested = {(repo_root / Path(path)).resolve() for path in args.readmes}
            readme_docs = {
                plan.path.resolve(): plan for plan in documents if plan.is_readme
            }
            missing = sorted(str(path) for path in requested - set(readme_docs))
            if missing:
                raise SystemExit(
                    "Manifest does not define the requested README path(s): "
                    + ", ".join(missing)
                )
            documents = [
                plan
                for plan in documents
                if not plan.is_readme or plan.path.resolve() in requested
            ]
    else:
        readmes = (
            [repo_root / path for path in args.readmes]
            if args.readmes
            else discover_readmes(repo_root)
        )
        documents = [
            DocumentPlan(path=readme, header="## Session Updates") for readme in readmes
        ]

        codex_path = repo_root / Path(
            args.codex_path or "Codex_Master_Task_Results.md"
        )
        roadmap_path = repo_root / Path(args.roadmap_path or "docs/ROADMAP.md")
        documents.append(DocumentPlan(path=codex_path, header="## Session Log"))
        notes_path = repo_root / Path(args.notes_path or "tasks/SESSION_NOTES.md")
        documents.append(DocumentPlan(path=notes_path, header="# Session Notes"))

    roadmap = RoadmapMaintainer(codex_path=codex_path, roadmap_path=roadmap_path)

    finalizer = SessionFinalizer(
        repo_root=repo_root,
        record=record,
        documents=documents,
        roadmap=roadmap,
        sync_agents=not args.skip_agent_sync,
        dry_run=args.dry_run,
    )
    report = finalizer.finalize()

    changed_docs = [str(path) for path in report.documents_changed]
    if changed_docs:
        LOGGER.info("Updated %d document(s): %s", len(changed_docs), ", ".join(changed_docs))
    else:
        LOGGER.info("No document updates required")

    if report.roadmap_changed is True:
        LOGGER.info("Roadmap snapshot refreshed")
    elif report.roadmap_changed is False:
        LOGGER.info("Roadmap already up to date")

    if report.agents_changed is True:
        LOGGER.info("AGENTS.md manifest synchronization applied changes")
    elif report.agents_changed is False:
        LOGGER.info("AGENTS.md manifest already in sync")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
