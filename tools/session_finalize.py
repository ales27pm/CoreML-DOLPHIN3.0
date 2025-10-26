"""Automate end-of-session synchronization for documentation and agent files."""

from __future__ import annotations

import argparse
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


class MarkdownUpdateError(RuntimeError):
    """Raised when markdown content cannot be updated as requested."""


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


def _update_markdown_content(content: str, header: str, entry: str, marker: str) -> tuple[str, bool]:
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
        return combined, True

    insert_pos = match.end()
    subsequent = content[insert_pos:]
    next_header_match = re.search(r"(?m)^##\s+", subsequent)
    if next_header_match:
        absolute_pos = insert_pos + next_header_match.start()
        updated = content[:absolute_pos] + entry_block + content[absolute_pos:]
    else:
        updated = content.rstrip() + entry_block + "\n"
    return updated, True


def update_markdown_log(path: Path, record: SessionRecord, header: str) -> bool:
    """Insert ``record`` under ``header`` in ``path`` if not already present."""

    content = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = _render_entry(record)
    updated, changed = _update_markdown_content(content, header, entry, record.marker)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
    return changed


def write_notes(path: Path, record: SessionRecord) -> bool:
    """Append an extended session entry to the notes file located at ``path``."""

    header = "# Session Notes"
    changed = update_markdown_log(path, record, header)
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
        self.codex_path = codex_path
        self.roadmap_path = roadmap_path

    def refresh(self, record: SessionRecord, *, dry_run: bool = False) -> None:
        """Update the roadmap document using the latest Codex status."""

        dashboard = self._extract_status_dashboard()
        content = self._render_content(record, dashboard)
        if dry_run:
            LOGGER.info("[dry-run] Would update roadmap at %s", self.roadmap_path)
            return
        self.roadmap_path.parent.mkdir(parents=True, exist_ok=True)
        self.roadmap_path.write_text(content, encoding="utf-8")

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
            "# Session Roadmap",
            "",
            f"Last updated: {record.timestamp.isoformat()}",
            "",
            "## Latest Session",
            "",
            f"- **Session:** {record.session_name}",
            f"- **Summary:** {record.summary}",
            "- **Notes:**",
            notes_section,
            "",
            "## Status Dashboard Snapshot",
            "",
            dashboard,
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

    def finalize(self) -> None:
        """Execute the session finalization workflow."""

        if self.sync_agents:
            LOGGER.info("Synchronizing AGENTS.md manifest")
            synchronize_agents(repo_root=self.repo_root, dry_run=self.dry_run)

        for plan in self.documents:
            self._update_document(plan)

        if self.roadmap is not None:
            LOGGER.info("Refreshing roadmap snapshot")
            self.roadmap.refresh(self.record, dry_run=self.dry_run)

    def _update_document(self, plan: DocumentPlan) -> None:
        LOGGER.info("Updating %s", plan.path)
        if self.dry_run:
            return
        update_markdown_log(plan.path, self.record, plan.header)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments for the session finalizer utility."""

    parser = argparse.ArgumentParser(description="Finalize repository session metadata")
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--timestamp", help="ISO timestamp; defaults to current UTC time")
    parser.add_argument("--readme", action="append", dest="readmes")
    parser.add_argument("--codex-path", default="Codex_Master_Task_Results.md")
    parser.add_argument("--notes-path", default="tasks/SESSION_NOTES.md")
    parser.add_argument("--roadmap-path", default="docs/ROADMAP.md")
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

    readmes = (
        [repo_root / path for path in args.readmes]
        if args.readmes
        else discover_readmes(repo_root)
    )

    documents = [
        DocumentPlan(path=readme, header="## Session Updates") for readme in readmes
    ]
    documents.append(DocumentPlan(path=repo_root / args.codex_path, header="## Session Log"))
    documents.append(DocumentPlan(path=repo_root / args.notes_path, header="# Session Notes"))

    roadmap = RoadmapMaintainer(
        codex_path=repo_root / args.codex_path,
        roadmap_path=repo_root / args.roadmap_path,
    )

    finalizer = SessionFinalizer(
        repo_root=repo_root,
        record=record,
        documents=documents,
        roadmap=roadmap,
        sync_agents=not args.skip_agent_sync,
        dry_run=args.dry_run,
    )
    finalizer.finalize()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
