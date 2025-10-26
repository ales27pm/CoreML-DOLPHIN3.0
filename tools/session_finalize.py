"""Utilities for synchronizing repository metadata at the end of a session."""

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


def slugify(text: str) -> str:
    """Generate a filesystem-safe slug from ``text`` for use in markers."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return cleaned.strip("-") or "session"


def discover_readmes(root: Path) -> List[Path]:
    """Return README files beneath ``root`` while skipping vendor directories."""

    excluded = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
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


class SessionFinalizer:
    """Coordinate end-of-session metadata updates for the repository."""

    def __init__(
        self,
        repo_root: Path,
        record: SessionRecord,
        readmes: Iterable[Path],
        codex_path: Path,
        notes_path: Path,
        *,
        sync_agents: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.record = record
        self.readmes = list(readmes)
        self.codex_path = codex_path
        self.notes_path = notes_path
        self.sync_agents = sync_agents
        self.dry_run = dry_run

    def finalize(self) -> None:
        """Execute the session finalization workflow."""

        if self.sync_agents and not self.dry_run:
            self._sync_agents()

        for readme in self.readmes:
            self._update_path(readme, "## Session Updates")

        self._update_path(self.codex_path, "## Session Log")
        self._update_path(self.notes_path, "# Session Notes")

    def _sync_agents(self) -> None:
        manage_script = self.repo_root / "tools" / "manage_agents.py"
        if not manage_script.exists():
            raise FileNotFoundError("tools/manage_agents.py not found")
        LOGGER.info("Running manage_agents sync")
        subprocess.run([sys.executable, str(manage_script), "sync"], check=True)

    def _update_path(self, path: Path, header: str) -> None:
        LOGGER.info("Updating %s", path)
        if self.dry_run:
            return
        update_markdown_log(path, self.record, header)


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
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--skip-agent-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _build_record(args: argparse.Namespace) -> SessionRecord:
    timestamp = (
        datetime.fromisoformat(args.timestamp)
        if args.timestamp
        else datetime.now(timezone.utc).replace(microsecond=0)
    )
    return SessionRecord(
        session_name=args.session_name,
        summary=args.summary,
        notes=tuple(args.note),
        timestamp=timestamp,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the session finalizer CLI."""

    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    repo_root = Path(args.repo_root).resolve()
    record = _build_record(args)
    readmes = (
        [repo_root / path for path in args.readmes]
        if args.readmes
        else discover_readmes(repo_root)
    )
    finalizer = SessionFinalizer(
        repo_root=repo_root,
        record=record,
        readmes=readmes,
        codex_path=repo_root / args.codex_path,
        notes_path=repo_root / args.notes_path,
        sync_agents=not args.skip_agent_sync,
        dry_run=args.dry_run,
    )
    finalizer.finalize()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
