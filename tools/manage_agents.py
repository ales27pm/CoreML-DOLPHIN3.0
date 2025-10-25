"""Synchronize scoped AGENTS.md files based on the root manifest.

This utility reads the manifest embedded inside the repository's root `AGENTS.md` file and
synchronizes directory-specific `AGENTS.md` files. It supports two workflows:

- `sync`: create/update/remove agent files to match the manifest (with optional dry-run output).
- `check`: verify the working tree already matches the manifest, exiting with a non-zero status if
  differences are detected.

The script enforces the repository requirement to keep all instructions authoritative, fresh,
and free of stale placeholders.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

MARKER_BEGIN = "<!--AGENTS_MANIFEST_BEGIN-->"
MARKER_END = "<!--AGENTS_MANIFEST_END-->"


@dataclass(frozen=True)
class AgentScope:
    """Represents a scoped AGENTS.md file defined in the manifest."""

    path: Path
    content: str

    @property
    def target_file(self) -> Path:
        return self.path / "AGENTS.md"


class ManifestError(RuntimeError):
    """Raised when the manifest cannot be parsed or is invalid."""


def find_repo_root(script_path: Path) -> Path:
    """Return the repository root directory from the script's location."""

    return script_path.resolve().parents[1]


def load_manifest(root_file: Path) -> Sequence[AgentScope]:
    """Parse the automation manifest from the root AGENTS.md file."""

    text = root_file.read_text(encoding="utf-8")
    start = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    if start == -1 or end == -1 or start >= end:
        raise ManifestError(
            "Root AGENTS.md must contain manifest markers "
            f"'{MARKER_BEGIN}' and '{MARKER_END}'."
        )
    start += len(MARKER_BEGIN)
    manifest_text = text[start:end].strip()
    if not manifest_text:
        raise ManifestError("Manifest block is empty; provide at least an empty JSON object.")

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Failed to parse manifest JSON: {exc}") from exc

    directories = manifest.get("directories")
    if directories is None:
        raise ManifestError("Manifest JSON must define a 'directories' array.")
    if not isinstance(directories, list):
        raise ManifestError("'directories' must be a list of directory descriptors.")

    scopes: List[AgentScope] = []
    for entry in directories:
        if not isinstance(entry, dict):
            raise ManifestError("Each directory entry must be an object.")
        raw_path = entry.get("path")
        if not raw_path or not isinstance(raw_path, str):
            raise ManifestError("Each entry requires a non-empty string 'path'.")
        raw_content = entry.get("content")
        if raw_content is None or not isinstance(raw_content, str):
            raise ManifestError("Each entry requires a string 'content'.")

        path = (root_file.parent / raw_path).resolve()
        if root_file.parent not in path.parents and path != root_file.parent:
            raise ManifestError(f"Path '{raw_path}' escapes the repository root.")

        scopes.append(AgentScope(path=path, content=ensure_trailing_newline(raw_content)))

    return scopes


def ensure_trailing_newline(content: str) -> str:
    """Guarantee that the generated file ends with a newline."""

    return content if content.endswith("\n") else f"{content}\n"


def gather_existing_agents(repo_root: Path) -> Dict[Path, Path]:
    """Return a mapping of directories to their AGENTS.md files (excluding the root)."""

    agents: Dict[Path, Path] = {}
    for file_path in repo_root.rglob("AGENTS.md"):
        if file_path == repo_root / "AGENTS.md":
            continue
        agents[file_path.parent.resolve()] = file_path
    return agents


def sync_scopes(scopes: Sequence[AgentScope], *, repo_root: Path, dry_run: bool) -> bool:
    """Apply the manifest to the filesystem. Returns True if changes were made."""

    existing_agents = gather_existing_agents(repo_root)
    changed = False

    for scope in scopes:
        directory = scope.path
        target = scope.target_file
        rel_dir = directory.relative_to(repo_root)
        directory.mkdir(parents=True, exist_ok=True)
        previous_content = target.read_text(encoding="utf-8") if target.exists() else None
        if previous_content != scope.content:
            changed = True
            action = "Would update" if dry_run else ("Updating" if previous_content else "Creating")
            logging.info("%s %s", action, rel_dir / target.name)
            if not dry_run:
                target.write_text(scope.content, encoding="utf-8")
        existing_agents.pop(directory, None)

    # Remove unmanaged agent files.
    for directory, file_path in existing_agents.items():
        rel_dir = directory.relative_to(repo_root)
        changed = True
        if dry_run:
            logging.info("Would remove %s", rel_dir / file_path.name)
        else:
            logging.info("Removing %s", rel_dir / file_path.name)
            file_path.unlink()

    return changed


def check_scopes(scopes: Sequence[AgentScope], *, repo_root: Path) -> bool:
    """Verify that the manifest and filesystem are in sync."""

    mismatch = False
    existing_agents = gather_existing_agents(repo_root)

    for scope in scopes:
        directory = scope.path
        target = scope.target_file
        rel_dir = directory.relative_to(repo_root)
        if not target.exists():
            logging.error("Missing AGENTS.md at %s", rel_dir / target.name)
            mismatch = True
            continue
        content = target.read_text(encoding="utf-8")
        if content != scope.content:
            logging.error("Outdated AGENTS.md at %s", rel_dir / target.name)
            mismatch = True
        existing_agents.pop(directory, None)

    for directory, file_path in existing_agents.items():
        rel_dir = directory.relative_to(repo_root)
        logging.error("Unexpected AGENTS.md at %s", rel_dir / file_path.name)
        mismatch = True

    return not mismatch


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Create/update agent files to match the manifest")
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying the working tree.",
    )

    subparsers.add_parser("check", help="Verify that agent files are already synchronized")

    return parser.parse_args(argv)


def configure_logging() -> None:
    """Configure logging output for consistent CLI feedback."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the CLI."""

    args = parse_args(argv or sys.argv[1:])
    configure_logging()

    repo_root = find_repo_root(Path(__file__))
    root_agents = repo_root / "AGENTS.md"
    if not root_agents.exists():
        raise ManifestError("Root AGENTS.md not found. Ensure you are in the repository root.")

    scopes = load_manifest(root_agents)

    if args.command == "sync":
        changed = sync_scopes(scopes, repo_root=repo_root, dry_run=args.dry_run)
        if changed and args.dry_run:
            logging.info("Dry run detected differences. Re-run without --dry-run to apply them.")
        return 0

    if args.command == "check":
        in_sync = check_scopes(scopes, repo_root=repo_root)
        return 0 if in_sync else 1

    raise ManifestError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
