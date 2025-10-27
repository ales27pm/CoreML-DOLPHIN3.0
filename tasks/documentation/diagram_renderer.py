"""Architecture diagram rendering utilities for Codex Task 35.

This module converts DOT graphs into deterministic ASCII tables so the
repository's documentation can embed human-readable topology summaries without
shipping binary diagrams.  The implementation intentionally leans on Graphviz's
``plain`` output format which describes each node's layout coordinates in a
stable textual representation.  We provide a wrapper that extracts the node
metadata, normalises ordering, enforces width constraints, and exposes a small
CLI for offline rendering.

The module is production-ready with robust error handling and logging so the
same logic can be embedded in automation pipelines that regenerate
architecture documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import graphviz

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 80
_NAME_WIDTH = 18
_LABEL_WIDTH = 32
_POSITION_WIDTH = 18


class DiagramRenderingError(RuntimeError):
    """Raised when a DOT source cannot be rendered into ASCII."""


@dataclass(frozen=True)
class DiagramNode:
    """Metadata extracted from Graphviz plain output."""

    name: str
    x: float
    y: float
    width: float
    height: float
    label: str

    def format_row(self) -> str:
        """Render the node into an ASCII table row."""

        position = f"({self.x:.2f}, {self.y:.2f})"
        label = _truncate(self.label, _LABEL_WIDTH)
        return (
            f"{self.name:<{_NAME_WIDTH}} | "
            f"{label:<{_LABEL_WIDTH}} | "
            f"{position:>{_POSITION_WIDTH}}"
        )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


def _parse_plain_output(plain_output: str) -> List[DiagramNode]:
    nodes: list[DiagramNode] = []
    for raw_line in plain_output.splitlines():
        if not raw_line.startswith("node "):
            continue
        parts = shlex.split(raw_line)
        if len(parts) < 10:
            raise DiagramRenderingError(
                "Unexpected Graphviz plain output: missing node attributes"
            )
        try:
            _, name, x_str, y_str, width_str, height_str, label, *_ = parts
            node = DiagramNode(
                name=name,
                x=float(x_str),
                y=float(y_str),
                width=float(width_str),
                height=float(height_str),
                label=label,
            )
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise DiagramRenderingError(
                f"Failed to parse node metrics from line: {raw_line}"
            ) from exc
        nodes.append(node)
    if not nodes:
        raise DiagramRenderingError("DOT source did not contain any nodes")
    # Stable ordering: sort by Y descending (top to bottom) then X ascending.
    nodes.sort(key=lambda item: (-item.y, item.x, item.name))
    return nodes


def _render(nodes: Sequence[DiagramNode], max_width: int) -> str:
    header = (
        f"{'Node':<{_NAME_WIDTH}} | "
        f"{'Label':<{_LABEL_WIDTH}} | "
        f"{'Position':>{_POSITION_WIDTH}}"
    )
    separator = "-" * min(max_width, max(len(header), _NAME_WIDTH + _LABEL_WIDTH + _POSITION_WIDTH + 6))
    lines = [header, separator]
    for node in nodes:
        row = node.format_row()
        if len(row) > max_width:
            raise DiagramRenderingError(
                f"Rendered row exceeds width limit ({max_width}): {row}"
            )
        lines.append(row)
    ascii_art = "\n".join(lines)
    if any(len(line) > max_width for line in lines):  # pragma: no cover - defensive
        raise DiagramRenderingError("ASCII output exceeded width constraint")
    return ascii_art


def generate_ascii_diagram(
    dot_source: str,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    source_factory: Callable[[str], graphviz.Source] | None = None,
) -> str:
    """Render ``dot_source`` into an ASCII table constrained by ``max_width``."""

    if max_width < 20:
        raise ValueError("max_width must be at least 20 characters")
    factory = source_factory or graphviz.Source
    try:
        plain_output = factory(dot_source).pipe(format="plain").decode("utf-8")
    except graphviz.backend.ExecutableNotFound as exc:
        raise DiagramRenderingError(
            "Graphviz executable not found. Install Graphviz to enable rendering."
        ) from exc
    except graphviz.backend.CalledProcessError as exc:
        raise DiagramRenderingError(f"Graphviz failed to render DOT source: {exc}") from exc
    logger.debug("Graphviz plain output:\n%s", plain_output)
    nodes = _parse_plain_output(plain_output)
    ascii_art = _render(nodes, max_width)
    logger.info("Rendered %s nodes into ASCII diagram", len(nodes))
    return ascii_art


def render_to_path(
    dot_source: str,
    output_path: Path,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    source_factory: Callable[[str], graphviz.Source] | None = None,
) -> Path:
    """Render ``dot_source`` and persist the ASCII output to ``output_path``."""

    ascii_art = generate_ascii_diagram(
        dot_source, max_width=max_width, source_factory=source_factory
    )
    output_path.write_text(ascii_art + "\n", encoding="utf-8")
    logger.debug("Wrote ASCII diagram to %s", output_path)
    return output_path


def render_from_file(
    dot_path: Path,
    *,
    output_path: Path | None = None,
    max_width: int = DEFAULT_MAX_WIDTH,
    source_factory: Callable[[str], graphviz.Source] | None = None,
) -> Path:
    """Load DOT contents from ``dot_path`` and write the ASCII rendering."""

    dot_source = dot_path.read_text(encoding="utf-8")
    destination = output_path or dot_path.with_suffix(".txt")
    return render_to_path(
        dot_source,
        destination,
        max_width=max_width,
        source_factory=source_factory,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render DOT diagrams into deterministic ASCII tables",
    )
    parser.add_argument("dot_file", type=Path, help="Path to the DOT file to render")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the ASCII output (defaults to <dot>.txt)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help="Maximum number of characters per line",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the ASCII rendering as JSON payload to stdout",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        output_path = render_from_file(
            args.dot_file,
            output_path=args.output,
            max_width=args.max_width,
        )
    except (DiagramRenderingError, OSError) as exc:
        logger.error("Failed to render diagram: %s", exc)
        return 1
    ascii_art = output_path.read_text(encoding="utf-8")
    if args.json:
        print(json.dumps({"path": str(output_path), "diagram": ascii_art}))
    else:
        print(ascii_art, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
