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
from typing import Callable, Protocol, Sequence, runtime_checkable, cast

try:  # pragma: no cover - import guard for optional dependency
    import graphviz  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - dependency may be absent in some envs
    graphviz = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MAX_WIDTH = 80
_NAME_WIDTH = 18
_LABEL_WIDTH = 32
_POSITION_WIDTH = 18
_TABLE_SEPARATOR_PADDING = 6
_MIN_TABLE_WIDTH = (
    _NAME_WIDTH + _LABEL_WIDTH + _POSITION_WIDTH + _TABLE_SEPARATOR_PADDING
)


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

        decimals = 2
        while decimals > 0:
            position = f"({self.x:.{decimals}f}, {self.y:.{decimals}f})"
            if len(position) <= _POSITION_WIDTH:
                break
            decimals -= 1
        else:
            position = f"({self.x:.0f}, {self.y:.0f})"

        if len(position) > _POSITION_WIDTH:
            position = _truncate(position, _POSITION_WIDTH)

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


def _parse_plain_output(plain_output: str) -> list[DiagramNode]:
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
    content_width = max(len(header), _MIN_TABLE_WIDTH)
    if content_width > max_width:
        raise DiagramRenderingError(
            f"max_width {max_width} is insufficient for table layout (requires {content_width})"
        )
    separator = "-" * content_width
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


@runtime_checkable
class _SupportsPipe(Protocol):
    def pipe(self, format: str) -> bytes:
        """Render the DOT payload using the requested Graphviz format."""


def generate_ascii_diagram(
    dot_source: str,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    source_factory: Callable[[str], _SupportsPipe] | None = None,
) -> str:
    """Render ``dot_source`` into an ASCII table constrained by ``max_width``."""

    if max_width < _MIN_TABLE_WIDTH:
        raise ValueError(f"max_width must be at least {_MIN_TABLE_WIDTH} characters")
    if source_factory is not None:
        factory = source_factory
    elif graphviz is not None:
        factory = graphviz.Source  # type: ignore[assignment]
    else:
        raise DiagramRenderingError(
            "Graphviz library is not installed. Provide a source_factory or install Graphviz."
        )

    try:
        pipeable = factory(dot_source)
    except Exception as exc:  # pragma: no cover - defensive
        raise DiagramRenderingError(f"Failed to build Graphviz source: {exc}") from exc

    if not isinstance(pipeable, _SupportsPipe):  # pragma: no cover - defensive
        raise DiagramRenderingError(
            "Provided source_factory returned unsupported object"
        )

    try:
        plain_output_bytes = pipeable.pipe(format="plain")
    except Exception as exc:
        if graphviz is not None:
            backend = getattr(graphviz, "backend", None)
            executable_not_found: tuple[type[BaseException], ...] = tuple()
            called_process_error: tuple[type[BaseException], ...] = tuple()
            if backend is not None:
                raw_executable_not_found: object = getattr(
                    backend, "ExecutableNotFound", tuple()
                )
                raw_called_process_error: object = getattr(
                    backend, "CalledProcessError", tuple()
                )
                if isinstance(raw_executable_not_found, type) and issubclass(
                    raw_executable_not_found, BaseException
                ):
                    executable_not_found = (raw_executable_not_found,)
                elif isinstance(raw_executable_not_found, tuple):
                    executable_not_found = cast(
                        tuple[type[BaseException], ...], raw_executable_not_found
                    )
                if isinstance(raw_called_process_error, type) and issubclass(
                    raw_called_process_error, BaseException
                ):
                    called_process_error = (raw_called_process_error,)
                elif isinstance(raw_called_process_error, tuple):
                    called_process_error = cast(
                        tuple[type[BaseException], ...], raw_called_process_error
                    )
            if executable_not_found and isinstance(exc, executable_not_found):
                raise DiagramRenderingError(
                    "Graphviz executable not found. Install Graphviz to enable rendering."
                ) from exc
            if called_process_error and isinstance(exc, called_process_error):
                raise DiagramRenderingError(
                    f"Graphviz failed to render DOT source: {exc}"
                ) from exc
        raise DiagramRenderingError(
            f"Graphviz failed to render DOT source: {exc}"
        ) from exc

    if not isinstance(plain_output_bytes, (bytes, bytearray)):
        raise DiagramRenderingError("Graphviz plain output must be bytes")
    plain_output = bytes(plain_output_bytes).decode("utf-8")
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
    source_factory: Callable[[str], _SupportsPipe] | None = None,
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
    source_factory: Callable[[str], _SupportsPipe] | None = None,
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
        logger.exception("Failed to render diagram: %s", exc)
        return 1
    ascii_art = output_path.read_text(encoding="utf-8")
    if args.json:
        print(json.dumps({"path": str(output_path), "diagram": ascii_art}))
    else:
        print(ascii_art, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
