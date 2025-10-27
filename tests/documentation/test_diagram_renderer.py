from __future__ import annotations

from pathlib import Path

import pytest

from tasks.documentation.diagram_renderer import (
    DEFAULT_MAX_WIDTH,
    DiagramRenderingError,
    generate_ascii_diagram,
    render_from_file,
    render_to_path,
)


class _FakeSource:
    def __init__(self, dot_source: str, *, plain_output: str) -> None:
        self.dot_source = dot_source
        self._plain_output = plain_output

    def pipe(self, format: str) -> bytes:
        assert format == "plain"
        return self._plain_output.encode("utf-8")


@pytest.fixture()
def simple_plain_output() -> str:
    return "\n".join(
        [
            "graph 1 1 1",
            "node api 1 3 0.5 0.5 'API Gateway' solid ellipse black none",
            "node db 2 1 0.5 0.5 'Primary Database' solid ellipse black none",
            "node cache 3 2 0.5 0.5 'Redis Cache' solid ellipse black none",
        ]
    )


def test_generate_ascii_diagram_orders_rows_by_coordinates(
    simple_plain_output: str,
) -> None:
    def factory(_: str) -> _FakeSource:
        return _FakeSource(_, plain_output=simple_plain_output)

    ascii_art = generate_ascii_diagram("digraph { }", source_factory=factory)
    lines = ascii_art.splitlines()
    assert lines[0].startswith("Node")
    assert "API Gateway" in lines[2]
    assert "Redis Cache" in lines[3]
    assert "Primary Database" in lines[4]
    # Ensure width constraint respected
    assert all(len(line) <= DEFAULT_MAX_WIDTH for line in lines)


def test_generate_ascii_diagram_errors_on_missing_nodes() -> None:
    def factory(_: str) -> _FakeSource:
        return _FakeSource(_, plain_output="graph 1 1 1")

    with pytest.raises(DiagramRenderingError):
        generate_ascii_diagram("digraph {}", source_factory=factory)


def test_generate_ascii_diagram_enforces_width(simple_plain_output: str) -> None:
    def factory(_: str) -> _FakeSource:
        return _FakeSource(_, plain_output=simple_plain_output)

    with pytest.raises(DiagramRenderingError):
        generate_ascii_diagram("digraph {}", max_width=40, source_factory=factory)


def test_render_to_path_writes_ascii(tmp_path: Path, simple_plain_output: str) -> None:
    output = tmp_path / "diagram.txt"

    def factory(_: str) -> _FakeSource:
        return _FakeSource(_, plain_output=simple_plain_output)

    render_to_path("digraph {}", output, source_factory=factory)
    assert output.exists()
    contents = output.read_text(encoding="utf-8").strip().splitlines()
    assert any("API Gateway" in line for line in contents)


def test_render_from_file_reads_dot(tmp_path: Path, simple_plain_output: str) -> None:
    dot_path = tmp_path / "topology.dot"
    dot_path.write_text("digraph { api -> db }", encoding="utf-8")

    def factory(_: str) -> _FakeSource:
        return _FakeSource(_, plain_output=simple_plain_output)

    output_path = render_from_file(dot_path, source_factory=factory)
    assert output_path.exists()
    assert output_path.suffix == ".txt"
    assert "Redis Cache" in output_path.read_text(encoding="utf-8")
