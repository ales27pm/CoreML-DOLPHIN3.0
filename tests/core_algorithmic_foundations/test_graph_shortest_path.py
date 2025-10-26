"""Regression tests for the graph shortest path toolkit (Task 5)."""

from __future__ import annotations

import math
import pytest

from tasks.core_algorithmic_foundations.graph_shortest_path import (
    GraphValidationError,
    ShortestPathResult,
    WeightedGraph,
    build_networkx_graph,
    dijkstra,
    format_distances,
    visualize_shortest_paths,
)


def sample_graph() -> dict[str, tuple[tuple[str, float], ...]]:
    return {
        "A": (("B", 7), ("C", 9), ("F", 14)),
        "B": (("C", 10), ("D", 15)),
        "C": (("D", 11), ("F", 2)),
        "D": (("E", 6),),
        "E": tuple(),
        "F": (("E", 9),),
    }


def test_dijkstra_computes_expected_costs() -> None:
    graph = sample_graph()
    result = dijkstra(graph, "A")

    assert isinstance(result, ShortestPathResult)
    assert math.isclose(result.distance_to("E"), 20.0)
    assert result.path_to("E") == ["A", "C", "F", "E"]
    assert result.path_to("D") == ["A", "C", "D"]


def test_unreachable_node_reports_infinite_distance() -> None:
    graph = {
        "A": (("B", 1),),
        "B": tuple(),
        "C": tuple(),
    }
    result = dijkstra(graph, "A")
    assert math.isinf(result.distance_to("C"))
    assert result.path_to("C") is None


@pytest.mark.parametrize(
    "graph",
    [
        {"A": (("B", -1),)},
        {"A": (("B", 1),), "B": (("A", "invalid"),)},
        {"A": (("B", 1, 2),)},
        {1: (("B", 1),)},
    ],
)
def test_invalid_graph_inputs_raise(graph: dict) -> None:
    with pytest.raises(GraphValidationError):
        WeightedGraph.from_mapping(graph)  # type: ignore[arg-type]


def test_unknown_source_raises_value_error() -> None:
    graph = sample_graph()
    with pytest.raises(ValueError):
        dijkstra(graph, "Z")


def test_format_distances_generates_human_readable_lines() -> None:
    result = dijkstra(sample_graph(), "A")
    formatted = format_distances(result)
    assert any("A -> E" in line and "cost=20.000" in line for line in formatted)
    unreachable = format_distances(
        dijkstra({"A": (("B", 1),), "B": tuple(), "C": tuple()}, "A")
    )
    assert any(line.endswith("unreachable") for line in unreachable)


def test_networkx_visualization_payload_round_trip() -> None:
    nx = pytest.importorskip("networkx")
    payload = visualize_shortest_paths(sample_graph(), "A")
    assert set(payload.graph.nodes) == set(sample_graph())
    assert payload.distances["E"] == pytest.approx(20.0)

    diag = build_networkx_graph(sample_graph())
    assert isinstance(diag, nx.DiGraph)
    assert diag.has_edge("A", "B")
    assert payload.edge_labels[("A", "B")] == 7.0
