"""Graph shortest path computation and NetworkX visualization helpers.

This module implements the production-ready solution for
``Codex_Master_Task_Results`` Task 5.  It provides a fully validated Dijkstra
solver, ergonomic accessors for reconstructing shortest paths, and utility
functions that prepare NetworkX visualisation payloads suitable for rendering
in notebooks or automated reports.

Design goals:

* **Deterministic correctness** - adjacency inputs are normalised and validated
  to prevent silent coercion of malformed payloads. Negative weights surface a
  ``GraphValidationError`` immediately because they violate Dijkstra's assumptions.
* **Observability** - ``ShortestPathResult`` exposes helper methods for
  reconstructing the optimal route to any node and for emitting human-readable
  summaries that integrate with logging pipelines.
* **NetworkX integration** - ``visualize_shortest_paths`` returns a
  ``VisualizationPayload`` dataclass containing an ``nx.DiGraph`` instance,
  layout coordinates, and labelled cost metadata so downstream tooling can call
  ``networkx.draw`` or export structured data without recomputing shortest
  paths.

The implementation is dependency-light: NetworkX is imported lazily and a clear
exception instructs the caller to install it when visualisation helpers are
invoked. This keeps the core solver usable in minimal environments while still
meeting the roadmap requirement for NetworkX support.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover - import is for type checking only
    import networkx as nx  # type: ignore[import-not-found,import-untyped]

    NxDiGraph: TypeAlias = nx.DiGraph
else:  # pragma: no cover - alias keeps runtime dependency optional
    NxDiGraph: TypeAlias = Any

GraphEdge = Tuple[str, float]
GraphInput = Mapping[str, Iterable[GraphEdge]]

__all__ = [
    "GraphValidationError",
    "ShortestPathResult",
    "VisualizationPayload",
    "WeightedGraph",
    "build_networkx_graph",
    "dijkstra",
    "format_distances",
    "visualize_shortest_paths",
]


class GraphValidationError(ValueError):
    """Raised when an input graph violates structural constraints."""


@dataclass(frozen=True)
class WeightedGraph:
    """Immutable adjacency representation for weighted directed graphs.

    The ``from_mapping`` constructor validates the input structure and promotes
    all weights to ``float`` for numerical stability.
    """

    adjacency: Mapping[str, Tuple[GraphEdge, ...]]

    @classmethod
    def from_mapping(cls, graph: GraphInput) -> "WeightedGraph":
        adjacency = _normalize_graph(graph)
        return cls(adjacency)

    def neighbors(self, node: str) -> Tuple[GraphEdge, ...]:
        return self.adjacency[node]

    def nodes(self) -> Iterator[str]:
        return iter(self.adjacency)


@dataclass(frozen=True)
class ShortestPathResult:
    """Container for single-source shortest path outcomes."""

    source: str
    distances: Mapping[str, float]
    previous: Mapping[str, Optional[str]]

    def distance_to(self, node: str) -> float:
        """Return the computed distance to *node* or ``math.inf`` when unreachable."""

        if node not in self.distances:
            raise KeyError(f"Unknown node: {node!r}")
        return self.distances[node]

    def path_to(self, node: str) -> Optional[List[str]]:
        """Reconstruct the optimal path from ``source`` to *node*.

        ``None`` is returned when the destination is unreachable. A
        ``KeyError`` is raised for unknown nodes to surface contract violations
        early.
        """

        if node not in self.distances:
            raise KeyError(f"Unknown node: {node!r}")
        if math.isinf(self.distances[node]):
            return None
        path: List[str] = []
        cursor: Optional[str] = node
        while cursor is not None:
            path.append(cursor)
            cursor = self.previous[cursor]
        path.reverse()
        return path

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the shortest paths tree."""

        snapshot: Dict[str, Any] = {}
        for node in self.distances:
            path = self.path_to(node)
            snapshot[node] = {
                "distance": self.distances[node],
                "path": path,
            }
        snapshot["source"] = self.source
        return snapshot


@dataclass(frozen=True)
class VisualizationPayload:
    """Structured artefact for NetworkX visualisation pipelines."""

    graph: NxDiGraph
    positions: Mapping[str, Tuple[float, float]]
    edge_labels: Mapping[Tuple[str, str], float]
    distances: Mapping[str, float]
    source: str
    target: Optional[str] = None
    focus_path: Optional[List[str]] = None


def dijkstra(
    graph: Union[GraphInput, WeightedGraph],
    source: str,
    *,
    target: Optional[str] = None,
) -> ShortestPathResult:
    """Compute single-source shortest paths using Dijkstra's algorithm.

    Parameters
    ----------
    graph:
        Mapping representing a weighted directed graph where keys are node
        identifiers and values are iterables of ``(neighbor, weight)`` tuples.
    source:
        Node to start the traversal from.
    target:
        Optional node at which the algorithm can early terminate.

    Returns
    -------
    ShortestPathResult
        Structure containing distances and parent pointers for path recovery.

    Raises
    ------
    GraphValidationError
        If the graph contains invalid keys or negative edge weights.
    ValueError
        If ``source`` is not present in the graph.
    """

    weighted = (
        graph if isinstance(graph, WeightedGraph) else WeightedGraph.from_mapping(graph)
    )
    if source not in weighted.adjacency:
        raise ValueError(f"Source node {source!r} is not present in the graph")

    distances: Dict[str, float] = {node: math.inf for node in weighted.adjacency}
    previous: Dict[str, Optional[str]] = {node: None for node in weighted.adjacency}

    queue: List[Tuple[float, str]] = [(0.0, source)]
    distances[source] = 0.0

    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distances[node]:
            continue
        if target is not None and node == target:
            break
        for neighbor, weight in weighted.neighbors(node):
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))

    return ShortestPathResult(source=source, distances=distances, previous=previous)


def build_networkx_graph(graph: Union[GraphInput, WeightedGraph]) -> NxDiGraph:
    """Convert *graph* to a NetworkX ``DiGraph``.

    The function imports NetworkX lazily to avoid forcing the dependency for
    callers that only require the pure-Python solver.
    """

    try:
        import networkx as nx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via tests when missing
        raise ModuleNotFoundError(
            "NetworkX is required for visualization support. Install it via 'pip install networkx'."
        ) from exc

    weighted = (
        graph if isinstance(graph, WeightedGraph) else WeightedGraph.from_mapping(graph)
    )
    nx_graph = nx.DiGraph()
    for node in weighted.adjacency:
        nx_graph.add_node(node)
    for node, edges in weighted.adjacency.items():
        for neighbor, weight in edges:
            nx_graph.add_edge(node, neighbor, weight=weight)
    return nx_graph


def visualize_shortest_paths(
    graph: Union[GraphInput, WeightedGraph],
    source: str,
    *,
    layout: str = "spring",
    target: Optional[str] = None,
    layout_seed: Optional[int] = 13,
) -> VisualizationPayload:
    """Prepare NetworkX artefacts for rendering the shortest path tree.

    Parameters
    ----------
    graph:
        Input weighted graph mapping.
    source:
        Source node for the Dijkstra traversal.
    layout:
        Name of the NetworkX layout algorithm to use (``spring``, ``kamada_kawai``,
        or ``circular``). The layout is resolved lazily to keep integration tests
        deterministic and side-effect free.
    target:
        Optional node that downstream consumers may highlight in the resulting
        visualisation. The solver still computes the full shortest paths tree so
        aggregate metrics remain available.
    layout_seed:
        Seed forwarded to layout functions that support deterministic placement.
    """

    try:
        import networkx as nx  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via tests when missing
        raise ModuleNotFoundError(
            "NetworkX is required for visualization support. Install it via 'pip install networkx'."
        ) from exc

    weighted = (
        graph if isinstance(graph, WeightedGraph) else WeightedGraph.from_mapping(graph)
    )
    result = dijkstra(weighted, source)
    nx_graph = build_networkx_graph(weighted)

    layout_resolvers = {
        "spring": lambda g: nx.spring_layout(g, seed=layout_seed),
        "kamada_kawai": lambda g: nx.kamada_kawai_layout(g, weight="weight"),
        "circular": nx.circular_layout,
    }
    try:
        resolver = layout_resolvers[layout]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported layout '{layout}'. Choose from {sorted(layout_resolvers)}"
        ) from exc

    positions = resolver(nx_graph)
    edge_labels = {(u, v): data["weight"] for u, v, data in nx_graph.edges(data=True)}

    focus_path = result.path_to(target) if target is not None else None

    return VisualizationPayload(
        graph=nx_graph,
        positions=positions,
        edge_labels=edge_labels,
        distances=result.distances,
        source=source,
        target=target,
        focus_path=focus_path,
    )


def format_distances(result: ShortestPathResult) -> List[str]:
    """Render the shortest-path metrics as human-readable strings."""

    lines: List[str] = []
    for node, distance in sorted(result.distances.items()):
        if math.isinf(distance):
            lines.append(f"{result.source} -> {node}: unreachable")
            continue
        path = result.path_to(node)
        assert (
            path is not None
        )  # pragma: no cover - defensive; unreachable handled above
        path_str = " → ".join(path)
        lines.append(f"{result.source} -> {node}: cost={distance:.3f}, path={path_str}")
    return lines


def _normalize_graph(graph: GraphInput) -> Dict[str, Tuple[GraphEdge, ...]]:
    if not isinstance(graph, Mapping):
        raise GraphValidationError("Graph must be a mapping of node -> edges")
    adjacency: Dict[str, Tuple[GraphEdge, ...]] = {}
    pending_neighbors: Dict[str, List[GraphEdge]] = {}

    for node, edges in graph.items():
        if not isinstance(node, str):
            raise GraphValidationError("Node identifiers must be strings")
        normalized_edges: List[GraphEdge] = []
        for edge in edges:
            if not isinstance(edge, Sequence) or len(edge) != 2:
                raise GraphValidationError(
                    "Edges must be two-item sequences of (neighbor, weight)"
                )
            neighbor, weight = edge
            if not isinstance(neighbor, str):
                raise GraphValidationError("Neighbor identifiers must be strings")
            if not isinstance(weight, (int, float)):
                raise GraphValidationError("Edge weights must be numeric")
            if weight < 0:
                raise GraphValidationError("Edge weights must be non-negative")
            normalized_edges.append((neighbor, float(weight)))
            pending_neighbors.setdefault(neighbor, [])
        adjacency[node] = tuple(normalized_edges)

    for neighbor in pending_neighbors:
        adjacency.setdefault(neighbor, tuple())

    return adjacency
