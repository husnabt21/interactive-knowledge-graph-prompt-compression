"""Graph-level metrics for the semantic prompt graph.

Input : networkx.Graph from builder.build_semantic_graph
Output: GraphMetrics

Metrics that are undefined for the given graph (e.g. average shortest path
of a disconnected graph) are returned as None, with the reason shown in
summary(). Nothing is estimated or fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


class GraphMetricsError(ValueError):
    """Raised when metrics cannot be computed for the given graph."""


@dataclass(frozen=True)
class GraphMetrics:
    num_nodes: int
    num_edges: int
    density: float
    average_degree: float
    num_components: int
    component_sizes: tuple[int, ...]              # sorted, largest first
    largest_component_size: int
    is_connected: bool
    isolated_nodes: tuple[str, ...]
    average_clustering: float
    average_edge_weight: float | None             # None if no edges
    average_shortest_path_length: float | None    # None unless connected and n >= 2
    diameter: int | None                          # None unless connected and n >= 2
    type_counts: dict[str, int]
    degree_centrality: dict[str, float] = field(default_factory=dict)
    betweenness_centrality: dict[str, float] = field(default_factory=dict)
    top_by_degree: tuple[tuple[str, float], ...] = ()
    top_by_betweenness: tuple[tuple[str, float], ...] = ()

    def _path_reason(self) -> str:
        if self.num_nodes < 2:
            return "fewer than 2 nodes"
        return "graph not connected"

    def summary(self) -> str:
        """Readable multi-line summary."""
        sizes = ", ".join(str(s) for s in self.component_sizes)
        isolated = ", ".join(self.isolated_nodes) if self.isolated_nodes else "none"
        weight = (f"{self.average_edge_weight:.2f}"
                  if self.average_edge_weight is not None else "n/a (no edges)")
        if self.average_shortest_path_length is not None:
            path = f"{self.average_shortest_path_length:.2f}"
            diam = str(self.diameter)
        else:
            path = f"n/a ({self._path_reason()})"
            diam = f"n/a ({self._path_reason()})"
        top_deg = ", ".join(f"{n} ({v:.2f})" for n, v in self.top_by_degree)
        top_bet = ", ".join(f"{n} ({v:.2f})" for n, v in self.top_by_betweenness)
        types = ", ".join(f"{t}={c}" for t, c in self.type_counts.items())
        return "\n".join([
            f"Nodes: {self.num_nodes}",
            f"Edges: {self.num_edges}",
            f"Density: {self.density:.3f}",
            f"Average degree: {self.average_degree:.2f}",
            f"Components: {self.num_components} (sizes: {sizes})",
            f"Isolated nodes: {isolated}",
            f"Average clustering: {self.average_clustering:.3f}",
            f"Average edge weight: {weight}",
            f"Average shortest path: {path}",
            f"Diameter: {diam}",
            f"Top degree centrality: {top_deg}",
            f"Top betweenness: {top_bet}",
            f"Node types: {types}",
        ])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (for saving, or the Day 3 UI)."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.density,
            "average_degree": self.average_degree,
            "num_components": self.num_components,
            "component_sizes": list(self.component_sizes),
            "largest_component_size": self.largest_component_size,
            "is_connected": self.is_connected,
            "isolated_nodes": list(self.isolated_nodes),
            "average_clustering": self.average_clustering,
            "average_edge_weight": self.average_edge_weight,
            "average_shortest_path_length": self.average_shortest_path_length,
            "diameter": self.diameter,
            "type_counts": dict(self.type_counts),
            "degree_centrality": dict(self.degree_centrality),
            "betweenness_centrality": dict(self.betweenness_centrality),
        }


def compute_graph_metrics(graph: nx.Graph, top_k: int = 3) -> GraphMetrics:
    """Compute graph-level metrics.

    Raises:
        TypeError: ``graph`` is not a networkx Graph.
        GraphMetricsError: empty graph, missing edge weight, or top_k < 1.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}.")
    n = graph.number_of_nodes()
    if n == 0:
        raise GraphMetricsError("Cannot compute metrics for an empty graph.")
    if top_k < 1:
        raise GraphMetricsError("top_k must be >= 1.")

    m = graph.number_of_edges()

    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    connected = len(components) == 1
    isolated = tuple(node for node in graph.nodes if graph.degree(node) == 0)

    if m > 0:
        try:
            weights = [d["weight"] for _, _, d in graph.edges(data=True)]
        except KeyError as exc:
            raise GraphMetricsError("An edge is missing the 'weight' attribute.") from exc
        avg_weight: float | None = sum(weights) / len(weights)
    else:
        avg_weight = None

    if connected and n >= 2:
        avg_path: float | None = nx.average_shortest_path_length(graph)
        diameter: int | None = nx.diameter(graph)
    else:
        avg_path, diameter = None, None

    # nx.degree_centrality returns 1.0 for a single node; report 0.0 instead.
    degree_c = ({node: 0.0 for node in graph.nodes} if n < 2
                else nx.degree_centrality(graph))
    between_c = nx.betweenness_centrality(graph, normalized=True)

    def top(scores: dict[str, float]) -> tuple[tuple[str, float], ...]:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return tuple(ranked[:top_k])

    type_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return GraphMetrics(
        num_nodes=n,
        num_edges=m,
        density=nx.density(graph),
        average_degree=2.0 * m / n,
        num_components=len(components),
        component_sizes=tuple(components),
        largest_component_size=components[0],
        is_connected=connected,
        isolated_nodes=isolated,
        average_clustering=nx.average_clustering(graph),
        average_edge_weight=avg_weight,
        average_shortest_path_length=avg_path,
        diameter=diameter,
        type_counts=type_counts,
        degree_centrality=dict(degree_c),
        betweenness_centrality=dict(between_c),
        top_by_degree=top(degree_c),
        top_by_betweenness=top(between_c),
    )


# --------------------------------------------------------------------------
# Manual check:  python -m src.graph.metrics
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Hand-made graph to verify the formulas (NOT part of the pipeline).
    g = nx.Graph()
    for nid, t in (("N01", "instruction"), ("N02", "constraint"),
                   ("N03", "context"), ("N04", "other")):
        g.add_node(nid, node_id=nid, text=f"demo {nid}", type=t)
    g.add_edge("N01", "N02", weight=0.8)
    g.add_edge("N02", "N03", weight=0.6)

    print(compute_graph_metrics(g).summary())