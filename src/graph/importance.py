"""Quality-aware node importance for the semantic prompt graph.

Input : networkx.Graph from builder.build_semantic_graph
Output: ImportanceResult (one NodeImportance record per node)

    importance = w_type*type_weight + w_conn*connectivity
               + w_cent*centrality  + w_prot*protection      (all in [0, 1])

  type_weight  : fixed prior per unit type (design assumption, configurable)
  connectivity : weighted degree / max weighted degree in the graph
  centrality   : betweenness (unweighted) / max betweenness in the graph
  protection   : 1.0 if the type is in protected_types, else 0.0

Deterministic and explainable. No compression happens here (Day 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import networkx as nx

from src.extraction.unit_classifier import UNIT_TYPES


DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "instruction": 1.00,
    "constraint": 0.90,
    "requirement": 0.90,
    "output_format": 0.85,
    "role": 0.70,
    "example": 0.50,
    "context": 0.50,
    "other": 0.20,
}


class ImportanceError(ValueError):
    """Raised when importance cannot be computed."""


@dataclass(frozen=True)
class ImportanceConfig:
    w_type: float = 0.40
    w_connectivity: float = 0.25
    w_centrality: float = 0.15
    w_protection: float = 0.20
    type_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS))
    protected_types: frozenset[str] = frozenset({"constraint", "requirement", "output_format"})

    def __post_init__(self) -> None:
        weights = (self.w_type, self.w_connectivity, self.w_centrality, self.w_protection)
        if any(w < 0 for w in weights):
            raise ImportanceError("Importance weights must be non-negative.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ImportanceError(f"Importance weights must sum to 1.0, got {sum(weights):.4f}.")
        missing = set(UNIT_TYPES) - set(self.type_weights)
        if missing:
            raise ImportanceError(f"type_weights is missing types: {sorted(missing)}")
        if any(not 0.0 <= v <= 1.0 for v in self.type_weights.values()):
            raise ImportanceError("type_weights values must be in [0, 1].")
        unknown = set(self.protected_types) - set(UNIT_TYPES)
        if unknown:
            raise ImportanceError(f"Unknown protected types: {sorted(unknown)}")


@dataclass(frozen=True)
class NodeImportance:
    node_id: str
    text: str
    unit_type: str
    importance: float
    connectivity: float          # normalized weighted degree, [0, 1]
    centrality: float            # normalized betweenness, [0, 1]
    degree: int
    degree_centrality: float     # degree / (n - 1), for reporting
    betweenness: float           # raw (unnormalized-by-max) betweenness
    type_weight: float
    protection: float
    contributions: dict[str, float]

    def explain(self) -> str:
        c = self.contributions
        return "\n".join([
            f"Node {self.node_id}",
            f"Type: {self.unit_type}",
            f"Importance: {self.importance:.2f}",
            f"Centrality: {self.centrality:.2f}",
            f"Connectivity: {self.connectivity:.2f} (degree {self.degree})",
            f"Type weight: {self.type_weight:.2f}",
            f"Protected: {'yes' if self.protection > 0 else 'no'}",
            f"Breakdown: type +{c['type']:.2f}, connectivity +{c['connectivity']:.2f}, "
            f"centrality +{c['centrality']:.2f}, protection +{c['protection']:.2f}",
        ])


@dataclass(frozen=True)
class ImportanceResult:
    scores: dict[str, NodeImportance]
    config: ImportanceConfig

    def __getitem__(self, node_id: str) -> NodeImportance:
        return self.scores[node_id]

    def __len__(self) -> int:
        return len(self.scores)

    def ranked(self) -> list[NodeImportance]:
        """Nodes sorted by importance (desc); ties broken by node id."""
        return sorted(self.scores.values(), key=lambda r: (-r.importance, r.node_id))

    def top(self, k: int) -> list[NodeImportance]:
        if k < 1:
            raise ImportanceError("k must be >= 1.")
        return self.ranked()[:k]


def compute_importance(
    graph: nx.Graph, config: ImportanceConfig | None = None
) -> ImportanceResult:
    """Compute importance for every node of the semantic graph.

    Raises:
        ImportanceError: empty graph, or a node with a missing/unknown type.
    """
    cfg = config or ImportanceConfig()
    n = graph.number_of_nodes()
    if n == 0:
        raise ImportanceError("Cannot compute importance for an empty graph.")

    for node, data in graph.nodes(data=True):
        t = data.get("type")
        if t is None:
            raise ImportanceError(f"Node {node!r} has no 'type' attribute.")
        if t not in cfg.type_weights:
            raise ImportanceError(f"Node {node!r} has unknown type {t!r}.")

    strength = dict(graph.degree(weight="weight"))
    max_strength = max(strength.values())
    betweenness = nx.betweenness_centrality(graph, normalized=True)   # unweighted
    max_bet = max(betweenness.values())

    scores: dict[str, NodeImportance] = {}
    for node, data in graph.nodes(data=True):
        utype = data["type"]
        degree = graph.degree(node)
        connectivity = strength[node] / max_strength if max_strength > 0 else 0.0
        centrality = betweenness[node] / max_bet if max_bet > 0 else 0.0
        type_weight = cfg.type_weights[utype]
        protection = 1.0 if utype in cfg.protected_types else 0.0

        contributions = {
            "type": cfg.w_type * type_weight,
            "connectivity": cfg.w_connectivity * connectivity,
            "centrality": cfg.w_centrality * centrality,
            "protection": cfg.w_protection * protection,
        }
        scores[node] = NodeImportance(
            node_id=node,
            text=data.get("text", ""),
            unit_type=utype,
            importance=sum(contributions.values()),
            connectivity=connectivity,
            centrality=centrality,
            degree=degree,
            degree_centrality=(degree / (n - 1)) if n > 1 else 0.0,
            betweenness=betweenness[node],
            type_weight=type_weight,
            protection=protection,
            contributions=contributions,
        )
    return ImportanceResult(scores=scores, config=cfg)


def annotate_graph(graph: nx.Graph, result: ImportanceResult) -> nx.Graph:
    """Copy importance values onto the graph's nodes (in place) and return it."""
    for node, rec in result.scores.items():
        graph.nodes[node].update(
            importance=rec.importance,
            connectivity=rec.connectivity,
            centrality=rec.centrality,
            protection=rec.protection,
            snii=rec.importance,  
        )
    return graph


# --------------------------------------------------------------------------
# Manual check:  python -m src.graph.importance
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Hand-made graph to verify the formula (NOT part of the pipeline).
    g = nx.Graph()
    for nid, t in (("N01", "instruction"), ("N02", "constraint"),
                   ("N03", "context"), ("N04", "other")):
        g.add_node(nid, node_id=nid, text=f"demo {nid}", type=t)
    g.add_edge("N01", "N02", weight=0.8)
    g.add_edge("N02", "N03", weight=0.6)

    result = compute_importance(g)
    for r in result.ranked():
        print(f"{r.node_id} | {r.unit_type:<12} | importance {r.importance:.2f} | "
              f"connectivity {r.connectivity:.2f} | centrality {r.centrality:.2f}")
    print()
    print(result["N02"].explain())