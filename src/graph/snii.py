"""Semantic Node Importance Index (SNII) — Day 1 upgrade.

SNII is a formalization of the existing quality-aware importance score
computed by src.graph.importance.compute_importance(). It is NOT a second,
independent scoring system: SNII(n) == importance(n) by construction. This
module adds a documented formula, a percentage-contribution breakdown, a
deterministic ranking, and a grounded explanation string on top of the
already-computed ImportanceResult, without recomputing anything.

Formula (identical to compute_importance):

    SNII(n) = w_type * TypeWeight(n)
            + w_conn * Connectivity(n)
            + w_cent * Centrality(n)
            + w_prot * Protection(n)

All four components are normalized to [0, 1] by compute_importance(), and
the weights (defaults: 0.40 / 0.25 / 0.15 / 0.20) are validated there to sum
to 1.0, so SNII(n) is always in [0, 1].

SNII is the importance index proposed by this academic prototype. It is not
claimed to be a universally optimal measure of prompt-unit importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.graph.importance import ImportanceConfig, ImportanceResult, NodeImportance


class SNIIError(ValueError):
    """Raised when SNII cannot be computed (e.g. empty ImportanceResult)."""


@dataclass(frozen=True)
class SNIIScore:
    node_id: str
    unit_type: str
    snii: float                       # == NodeImportance.importance
    rank: int                         # 1 = highest SNII; ties broken by node_id
    total_nodes: int
    raw_components: Mapping[str, float]   # type_weight, connectivity, centrality, protection (each in [0,1])
    weighted_contributions: Mapping[str, float]  # weight * component, sums to snii
    percent_contributions: Mapping[str, float]   # weighted_contributions as % of snii

    def explain(self) -> str:
        lines = [
            f"SNII({self.node_id}) = {self.snii:.4f}  (rank {self.rank} of {self.total_nodes})",
            f"  Type: {self.unit_type}",
        ]
        for key in ("type", "connectivity", "centrality", "protection"):
            pct = self.percent_contributions[key]
            lines.append(
                f"  {key:<12} contributes {self.weighted_contributions[key]:.4f} "
                f"({pct:.1f}% of score)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "unit_type": self.unit_type,
            "snii": self.snii,
            "rank": self.rank,
            "total_nodes": self.total_nodes,
            "raw_components": dict(self.raw_components),
            "weighted_contributions": dict(self.weighted_contributions),
            "percent_contributions": dict(self.percent_contributions),
        }


@dataclass(frozen=True)
class SNIIResult:
    scores: dict[str, SNIIScore]
    config: ImportanceConfig

    def __getitem__(self, node_id: str) -> SNIIScore:
        return self.scores[node_id]

    def __len__(self) -> int:
        return len(self.scores)

    def ranked(self) -> list[SNIIScore]:
        """All scores sorted by SNII descending; ties broken by node id."""
        return sorted(self.scores.values(), key=lambda s: (-s.snii, s.node_id))

    def top(self, k: int) -> list[SNIIScore]:
        if k < 1:
            raise SNIIError("k must be >= 1.")
        return self.ranked()[:k]

    def to_dict(self) -> dict[str, Any]:
        return {"scores": {n: s.to_dict() for n, s in self.scores.items()}}


def compute_snii(importance: ImportanceResult) -> SNIIResult:
    """Wrap an already-computed ImportanceResult as a formal SNIIResult.

    Deterministic: identical ImportanceResult in -> identical SNIIResult out.

    Raises:
        TypeError: importance is not an ImportanceResult.
        SNIIError: importance has zero nodes.
    """
    if not isinstance(importance, ImportanceResult):
        raise TypeError(f"importance must be an ImportanceResult, got {type(importance).__name__}")
    if len(importance) == 0:
        raise SNIIError("Cannot compute SNII for an empty ImportanceResult.")

    total_nodes = len(importance)
    order = {r.node_id: i for i, r in enumerate(
        sorted(importance.scores.values(), key=lambda r: (-r.importance, r.node_id))
    )}

    scores: dict[str, SNIIScore] = {}
    for node_id, rec in importance.scores.items():
        scores[node_id] = _build_score(rec, order[node_id] + 1, total_nodes)
    return SNIIResult(scores=scores, config=importance.config)


def explain_snii(score: SNIIScore) -> str:
    """Grounded, per-component explanation for one node's SNII score."""
    return score.explain()


def _build_score(rec: NodeImportance, rank: int, total_nodes: int) -> SNIIScore:
    contributions = rec.contributions
    total = rec.importance
    percent = {
        k: (v / total * 100.0 if total > 0 else 0.0) for k, v in contributions.items()
    }
    raw = {
        "type": rec.type_weight,
        "connectivity": rec.connectivity,
        "centrality": rec.centrality,
        "protection": rec.protection,
    }
    return SNIIScore(
        node_id=rec.node_id,
        unit_type=rec.unit_type,
        snii=rec.importance,
        rank=rank,
        total_nodes=total_nodes,
        raw_components=raw,
        weighted_contributions=dict(contributions),
        percent_contributions=percent,
    )


# --------------------------------------------------------------------------
# Manual check:  python -m src.graph.snii
# Reuses the exact hand-made graph from test_day1.py's path_graph fixture,
# whose importance values are already pytest-verified.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import networkx as nx

    from src.graph.importance import compute_importance

    g = nx.Graph()
    for nid, t in (("N01", "instruction"), ("N02", "constraint"),
                   ("N03", "context"), ("N04", "other")):
        g.add_node(nid, node_id=nid, text=f"demo {nid}", type=t)
    g.add_edge("N01", "N02", weight=0.8)
    g.add_edge("N02", "N03", weight=0.6)

    result = compute_snii(compute_importance(g))
    for s in result.ranked():
        print(f"{s.node_id} | {s.unit_type:<12} | SNII {s.snii:.4f} | rank {s.rank} of {s.total_nodes}")
    print()
    print(explain_snii(result["N02"]))
    print()
    # Determinism check
    r2 = compute_snii(compute_importance(g))
    print("Deterministic:", [s.snii for s in result.ranked()] == [s.snii for s in r2.ranked()])