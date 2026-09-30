"""Node explanations for the Day 3 UI (Step 2).

Presentation only. Combines Day 1 node attributes (importance, connectivity,
centrality, Day 1 protection score, degree) with the Day 2 NodeDecision
(decision, reason, reason_codes, duplicate info). Nothing is recalculated:
the Day 2 ``reason`` is passed through verbatim, and every bullet is derived
from an actual reason code and the real numbers on the decision, so an
explanation cannot contradict the decision.

Importance is reported as a value plus its rank among all nodes rather than
with qualitative labels such as "high".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from src.compression.compresser import CompressionResult, Decision, NodeDecision

from src.graph.snii import SNIIScore, compute_snii
from src.graph.importance import ImportanceResult

NA = "N/A"


class ExplanationError(ValueError):
    """A node cannot be explained (unknown id, or graph/result mismatch)."""


# --------------------------------------------------------------------------
# Result structure
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeExplanation:
    node_id: str
    text: str
    unit_type: str
    kind: Optional[str]
    index: int
    parent_heading: Optional[str]
    decision: str                          # "RETAIN" or "REMOVE"
    status: str                            # "Retained" / "Removed" / "Removed (near-duplicate)"
    headline: str                          # "Why was this node retained?" / "...removed?"
    reason: str                            # Day 2 reason, verbatim
    reason_codes: tuple[str, ...]
    bullets: tuple[str, ...]               # derived from reason codes + real values
    importance: float
    priority: float
    redundancy: float
    protected: bool                        # Day 2 rule: strongly protected unit type
    connectivity: Optional[float]          # Day 1 (None if graph not annotated)
    centrality: Optional[float]
    protection_score: Optional[float]      # Day 1 score signal, not the Day 2 rule
    degree: int
    importance_rank: int                   # 1 = most important; ties broken by prompt order
    duplicate_of: Optional[str]
    duplicate_similarity: Optional[float]
    represents: tuple[str, ...]            # removed near-duplicates this node stands for
    neighbors: tuple[tuple[str, float], ...]   # (node_id, similarity), strongest first
    snii: float
    snii_rank: int
    snii_breakdown: tuple[str, ...]   # percentage-contribution lines; empty if not available

    def details_rows(self) -> list[tuple[str, str]]:
        """(label, value) pairs for the selected-node details panel."""
        sim = self.duplicate_similarity
        return [
            ("Node ID", self.node_id),
            ("Type", self.unit_type),
            ("SNII", f"{self.snii:.4f} (rank {self.snii_rank} of {self.total_nodes if hasattr(self, 'total_nodes') else '?'})"),
            ("Kind", self.kind or NA),
            ("Index", str(self.index)),
            ("Section", self.parent_heading or NA),
            ("Importance", f"{self.importance:.3f}"),
            ("Priority", f"{self.priority:.3f}"),
            ("Connectivity", _fmt(self.connectivity)),
            ("Centrality", _fmt(self.centrality)),
            ("Degree", str(self.degree)),
            ("Protection score (Day 1)", _fmt(self.protection_score)),
            ("Protected (Day 2 rule)", "Yes" if self.protected else "No"),
            ("Redundancy", f"{self.redundancy:.3f}"),
            ("Decision", self.decision),
            ("Reason", self.reason),
            ("Duplicate of", self.duplicate_of or NA),
            ("Duplicate similarity", _fmt(sim)),
            ("Represents", ", ".join(self.represents) if self.represents else NA),
            ("Connected to", ", ".join(f"{n} ({w:.2f})" for n, w in self.neighbors) or NA),
        ]

    def to_text(self) -> str:
        """Plain-text explanation for console output."""
        lines = [
            f"{self.node_id} [{self.unit_type}] {self.status.upper()}",
            self.headline,
            f"  {self.reason}",
        ]
        lines.extend(f"  - {b}" for b in self.bullets)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "unit_type": self.unit_type,
            "kind": self.kind,
            "index": self.index,
            "parent_heading": self.parent_heading,
            "decision": self.decision,
            "status": self.status,
            "headline": self.headline,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "bullets": list(self.bullets),
            "importance": self.importance,
            "priority": self.priority,
            "redundancy": self.redundancy,
            "protected": self.protected,
            "connectivity": self.connectivity,
            "centrality": self.centrality,
            "protection_score": self.protection_score,
            "degree": self.degree,
            "importance_rank": self.importance_rank,
            "duplicate_of": self.duplicate_of,
            "duplicate_similarity": self.duplicate_similarity,
            "represents": list(self.represents),
            "neighbors": [{"node_id": n, "similarity": w} for n, w in self.neighbors],
        }


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------
def status_label(decision: NodeDecision) -> str:
    """Display label; matches the legend labels used by graph_render."""
    if decision.decision is Decision.RETAIN:
        return "Retained"
    return "Removed (near-duplicate)" if decision.duplicate_of is not None else "Removed"


def explain_node(
    graph: nx.Graph, result: CompressionResult, node_id: str,
    importance: Optional[ImportanceResult] = None,
) -> NodeExplanation:
    _validate(graph, result)
    return _explain(graph, result, node_id, _importance_ranks(result), importance)


def explain_all(
    graph: nx.Graph, result: CompressionResult,
    importance: Optional[ImportanceResult] = None,
) -> dict[str, NodeExplanation]:
    _validate(graph, result)
    ranks = _importance_ranks(result)
    return {d.node_id: _explain(graph, result, d.node_id, ranks, importance) for d in result.decisions}


def search_nodes(result: CompressionResult, query: str) -> list[str]:
    """Node ids whose id, text or unit type contains ``query`` (case-insensitive).

    Returned in original prompt order. A blank query returns an empty list.
    """
    if not isinstance(result, CompressionResult):
        raise TypeError(f"result must be a CompressionResult, got {type(result).__name__}")
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    q = query.strip().lower()
    if not q:
        return []
    return [
        d.node_id for d in result.decisions
        if q in d.node_id.lower() or q in d.text.lower() or q in d.unit_type.lower()
    ]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _fmt(value: Optional[float], digits: int = 3) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _validate(graph: Any, result: Any) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"graph must be a networkx.Graph, got {type(graph).__name__}")
    if not isinstance(result, CompressionResult):
        raise TypeError(f"result must be a CompressionResult, got {type(result).__name__}")
    if {d.node_id for d in result.decisions} != set(graph.nodes):
        raise ExplanationError(
            "graph nodes and compression decisions do not match; "
            "explain the same graph that was compressed"
        )


def _importance_ranks(result: CompressionResult) -> dict[str, int]:
    ordered = sorted(result.decisions, key=lambda d: (-d.importance, d.index))
    return {d.node_id: i + 1 for i, d in enumerate(ordered)}


def _explain(
    graph: nx.Graph,
    result: CompressionResult,
    node_id: str,
    ranks: dict[str, int],
    importance: Optional[ImportanceResult] = None,
) -> NodeExplanation:
    if node_id not in graph:
        raise ExplanationError(f"unknown node id {node_id!r}")
    d = result.decision_for(node_id)
    data = graph.nodes[node_id]
    represents = tuple(x.node_id for x in result.decisions if x.duplicate_of == node_id)
    neighbors = tuple(sorted(
        ((str(m), float(attrs.get("weight", 0.0))) for m, attrs in graph[node_id].items()),
        key=lambda t: (-t[1], t[0]),
    ))
    connectivity = _opt_float(data.get("connectivity"))
    centrality = _opt_float(data.get("centrality"))
    degree = int(graph.degree(node_id))
    rank = ranks[node_id]

    bullets = list(_reason_bullets(d, result, represents))
    bullets.append(f"Importance {d.importance:.3f} (rank {rank} of {result.original_node_count})")
    if connectivity is not None and centrality is not None:
        bullets.append(
            f"Connectivity {connectivity:.3f}, centrality {centrality:.3f}, degree {degree}"
        )
    else:
        bullets.append(f"Degree {degree}")

    snii_value = d.importance                # SNII == importance by construction
    snii_rank_value = rank                    # same deterministic rank
    snii_breakdown: tuple[str, ...] = ()
    if importance is not None:
        snii_result = compute_snii(importance)
        if node_id in snii_result.scores:
            score = snii_result[node_id]
            snii_value = score.snii
            snii_rank_value = score.rank
            snii_breakdown = tuple(
                f"{k}: {score.weighted_contributions[k]:.4f} ({score.percent_contributions[k]:.1f}%)"
                for k in ("type", "connectivity", "centrality", "protection")
            )

    retained = d.decision is Decision.RETAIN
    return NodeExplanation(
        node_id=d.node_id,
        text=d.text,
        unit_type=d.unit_type,
        kind=d.kind,
        index=d.index,
        parent_heading=data.get("parent_heading"),
        decision=d.decision.value,
        status=status_label(d),
        headline="Why was this node retained?" if retained else "Why was this node removed?",
        reason=d.reason,
        reason_codes=d.reason_codes,
        bullets=tuple(bullets),
        importance=d.importance,
        priority=d.priority,
        redundancy=d.redundancy,
        protected=d.protected,
        connectivity=connectivity,
        centrality=centrality,
        protection_score=_opt_float(data.get("protection")),
        degree=degree,
        importance_rank=rank,
        duplicate_of=d.duplicate_of,
        duplicate_similarity=d.duplicate_similarity,
        represents=represents,
        neighbors=neighbors,
        snii=snii_value,
        snii_rank=snii_rank_value,
        snii_breakdown=snii_breakdown,
    )


def _reason_bullets(
    d: NodeDecision, result: CompressionResult, represents: tuple[str, ...]
) -> list[str]:
    """One or two bullets per Day 2 reason code, using only real values."""
    total = result.original_node_count
    target = result.target_node_count
    out: list[str] = []
    for code in d.reason_codes:
        if code == "PROTECTED_TYPE":
            out.append(f"Protected unit type: {d.unit_type}")
        elif code == "NEAR_DUPLICATE_REPRESENTATIVE":
            if represents:
                out.append(
                    f"Representative of a near-duplicate group (removed: {', '.join(represents)})"
                )
            else:
                out.append("Representative of a near-duplicate group")
        elif code == "WITHIN_RETENTION_TARGET":
            out.append(f"Within the retention target ({target} of {total} nodes)")
        elif code == "NEAR_DUPLICATE_REMOVED":
            if d.duplicate_of is not None:
                sim = _fmt(d.duplicate_similarity)
                out.append(f"Near-duplicate of {d.duplicate_of} (similarity {sim})")
                rep = result.decision_for(d.duplicate_of)
                out.append(
                    f"{rep.node_id} was kept as the representative "
                    f"(priority {rep.priority:.3f} vs {d.priority:.3f})"
                )
        elif code == "BELOW_RETENTION_TARGET":
            out.append(
                f"Ranked among the lowest removable units "
                f"(importance {d.importance:.3f}, redundancy {d.redundancy:.3f})"
            )
            out.append(f"Retention target: {target} of {total} nodes")
        else:
            out.append(f"Reason code: {code}")
    return out


# --------------------------------------------------------------------------
# Manual check:  python -m src.visualization.explain
# Hand-made graph, no models needed.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    from src.compression.compresser import CompressionConfig, compress_graph

    g = nx.Graph(near_duplicate_pairs=[("N02", "N03", 0.95)])
    rows = [
        ("N01", 0, "You are a travel planner.", "role", 0.55, 1),
        ("N02", 1, "Plan a 3-day trip to Tokyo.", "instruction", 0.80, 3),
        ("N03", 2, "Plan a three day trip to Tokyo.", "instruction", 0.70, 4),
        ("N04", 3, "Tokyo has many museums.", "context", 0.40, 5),
        ("N05", 4, "Budget must not exceed $500.", "constraint", 0.85, 7),
        ("N06", 5, "Thanks a lot.", "other", 0.10, 9),
        ("N07", 6, "e.g. Visit Ueno Park.", "example", 0.35, 10),
    ]
    for nid, idx, text, utype, imp, line in rows:
        g.add_node(
            nid, node_id=nid, text=text, type=utype, importance=imp, index=idx,
            kind="sentence", line_start=line, connectivity=0.5, centrality=0.25,
            protection=1.0 if utype in ("constraint", "requirement", "output_format") else 0.0,
        )
    g.add_edge("N02", "N03", weight=0.95, near_duplicate=True)
    g.add_edge("N02", "N04", weight=0.40, near_duplicate=False)
    g.add_edge("N03", "N04", weight=0.38, near_duplicate=False)
    g.add_edge("N04", "N07", weight=0.50, near_duplicate=False)
    g.add_edge("N01", "N02", weight=0.35, near_duplicate=False)

    result = compress_graph(g, CompressionConfig(retention_ratio=0.60))

    for nid in ("N02", "N03", "N06"):
        print(explain_node(g, result, nid).to_text())
        print()

    print("[Details panel rows for N03]")
    for label, value in explain_node(g, result, "N03").details_rows():
        print(f"{label}: {value}")

    print()
    for q in ("tokyo", "constraint", "  "):
        print(f"Search {q!r}: {search_nodes(result, q)}")

    everything = explain_all(g, result)
    json.dumps([e.to_dict() for e in everything.values()])
    print(f"JSON-safe: True ({len(everything)} explanations)")

    try:
        explain_node(g, result, "N99")
    except ExplanationError as exc:
        print("Unknown node ->", exc)