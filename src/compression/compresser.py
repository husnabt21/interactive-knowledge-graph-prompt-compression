"""Quality-aware, graph-aware prompt compression (Day 2, Step 2).

Input : annotated Day 1 graph (needs node attrs: text, type, importance, index;
        optional: kind, line_start; near-duplicate info on edges / graph.graph)
Output: CompressionResult (decisions, reasons, compressed prompt)

Algorithm (deterministic, no ML, no LLM)::

    Stage A  redundancy : visit nodes by priority (desc, then original index).
             A node with a near-duplicate edge to an already-chosen
             representative is removed as redundant. Otherwise it becomes a
             representative. Both nodes of a pair are never removed.
    Stage B  budget     : target = ceil(retention_ratio * N). If more nodes
             survive than the target, remove the lowest-ranked *removable*
             nodes by  adjusted = priority - redundancy_penalty * redundancy,
             where redundancy = highest similarity to another surviving node.
             Locked nodes are never removed here:
               - strongly protected types (if config.protect_strong)
               - representatives of a removed near-duplicate group
    Rebuild  : retained nodes are sorted by node["index"], never by score.

The requested retention ratio is a target, not a guarantee: locked nodes can
keep the actual ratio higher, and Stage A can push it lower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

import networkx as nx

from src.compression.protection_rules import (
    DEFAULT_POLICY,
    InvalidGraphError,
    ProtectionAssessment,
    ProtectionPolicy,
    assess_graph,
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class CompressionError(Exception):
    """Base class for compression errors."""


class InvalidRetentionRatioError(CompressionError, ValueError):
    """retention_ratio is not a number in (0, 1]."""


class InvalidConfigError(CompressionError, ValueError):
    """A CompressionConfig field is invalid."""


class EmptyGraphError(CompressionError, ValueError):
    """The graph has no nodes, so there is nothing to compress."""


# --------------------------------------------------------------------------
# Config and results
# --------------------------------------------------------------------------
class Decision(str, Enum):
    RETAIN = "RETAIN"
    REMOVE = "REMOVE"


@dataclass(frozen=True)
class CompressionConfig:
    retention_ratio: float = 0.70
    protect_strong: bool = True        # lock strong types out of Stage B
    remove_near_duplicates: bool = True
    redundancy_penalty: float = 0.10   # weight of redundancy in Stage B ranking
    list_marker: str = "- "            # prefix restored for list_item nodes on rebuild
    policy: ProtectionPolicy = DEFAULT_POLICY

    def __post_init__(self) -> None:
        r = self.retention_ratio
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0.0 < r <= 1.0:
            raise InvalidRetentionRatioError(
                f"retention_ratio must be a number in (0, 1], got {r!r}"
            )
        p = self.redundancy_penalty
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0:
            raise InvalidConfigError(f"redundancy_penalty must be finite and >= 0, got {p!r}")
        if not isinstance(self.list_marker, str):
            raise InvalidConfigError("list_marker must be a string")
        if not isinstance(self.policy, ProtectionPolicy):
            raise InvalidConfigError("policy must be a ProtectionPolicy")


@dataclass(frozen=True)
class NodeDecision:
    """The decision and its explanation for one node."""

    node_id: str
    index: int
    decision: Decision
    unit_type: str
    kind: Optional[str]
    text: str
    importance: float
    priority: float                 # importance + protection level bonus
    protected: bool                 # strongly protected type
    redundancy: float               # max similarity to another surviving node
    reason_codes: tuple[str, ...]
    reason: str
    duplicate_of: Optional[str] = None
    duplicate_similarity: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "index": self.index,
            "decision": self.decision.value,
            "unit_type": self.unit_type,
            "kind": self.kind,
            "text": self.text,
            "importance": self.importance,
            "priority": self.priority,
            "protected": self.protected,
            "redundancy": self.redundancy,
            "reason_codes": list(self.reason_codes),
            "reason": self.reason,
            "duplicate_of": self.duplicate_of,
            "duplicate_similarity": self.duplicate_similarity,
        }


@dataclass(frozen=True)
class CompressionResult:
    original_prompt: str
    compressed_prompt: str
    requested_retention_ratio: float
    actual_retention_ratio: float
    target_node_count: int
    retained_node_ids: tuple[str, ...]      # original prompt order
    removed_node_ids: tuple[str, ...]       # original prompt order
    decisions: tuple[NodeDecision, ...]     # original prompt order, all nodes
    warnings: tuple[str, ...] = ()

    @property
    def original_node_count(self) -> int:
        return len(self.decisions)

    @property
    def retained_node_count(self) -> int:
        return len(self.retained_node_ids)

    @property
    def removed_node_count(self) -> int:
        return len(self.removed_node_ids)

    @property
    def retained_units(self) -> tuple[NodeDecision, ...]:
        return tuple(d for d in self.decisions if d.decision is Decision.RETAIN)

    @property
    def removed_units(self) -> tuple[NodeDecision, ...]:
        return tuple(d for d in self.decisions if d.decision is Decision.REMOVE)

    def decision_for(self, node_id: str) -> NodeDecision:
        for d in self.decisions:
            if d.node_id == node_id:
                return d
        raise KeyError(f"no decision for node {node_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "compressed_prompt": self.compressed_prompt,
            "requested_retention_ratio": self.requested_retention_ratio,
            "actual_retention_ratio": self.actual_retention_ratio,
            "target_node_count": self.target_node_count,
            "original_node_count": self.original_node_count,
            "retained_node_count": self.retained_node_count,
            "removed_node_count": self.removed_node_count,
            "retained_node_ids": list(self.retained_node_ids),
            "removed_node_ids": list(self.removed_node_ids),
            "decisions": [d.to_dict() for d in self.decisions],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------
def reconstruct_prompt(
    graph: nx.Graph,
    node_ids: Iterable[str],
    list_marker: str = "- ",
) -> str:
    """Rebuild prompt text from nodes, ordered by ``node["index"]``.

    Node text is used as-is: code_block nodes keep their raw fenced text.
    list_item nodes do not store their bullet, so ``list_marker`` is prefixed
    to restore list structure (pass "" to disable). Separators: a single space
    for units starting on the same line, a newline between consecutive list
    items, otherwise a blank line.
    """
    ids = set(node_ids)
    unknown = [n for n in ids if n not in graph]
    if unknown:
        raise InvalidGraphError(f"unknown node ids: {sorted(unknown)}")

    ordered = sorted(ids, key=lambda n: graph.nodes[n]["index"])
    parts: list[str] = []
    prev: Optional[dict[str, Any]] = None
    for n in ordered:
        data = graph.nodes[n]
        text = data["text"]
        if data.get("kind") == "list_item":
            text = f"{list_marker}{text}"
        if prev is not None:
            parts.append(_separator(prev, data))
        parts.append(text)
        prev = data
    return "".join(parts)


def compress_graph(
    graph: nx.Graph,
    config: Optional[CompressionConfig] = None,
    original_prompt: Optional[str] = None,
) -> CompressionResult:
    """Compress an annotated Day 1 graph.

    Args:
        graph: output of build_semantic_graph() + annotate_graph().
        config: compression settings (defaults: retention 0.70).
        original_prompt: text to report as the original (e.g. the cleaned
            prompt). If omitted, it is rebuilt from all nodes.

    Raises:
        EmptyGraphError, InvalidGraphError, InvalidRetentionRatioError,
        InvalidConfigError, InvalidUnitTypeError (from protection rules).
    """
    cfg = config if config is not None else CompressionConfig()
    if not isinstance(cfg, CompressionConfig):
        raise TypeError(f"config must be CompressionConfig, got {type(cfg).__name__}")
    if original_prompt is not None and not isinstance(original_prompt, str):
        raise TypeError("original_prompt must be a string or None")

    _validate_graph(graph)
    assess = assess_graph(graph, cfg.policy)
    index = {n: graph.nodes[n]["index"] for n in graph.nodes}
    all_ids = sorted(graph.nodes, key=index.__getitem__)
    total = len(all_ids)

    # ---- Stage A: near-duplicate redundancy ----
    dup_of = _find_duplicates(graph, assess, index) if cfg.remove_near_duplicates else {}
    survivors = [n for n in all_ids if n not in dup_of]
    survivor_set = set(survivors)
    represented: dict[str, list[str]] = {}
    for dup in all_ids:
        if dup in dup_of:
            represented.setdefault(dup_of[dup][0], []).append(dup)

    redundancy = {n: _redundancy(graph, n, survivor_set) for n in survivors}
    adjusted = {
        n: assess[n].priority - cfg.redundancy_penalty * redundancy[n] for n in survivors
    }
    rank = {
        n: i + 1
        for i, n in enumerate(sorted(survivors, key=lambda n: (-adjusted[n], index[n])))
    }

    # ---- Stage B: retention budget ----
    target = max(1, math.ceil(round(cfg.retention_ratio * total, 9)))
    locked = {
        n for n in survivors
        if (cfg.protect_strong and assess[n].protected) or n in represented
    }
    excess = len(survivors) - target
    removed_by_budget: set[str] = set()
    if excess > 0:
        removable = sorted(
            (n for n in survivors if n not in locked),
            key=lambda n: (adjusted[n], -index[n]),   # lowest score first; ties: later node first
        )
        removed_by_budget = set(removable[:excess])

    # ---- decisions (original order) ----
    decisions: list[NodeDecision] = []
    for n in all_ids:
        if n in dup_of:
            decisions.append(_duplicate_decision(graph, assess[n], dup_of[n], assess, index))
        elif n in removed_by_budget:
            decisions.append(
                _budget_removal(graph, assess[n], adjusted[n], redundancy[n], target, total, excess)
            )
        else:
            decisions.append(
                _retained(
                    graph, assess[n], adjusted[n], redundancy[n], rank[n], len(survivors),
                    target, protected_lock=cfg.protect_strong and assess[n].protected,
                    removed_dups=represented.get(n, []),
                )
            )

    retained_ids = tuple(d.node_id for d in decisions if d.decision is Decision.RETAIN)
    removed_ids = tuple(d.node_id for d in decisions if d.decision is Decision.REMOVE)

    warnings: list[str] = []
    if len(retained_ids) > target:
        warnings.append(
            f"Retention target of {target} node(s) not reached: {len(locked)} node(s) are "
            f"locked (protected type or near-duplicate representative); "
            f"{len(retained_ids)} retained."
        )
    elif len(retained_ids) < target:
        warnings.append(
            f"{len(retained_ids)} node(s) retained, fewer than the target of {target}, "
            f"because {len(dup_of)} near-duplicate node(s) were removed as redundant."
        )

    original = (
        original_prompt
        if original_prompt is not None
        else reconstruct_prompt(graph, all_ids, cfg.list_marker)
    )
    return CompressionResult(
        original_prompt=original,
        compressed_prompt=reconstruct_prompt(graph, retained_ids, cfg.list_marker),
        requested_retention_ratio=float(cfg.retention_ratio),
        actual_retention_ratio=len(retained_ids) / total,
        target_node_count=target,
        retained_node_ids=retained_ids,
        removed_node_ids=removed_ids,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _validate_graph(graph: Any) -> None:
    if not isinstance(graph, nx.Graph):
        raise InvalidGraphError(f"expected a networkx.Graph, got {type(graph).__name__}")
    if graph.number_of_nodes() == 0:
        raise EmptyGraphError("cannot compress an empty graph")
    seen: dict[int, str] = {}
    for key, data in graph.nodes(data=True):
        if not isinstance(data.get("text"), str):
            raise InvalidGraphError(f"node {key!r} is missing a string 'text' attribute")
        idx = data.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise InvalidGraphError(f"node {key!r} is missing an integer 'index' attribute")
        if idx in seen:
            raise InvalidGraphError(f"nodes {seen[idx]!r} and {key!r} share index {idx}")
        seen[idx] = key
        if data.get("node_id", key) != key:
            raise InvalidGraphError(f"node key {key!r} != node_id {data.get('node_id')!r}")


def _near_duplicate_neighbours(graph: nx.Graph) -> dict[str, dict[str, float]]:
    """Adjacency of near-duplicate links, from edge flags + graph.graph pairs."""
    adj: dict[str, dict[str, float]] = {n: {} for n in graph.nodes}
    for a, b, d in graph.edges(data=True):
        if a != b and d.get("near_duplicate"):
            adj[a][b] = adj[b][a] = float(d.get("weight", 1.0))
    for pair in graph.graph.get("near_duplicate_pairs", ()):
        a, b = pair[0], pair[1]
        sim = float(pair[2]) if len(pair) > 2 else 1.0
        if a not in adj or b not in adj:
            raise InvalidGraphError(f"near_duplicate_pairs references unknown node in {pair!r}")
        if a != b:
            adj[a].setdefault(b, sim)
            adj[b].setdefault(a, sim)
    return adj


def _find_duplicates(
    graph: nx.Graph,
    assess: dict[str, ProtectionAssessment],
    index: dict[str, int],
) -> dict[str, tuple[str, float]]:
    """Return {removed_node: (representative, similarity)}."""
    adj = _near_duplicate_neighbours(graph)
    order = sorted(graph.nodes, key=lambda n: (-assess[n].priority, index[n]))
    reps: set[str] = set()
    dup_of: dict[str, tuple[str, float]] = {}
    for n in order:
        candidates = [(sim, m) for m, sim in adj[n].items() if m in reps]
        if candidates:
            sim, rep = max(candidates, key=lambda t: (t[0], -index[t[1]]))
            dup_of[n] = (rep, sim)
        else:
            reps.add(n)
    return dup_of


def _redundancy(graph: nx.Graph, node: str, others: set[str]) -> float:
    """Highest edge similarity from ``node`` to another surviving node (0 if none)."""
    return max(
        (float(d.get("weight", 0.0)) for m, d in graph[node].items() if m in others and m != node),
        default=0.0,
    )


def _separator(prev: dict[str, Any], cur: dict[str, Any]) -> str:
    if prev.get("line_start") is not None and prev.get("line_start") == cur.get("line_start"):
        return " "
    if prev.get("kind") == "list_item" and cur.get("kind") == "list_item":
        return "\n"
    return "\n\n"


def _make(
    graph: nx.Graph,
    a: ProtectionAssessment,
    decision: Decision,
    redundancy: float,
    codes: tuple[str, ...],
    reason: str,
    dup: Optional[tuple[str, float]] = None,
) -> NodeDecision:
    data = graph.nodes[a.node_id]
    return NodeDecision(
        node_id=a.node_id,
        index=data["index"],
        decision=decision,
        unit_type=a.unit_type,
        kind=data.get("kind"),
        text=data["text"],
        importance=a.importance,
        priority=a.priority,
        protected=a.protected,
        redundancy=redundancy,
        reason_codes=codes,
        reason=reason,
        duplicate_of=dup[0] if dup else None,
        duplicate_similarity=dup[1] if dup else None,
    )


def _duplicate_decision(
    graph: nx.Graph,
    a: ProtectionAssessment,
    dup: tuple[str, float],
    assess: dict[str, ProtectionAssessment],
    index: dict[str, int],
) -> NodeDecision:
    rep, sim = dup
    reason = (
        f"Removed because it is a near-duplicate of {rep} (similarity {sim:.3f}); "
        f"{rep} was kept as the representative "
        f"(priority {assess[rep].priority:.3f} vs {a.priority:.3f})."
    )
    return _make(graph, a, Decision.REMOVE, sim, ("NEAR_DUPLICATE_REMOVED",), reason, dup)


def _budget_removal(
    graph: nx.Graph,
    a: ProtectionAssessment,
    adjusted: float,
    redundancy: float,
    target: int,
    total: int,
    excess: int,
) -> NodeDecision:
    reason = (
        f"Removed because its adjusted priority {adjusted:.3f} "
        f"(importance {a.importance:.3f}, redundancy {redundancy:.3f}) was among the lowest "
        f"of the removable units and the retention target ({target} of {total} units) "
        f"required removing {excess}."
    )
    return _make(graph, a, Decision.REMOVE, redundancy, ("BELOW_RETENTION_TARGET",), reason)


def _retained(
    graph: nx.Graph,
    a: ProtectionAssessment,
    adjusted: float,
    redundancy: float,
    rank: int,
    n_survivors: int,
    target: int,
    protected_lock: bool,
    removed_dups: list[str],
) -> NodeDecision:
    codes: list[str] = []
    parts: list[str] = []
    if protected_lock:
        codes.append("PROTECTED_TYPE")
        parts.append(f"it is a protected {a.unit_type} unit (locked from retention-target removal)")
    if removed_dups:
        codes.append("NEAR_DUPLICATE_REPRESENTATIVE")
        parts.append(
            f"it is the representative of a near-duplicate group (removed: {', '.join(removed_dups)})"
        )
    if not codes:
        codes.append("WITHIN_RETENTION_TARGET")
        parts.append(
            f"its adjusted priority {adjusted:.3f} ranked {rank} of {n_survivors} surviving "
            f"units, within the retention target of {target}"
        )
    reason = "Retained because " + "; ".join(parts) + "."
    return _make(graph, a, Decision.RETAIN, redundancy, tuple(codes), reason)


# --------------------------------------------------------------------------
# Manual check:  python -m src.compression.compressor
# (hand-made graph; no models or Day 1 pipeline needed)
# --------------------------------------------------------------------------
if __name__ == "__main__":
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
        g.add_node(nid, node_id=nid, text=text, type=utype, importance=imp,
                   index=idx, kind="sentence", line_start=line)
    g.add_edge("N02", "N03", weight=0.95, near_duplicate=True)
    g.add_edge("N02", "N04", weight=0.40, near_duplicate=False)
    g.add_edge("N03", "N04", weight=0.38, near_duplicate=False)
    g.add_edge("N04", "N07", weight=0.50, near_duplicate=False)
    g.add_edge("N01", "N02", weight=0.35, near_duplicate=False)

    r = compress_graph(g, CompressionConfig(retention_ratio=0.60))
    print(f"Requested retention: {r.requested_retention_ratio:.2f} | target nodes: "
          f"{r.target_node_count} | actual: {r.actual_retention_ratio:.3f} "
          f"({r.retained_node_count}/{r.original_node_count})")
    print(f"Retained: {list(r.retained_node_ids)}")
    print(f"Removed:  {list(r.removed_node_ids)}")
    print()
    for d in r.decisions:
        print(f"{d.node_id} {d.decision.value:<6} {d.reason}")
    print("\n[Compressed prompt]")
    print(r.compressed_prompt)
    print("\n[Order check: ids given as N05, N01, N02]")
    print(repr(reconstruct_prompt(g, ["N05", "N01", "N02"])))
    print(f"\nWarnings: {list(r.warnings)}")