"""Semantic graph construction for prompt information units.

Input : classified units + EmbeddingResult (same unit ids, same order)
Output: networkx.Graph (undirected)

Nodes : one per information unit; key = unit_id
        attrs: node_id, text, type, embedding, confidence, kind, index,
               line_start, parent_heading
Edges : between units whose cosine similarity >= similarity_threshold
        attrs: source, target, weight, relationship, near_duplicate
        relationship is always 'semantic_similarity' -- this graph does NOT
        claim causal, hierarchical or factual relations.

Graph-level info is stored in graph.graph (thresholds, model_name,
near_duplicate_pairs, warnings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np

from src.embeddings.embedder import EmbeddingResult, similarity_matrix
from src.extraction.unit_classifier import ClassifiedUnit

RELATIONSHIP = "semantic_similarity"


class GraphBuildError(ValueError):
    """Raised when a graph cannot be built from the given inputs."""


@dataclass(frozen=True)
class GraphConfig:
    similarity_threshold: float = 0.30       # edge if cosine sim >= this
    near_duplicate_threshold: float = 0.90   # flag pair if cosine sim >= this

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise GraphBuildError("similarity_threshold must be in (0, 1].")
        if not self.similarity_threshold <= self.near_duplicate_threshold <= 1.0:
            raise GraphBuildError(
                "near_duplicate_threshold must be >= similarity_threshold and <= 1."
            )


def build_semantic_graph(
    classified: Sequence[ClassifiedUnit],
    embeddings: EmbeddingResult,
    config: GraphConfig | None = None,
) -> nx.Graph:
    """Build the semantic graph.

    Raises:
        TypeError: wrong input types.
        GraphBuildError: no units, or unit ids/order differ from the embeddings.
    """
    cfg = config or GraphConfig()
    items = tuple(classified)

    if not items:
        raise GraphBuildError("Cannot build a graph from zero information units.")
    for c in items:
        if not isinstance(c, ClassifiedUnit):
            raise TypeError(f"Expected ClassifiedUnit, got {type(c).__name__}.")
    if not isinstance(embeddings, EmbeddingResult):
        raise TypeError(f"Expected EmbeddingResult, got {type(embeddings).__name__}.")

    ids = tuple(c.unit_id for c in items)
    if ids != embeddings.unit_ids:
        raise GraphBuildError(
            "Unit ids/order of the classified units and the embeddings differ; "
            "embed the same units, in the same order, that you pass here."
        )

    warnings: list[str] = []
    dup_pairs: list[tuple[str, str, float]] = []

    graph = nx.Graph(
        model_name=embeddings.model_name,
        relationship_type=RELATIONSHIP,
        similarity_threshold=cfg.similarity_threshold,
        near_duplicate_threshold=cfg.near_duplicate_threshold,
        near_duplicate_pairs=dup_pairs,
        warnings=warnings,
    )

    # ---- nodes ----
    for i, c in enumerate(items):
        graph.add_node(
            c.unit_id,
            node_id=c.unit_id,
            text=c.text,
            type=c.unit_type,
            embedding=embeddings.vectors[i],
            confidence=c.confidence,
            kind=c.unit.kind,
            index=c.unit.index,
            line_start=c.unit.line_start,
            parent_heading=c.unit.parent_heading,
        )

    # ---- edges (upper triangle only: no self-loops, no duplicates) ----
    n = len(items)
    if n >= 2:
        sim = similarity_matrix(embeddings.vectors)
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j])
                if s < cfg.similarity_threshold:
                    continue
                is_dup = s >= cfg.near_duplicate_threshold
                graph.add_edge(
                    ids[i], ids[j],
                    source=ids[i], target=ids[j],
                    weight=s,
                    relationship=RELATIONSHIP,
                    near_duplicate=is_dup,
                )
                if is_dup:
                    dup_pairs.append((ids[i], ids[j], s))
                    warnings.append(
                        f"{ids[i]} and {ids[j]} are near-duplicates (similarity {s:.2f})."
                    )

    if n == 1:
        warnings.append("Only one information unit; the graph has no edges.")
    elif graph.number_of_edges() == 0:
        warnings.append(
            f"No semantic edges at threshold {cfg.similarity_threshold:.2f}; "
            "consider lowering the threshold."
        )
    return graph


def graph_to_dict(graph: nx.Graph, include_embeddings: bool = False) -> dict[str, Any]:
    """JSON-safe export of the graph (for saving, or the Day 3 UI)."""
    meta = {
        "model_name": graph.graph.get("model_name"),
        "relationship_type": graph.graph.get("relationship_type"),
        "similarity_threshold": graph.graph.get("similarity_threshold"),
        "near_duplicate_threshold": graph.graph.get("near_duplicate_threshold"),
        "near_duplicate_pairs": [list(p) for p in graph.graph.get("near_duplicate_pairs", [])],
        "warnings": list(graph.graph.get("warnings", [])),
    }
    nodes = []
    for _, data in graph.nodes(data=True):
        row = {k: v for k, v in data.items() if k != "embedding"}
        if include_embeddings and "embedding" in data:
            row["embedding"] = np.asarray(data["embedding"]).tolist()
        nodes.append(row)
    nodes.sort(key=lambda r: r.get("index", 0))
    edges = [dict(d) for _, _, d in graph.edges(data=True)]
    return {"graph": meta, "nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# Manual check:  python -m src.graph.builder
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from src.embeddings.embedder import embed_units
    from src.extraction.unit_classifier import classify_units
    from src.preprocessing.cleaner import clean_prompt
    from src.preprocessing.segmenter import InformationUnit, segment_prompt

    # 1) Structural check with hand-made vectors (NOT part of the pipeline).
    def _demo(uid: str, idx: int, text: str, utype: str) -> ClassifiedUnit:
        u = InformationUnit(uid, idx, text, "sentence", idx + 1)
        return ClassifiedUnit(u, utype, 1.0, ("demo",), {utype: 1.0})  # type: ignore[arg-type]

    units = (
        _demo("N01", 0, "Plan a trip.", "instruction"),
        _demo("N02", 1, "Plan a trip.", "instruction"),
        _demo("N03", 2, "Use JSON.", "output_format"),
    )
    vecs = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    emb = EmbeddingResult(("N01", "N02", "N03"), tuple(u.text for u in units), vecs, "hand-made")
    g = build_semantic_graph(units, emb)
    print("[Structural check with hand-made vectors]")
    print(f"Nodes: {g.number_of_nodes()}")
    print(f"Edges: {g.number_of_edges()}")
    for a, b, d in g.edges(data=True):
        print(f"{a} -- {b} | weight {d['weight']:.3f} | {d['relationship']} | near_duplicate={d['near_duplicate']}")
    print(f"Warnings: {g.graph['warnings']}")

    # 2) Real pipeline on a sample prompt.
    sample = (
        "You are an expert travel planner. Your goal is to help users plan trips.\n\n"
        "Task: Create a 3-day itinerary for Tokyo. Include e.g. museums and parks.\n\n"
        "Constraints:\n"
        "- Budget must not exceed $500.\n"
        "- Avoid crowded places.\n\n"
        "Output format:\n"
        "```json\n"
        '{"days": []}\n'
        "```\n"
    )
    classified = classify_units(segment_prompt(clean_prompt(sample)))
    cfg = GraphConfig()
    graph = build_semantic_graph(classified, embed_units(classified), cfg)
    print("\n[Real pipeline on sample prompt]")
    print(f"Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}  "
          f"(threshold {cfg.similarity_threshold})")
    for node, d in graph.nodes(data=True):
        print(f"{node} | {d['type']:<13} | {d['text'][:45]!r}")
    for a, b, d in graph.edges(data=True):
        print(f"{a} -- {b} | weight {d['weight']:.2f}")
    print(f"Warnings: {graph.graph['warnings']}")