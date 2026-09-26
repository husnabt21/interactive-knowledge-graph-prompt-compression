"""Day 1 demo: Prompt -> Cleaning -> Segmentation -> Classification
-> Embeddings -> Semantic Graph -> Importance -> Graph Metrics.

Run from the project root:
    python scripts/day1_demo.py
    python scripts/day1_demo.py --file my_prompt.txt --threshold 0.35 --top 5
    python scripts/day1_demo.py --export data/day1_graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedder import EmbeddingError, embed_units
from src.extraction.unit_classifier import ClassificationError, classify_units
from src.graph.builder import GraphBuildError, GraphConfig, build_semantic_graph, graph_to_dict
from src.graph.importance import ImportanceError, annotate_graph, compute_importance
from src.graph.metrics import GraphMetricsError, compute_graph_metrics
from src.preprocessing.cleaner import PromptCleaningError, clean_prompt
from src.preprocessing.segmenter import SegmentationError, segment_prompt

SAMPLE_PROMPT = """\
You are a senior financial analyst at a retail bank.

Task: Summarize the quarterly credit-risk report for the executive team.

Context:
The report covers Q3 loan performance across personal, auto and mortgage portfolios. Default rates rose in the auto portfolio.

Constraints:
- Do not exceed 200 words.
- Avoid technical jargon.

Requirements:
- Include the three biggest risk drivers.

Example: "Auto defaults rose 0.4 points, driven by used-car price declines."

Output format:
Respond in bullet points followed by a one-line recommendation.
"""

EXPECTED_ERRORS = (
    OSError, PromptCleaningError, SegmentationError, ClassificationError,
    EmbeddingError, GraphBuildError, ImportanceError, GraphMetricsError,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 1 pipeline demo.")
    p.add_argument("--file", help="Path to a UTF-8 text file containing the prompt.")
    p.add_argument("--threshold", type=float, default=GraphConfig().similarity_threshold,
                   help="Cosine similarity threshold for creating an edge.")
    p.add_argument("--top", type=int, default=3, help="How many top nodes to list.")
    p.add_argument("--export", help="Optional path to save the annotated graph as JSON.")
    return p.parse_args(argv)


def _load_prompt(path: str | None) -> str:
    if path is None:
        return SAMPLE_PROMPT
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        raw = _load_prompt(args.file)
        cleaned = clean_prompt(raw)
        segmented = segment_prompt(cleaned)
        classified = classify_units(segmented)
        embeddings = embed_units(classified)
        graph = build_semantic_graph(
            classified, embeddings, GraphConfig(similarity_threshold=args.threshold)
        )
        importance = compute_importance(graph)
        annotate_graph(graph, importance)
        metrics = compute_graph_metrics(graph, top_k=args.top)
    except EXPECTED_ERRORS as exc:
        print(f"Pipeline error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print("=== PROMPT ANALYSIS ===\n")
    print(f"Original units: {len(classified)}")
    print(f"Embedding model: {embeddings.model_name} (dim {embeddings.dim})")
    print(f"Edge threshold: {args.threshold:.2f}\n")

    print("Nodes:")
    for r in sorted(importance.scores.values(), key=lambda r: r.node_id):
        text = r.text.replace("\n", " ")
        text = text if len(text) <= 48 else text[:45] + "..."
        print(f"{r.node_id} | {r.unit_type:<13} | importance: {r.importance:.2f} | {text}")

    print("\nEdges (semantic_similarity):")
    edges = sorted(graph.edges(data=True), key=lambda e: -e[2]["weight"])
    if edges:
        for a, b, d in edges:
            flag = "  [near-duplicate]" if d["near_duplicate"] else ""
            print(f"{a} -- {b} | weight {d['weight']:.2f}{flag}")
    else:
        print("(none)")

    print("\nGraph:")
    print(metrics.summary())

    print("\nTop Important Nodes:")
    for rank, r in enumerate(importance.top(args.top), start=1):
        print(f"{rank}. {r.node_id} ({r.unit_type}) importance {r.importance:.2f}")

    print("\nMost important node in detail:")
    print(importance.ranked()[0].explain())

    warnings = [*cleaned.warnings, *segmented.warnings, *graph.graph["warnings"]]
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = graph_to_dict(graph)
        payload["metrics"] = metrics.to_dict()
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nExported graph to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())