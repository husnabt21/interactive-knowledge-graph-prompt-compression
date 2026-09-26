"""Day 2 demo: Day 1 pipeline -> protection rules -> compression -> evaluation.

Run from the project root:
    python scripts/day2_demo.py
    python scripts/day2_demo.py --file my_prompt.txt --retention 0.5 --top 5
    python scripts/day2_demo.py --export data/day2_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compression.compresser import (
    CompressionConfig,
    CompressionError,
    compress_graph,
)
from src.compression.protection_rules import ProtectionRuleError, assess_graph
from src.embeddings.embedder import EmbeddingError, embed_units
from src.evaluation.baseline import BaselineError
from src.evaluation.metrics import EvaluationError, evaluate_compression
from src.extraction.unit_classifier import ClassificationError, classify_units
from src.graph.builder import GraphBuildError, GraphConfig, build_semantic_graph
from src.graph.importance import ImportanceError, annotate_graph, compute_importance
from src.preprocessing.cleaner import PromptCleaningError, clean_prompt
from src.preprocessing.segmenter import SegmentationError, segment_prompt

# Same sample prompt as Day 1, so Day 1 and Day 2 demos are directly comparable.
from scripts.day1_demo import SAMPLE_PROMPT

EXPECTED_ERRORS = (
    OSError, PromptCleaningError, SegmentationError, ClassificationError,
    EmbeddingError, GraphBuildError, ImportanceError, ProtectionRuleError,
    CompressionError, BaselineError, EvaluationError,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 2 compression + evaluation demo.")
    p.add_argument("--file", help="Path to a UTF-8 text file containing the prompt.")
    p.add_argument("--threshold", type=float, default=GraphConfig().similarity_threshold,
                   help="Cosine similarity threshold for creating a graph edge (Day 1).")
    p.add_argument("--retention", type=float, default=CompressionConfig().retention_ratio,
                   help="Target fraction of information units to retain (Day 2).")
    p.add_argument("--top", type=int, default=3, help="How many top-priority nodes to list.")
    p.add_argument("--export", help="Optional path to save the full result as JSON.")
    return p.parse_args(argv)


def _load_prompt(path: str | None) -> str:
    if path is None:
        return SAMPLE_PROMPT
    return Path(path).read_text(encoding="utf-8")


def _to_dict(evaluation, compression) -> dict:
    return {
        "evaluation": evaluation.to_dict(),
        "compression": compression.to_dict(),
    }


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

        protection = assess_graph(graph)
        compression = compress_graph(
            graph,
            CompressionConfig(retention_ratio=args.retention),
            original_prompt=cleaned.text,
        )
        evaluation = evaluate_compression(compression)
    except EXPECTED_ERRORS as exc:
        print(f"Pipeline error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print("=== DAY 2: COMPRESSION & EVALUATION ===\n")
    print(f"Original units: {len(classified)}")
    print(f"Embedding model: {embeddings.model_name} (dim {embeddings.dim})")
    print(f"Edge threshold: {args.threshold:.2f}")
    print(f"Requested retention ratio: {args.retention:.2f}\n")

    print("Original Prompt:")
    print(compression.original_prompt)
    print("\nCompressed Prompt:")
    print(compression.compressed_prompt)

    print(f"\nOriginal Nodes:  {compression.original_node_count}")
    print(f"Retained Nodes:  {compression.retained_node_count}")
    print(f"Removed Nodes:   {compression.removed_node_count}")
    print(f"Actual retention ratio: {compression.actual_retention_ratio:.3f}")

    print(f"\nOriginal Tokens:   {evaluation.original_tokens}")
    print(f"Compressed Tokens: {evaluation.compressed_tokens}")
    print(f"Token Reduction:   {evaluation.token_reduction} "
          f"({evaluation.token_reduction_percent:.1f}%)")
    ratio = "n/a" if evaluation.compression_ratio is None else f"{evaluation.compression_ratio:.2f}x"
    print(f"Compression Ratio: {ratio}")

    sim = "n/a" if evaluation.semantic_similarity is None else f"{evaluation.semantic_similarity:.3f}"
    print(f"\nSemantic Similarity: {sim}")
    if evaluation.instruction_fidelity is None:
        print("Instruction Fidelity: n/a (no protected units)")
    else:
        preserved = evaluation.protected_retained + evaluation.protected_covered_by_duplicate
        print(f"Instruction Fidelity: {evaluation.instruction_fidelity:.2f} "
              f"({preserved}/{evaluation.protected_total} protected units preserved)")
        if evaluation.protected_lost_ids:
            print(f"  Lost: {', '.join(evaluation.protected_lost_ids)}")

    print(f"\nTop {min(args.top, len(protection))} Priority Nodes:")
    top_assessed = sorted(protection.values(), key=lambda a: -a.priority)[: args.top]
    for rank, a in enumerate(top_assessed, start=1):
        print(f"{rank}. {a.node_id} ({a.unit_type}) priority {a.priority:.2f} "
              f"[{a.level.value}]")

    print("\nRetention Decisions:")
    for d in compression.decisions:
        print(f"{d.node_id} {d.decision.value:<6} {d.reason}")

    warnings = [
        *cleaned.warnings, *segmented.warnings, *graph.graph["warnings"],
        *compression.warnings, *evaluation.warnings,
    ]
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_to_dict(evaluation, compression), indent=2), encoding="utf-8")
        print(f"\nExported result to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())