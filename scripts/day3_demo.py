"""Day 3 demo: the complete backend pipeline, without Streamlit.

Prompt -> cleaning -> segmentation -> classification -> embeddings -> graph
-> importance -> compression -> evaluation -> graph export.

Run from the project root:
    python scripts/day3_demo.py
    python scripts/day3_demo.py --file my_prompt.txt --retention 0.5 --top 5
    python scripts/day3_demo.py --export data/day3_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.day1_demo import SAMPLE_PROMPT
from src.graph.builder import graph_to_dict
from src.pipeline import PIPELINE_ERRORS, AnalysisSettings, run_analysis

EXPECTED_ERRORS = (OSError, UnicodeDecodeError, *PIPELINE_ERRORS)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = AnalysisSettings()
    p = argparse.ArgumentParser(description="Day 3 full-pipeline demo (no UI).")
    p.add_argument("--file", help="Path to a UTF-8 text file containing the prompt.")
    p.add_argument("--threshold", type=float, default=defaults.similarity_threshold,
                   help="Cosine similarity threshold for creating a graph edge.")
    p.add_argument("--retention", type=float, default=defaults.retention_ratio,
                   help="Target fraction of information units to retain.")
    p.add_argument("--top", type=int, default=3, help="How many top important nodes to list.")
    p.add_argument("--export", help="Optional path to save graph + compression + evaluation as JSON.")
    args = p.parse_args(argv)
    if args.top < 1:
        p.error("--top must be at least 1")
    return args


def _load_prompt(path: str | None) -> str:
    if path is None:
        return SAMPLE_PROMPT
    return Path(path).read_text(encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):          # numpy arrays and scalars
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = AnalysisSettings(
        similarity_threshold=args.threshold, retention_ratio=args.retention
    )

    try:
        analysis = run_analysis(_load_prompt(args.file), settings)
    except EXPECTED_ERRORS as exc:
        print(f"Pipeline error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    gm, comp, ev = analysis.graph_metrics, analysis.compression, analysis.evaluation
    ratio = "n/a" if ev.compression_ratio is None else f"{ev.compression_ratio:.2f}x"
    sim = "n/a" if ev.semantic_similarity is None else f"{ev.semantic_similarity:.3f}"

    print("=== INTERACTIVE KNOWLEDGE GRAPH PROMPT COMPRESSION ===\n")
    print(f"Embedding model: {analysis.embeddings.model_name} (dim {analysis.embeddings.dim})")
    print(f"Edge threshold: {args.threshold:.2f} | Requested retention ratio: {args.retention:.2f}\n")

    print(f"Original Nodes: {ev.original_nodes}")
    print(f"Edges: {gm.num_edges}")
    print(f"Connected Components: {gm.num_components}\n")

    print(f"Retained Nodes: {ev.retained_nodes}")
    print(f"Removed Nodes: {ev.removed_nodes}")
    print(f"Actual Retention Ratio: {comp.actual_retention_ratio:.3f} "
          f"(target {comp.target_node_count} nodes)\n")

    print(f"Original Tokens: {ev.original_tokens}")
    print(f"Compressed Tokens: {ev.compressed_tokens}\n")

    print(f"Token Reduction: {ev.token_reduction} ({ev.token_reduction_percent:.1f}%)")
    print(f"Compression Ratio: {ratio}\n")

    print(f"Semantic Similarity: {sim}")
    if ev.instruction_fidelity is None:
        print("Instruction Fidelity: n/a (no protected units)")
    else:
        preserved = ev.protected_retained + ev.protected_covered_by_duplicate
        print(f"Instruction Fidelity: {ev.instruction_fidelity:.2f} "
              f"({preserved}/{ev.protected_total} protected units preserved)")
        if ev.protected_lost_ids:
            print(f"  Lost: {', '.join(ev.protected_lost_ids)}")

    print("\nTop Important Nodes (Day 1 importance):")
    top_k = min(args.top, comp.original_node_count)
    for rank, r in enumerate(analysis.importance.top(top_k), start=1):
        verdict = comp.decision_for(r.node_id).decision.value
        print(f"{rank}. {r.node_id} ({r.unit_type}) importance {r.importance:.2f} - {verdict}")

    print("\nRemoved Nodes + Reasons:")
    if comp.removed_units:
        for d in comp.removed_units:
            preview = " ".join(d.text.split())
            preview = preview if len(preview) <= 60 else preview[:57] + "..."
            print(f"{d.node_id} [{d.unit_type}] {preview}")
            print(f"    {d.reason}")
    else:
        print("(none)")

    if analysis.warnings:
        print("\nWarnings:")
        for w in analysis.warnings:
            print(f"- {w}")

    if args.export:
        out = Path(args.export)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = graph_to_dict(analysis.graph)
            payload["compression"] = comp.to_dict()
            payload["evaluation"] = ev.to_dict()
            out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        except OSError as exc:
            print(f"Export error ({type(exc).__name__}): {exc}", file=sys.stderr)
            return 1
        print(f"\nExported graph, compression and evaluation to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())