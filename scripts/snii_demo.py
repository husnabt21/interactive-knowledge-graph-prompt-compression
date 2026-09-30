"""SNII demo: ranks nodes before compression, then shows which survive it.

SNII values are fixed by the graph alone -- compression only decides which
nodes are kept, it never changes a node's SNII. This script makes that
visible directly.

Run from the project root:
    python scripts/snii_demo.py
    python scripts/snii_demo.py --retention 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.day1_demo import SAMPLE_PROMPT
from src.graph.snii import compute_snii, explain_snii
from src.pipeline import PIPELINE_ERRORS, AnalysisSettings, run_analysis


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SNII before/after-compression demo.")
    p.add_argument("--retention", type=float, default=0.70)
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args(argv)

    try:
        analysis = run_analysis(SAMPLE_PROMPT, AnalysisSettings(retention_ratio=args.retention))
    except PIPELINE_ERRORS as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1

    snii = compute_snii(analysis.importance)
    retained = set(analysis.compression.retained_node_ids)

    print("=== SNII: FULL RANKING (before compression) ===")
    for s in snii.ranked():
        mark = "KEPT" if s.node_id in retained else "removed"
        print(f"{s.rank}. {s.node_id} ({s.unit_type}) SNII={s.snii:.4f} -> {mark}")

    print(f"\n=== Top {args.top} by SNII (explained) ===")
    for s in snii.top(args.top):
        print(explain_snii(s))
        print()

    print("=== Nodes that dropped out of compression despite the ranking above ===")
    removed_ranked = [s for s in snii.ranked() if s.node_id not in retained]
    if removed_ranked:
        for s in removed_ranked:
            print(f"{s.node_id} (SNII rank {s.rank}) -> removed")
    else:
        print("(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())