# Project Roadmap

Interactive Knowledge Graph Interface for Prompt Compression — M.Sc. CS
academic prototype, built over three implementation phases (Day 1–3).
This file records what was planned and what was actually delivered at each
phase; all results below are real, from local test/demo runs, not projected.

---

## Day 1 — Prompt Intelligence Engine (COMPLETE)

Goal: turn a raw prompt into a semantic knowledge graph.

Pipeline:Raw Prompt -> Cleaning -> Segmentation -> Information Units -> Classification
-> Embeddings -> Semantic Graph -> Importance -> Graph Metrics

Delivered:
- `src/preprocessing/cleaner.py`, `segmenter.py`
- `src/extraction/unit_classifier.py` (8 unit types: role, instruction,
  context, constraint, requirement, example, output_format, other)
- `src/embeddings/embedder.py` (`all-MiniLM-L6-v2`, sentence-transformers)
- `src/graph/builder.py` (semantic-similarity graph; `similarity_threshold`,
  `near_duplicate_threshold` configurable), `importance.py`, `metrics.py`
- `scripts/day1_demo.py`

Config finalized during Day 1: `similarity_threshold = 0.30`,
`near_duplicate_threshold = 0.90`.

Tests: 83 passed.

---

## Day 2 — Quality-Aware Compression + Evaluation (COMPLETE)

Goal: compress the prompt using the graph, with explainable
retain/remove decisions, and measure the result.

Pipeline:

Annotated Graph + Classified Units -> Protection Rules
-> Quality-Aware Compression -> Compressed Prompt
-> Baseline Comparison -> Evaluation Metrics

Delivered:
- `src/compression/protection_rules.py` — protection level (strong/normal/
  low) and priority score per unit type
- `src/compression/compresser.py` — two-stage deterministic compressor
  (near-duplicate collapse, then retention-target trim), original prompt
  order preserved on reconstruction, every decision carries a reason
- `src/evaluation/baseline.py` — token counting via `tiktoken` (`cl100k_base`)
- `src/evaluation/metrics.py` — token reduction, compression ratio, node
  retention, semantic similarity, instruction fidelity
- `scripts/day2_demo.py`

Note: the compression module's filename (`compresser.py`) was fixed
during Day 2 and intentionally kept as-is for Day 3 and beyond.

Verified demo result (financial-analyst sample prompt, retention 0.70):

Original Nodes: 9 Retained: 7 Removed: 2
Original Tokens: 110 Compressed Tokens: 74
Token Reduction: 36 (32.7%) Compression Ratio: 1.49x
Semantic Similarity: 0.857
Instruction Fidelity: 1.00 (5/5 protected units preserved)


Tests: 149 passed (83 Day 1 + 66 Day 2).

---

## Day 3 — Interactive Interface + Final Integration (COMPLETE)

Goal: an interactive web UI over the Day 1 + Day 2 pipeline; no new
analysis logic, presentation and orchestration only.

Pipeline addition:

Semantic Graph + Compression Result -> Interactive Graph UI
-> Explanation -> Evaluation Dashboard -> Export


Delivered:
- `src/pipeline.py` — shared orchestration (`run_analysis`), used
  identically by the app, the demo script and the tests
- `src/visualization/graph_render.py` — interactive Plotly graph (zoom,
  pan, click-to-select, retained/removed/near-duplicate colour coding,
  protected-type markers, deterministic layout)
- `src/visualization/explain.py` — per-node explanation built from Day 1
  attributes + Day 2 decision/reason data (no new logic invented)
- `app.py` — Streamlit UI (sidebar controls, graph, node details, side-by-
  side + diff prompt view, evaluation dashboard, filters/search, exports)
- `scripts/day3_demo.py` — full backend pipeline, no UI required
- `tests/test_day3.py`
- `README.md`

Verified manually in the running app: click-to-select, empty-prompt
handling, retention slider (0.50 → 5/9 retained, matching CLI), unit-level
diff view, all 4 export downloads (compressed prompt, graph/compression/
evaluation JSON).

Tests: 172 passed (149 Day 1+2 + 23 Day 3).

---

## Status Summary

| Phase | Status | Tests passing |
|---|---|---|
| Day 1 | Complete | 83 |
| Day 2 | Complete | 149 (cumulative) |
| Day 3 | Complete | 172 (cumulative) |

## Future Scope (not implemented)
- Custom/extensible unit-type taxonomies
- Alternative compression strategies (e.g. summarization-aware merging)
- Multi-prompt / batch analysis and comparison view