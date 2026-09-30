# Interactive Knowledge Graph Interface for Prompt Compression

## Problem
Long, unstructured prompts waste tokens and make it hard to see which parts
actually matter to a language model. Manually trimming a prompt risks
silently dropping an instruction, constraint or output-format requirement.

## Objective
An academic prototype that represents a prompt as a semantic knowledge
graph, uses that graph to make explainable, quality-aware compression
decisions, and presents the whole process — graph, decisions, reasons,
metrics — through an interactive web interface.

## Architecture

```
Raw Prompt
    -> Cleaning -> Segmentation -> Classification -> Embeddings
    -> Semantic Graph -> Importance -> Graph Metrics        (Day 1)
    -> Protection Rules -> Quality-Aware Compression
    -> Baseline -> Evaluation Metrics                       (Day 2)
    -> Interactive Graph UI -> Explanation -> Export         (Day 3)
```
src/
├── preprocessing/ cleaner.py, segmenter.py
├── extraction/ unit_classifier.py
├── embeddings/ embedder.py (all-MiniLM-L6-v2, sentence-transformers)
├── graph/ builder.py, importance.py, metrics.py
├── compression/ protection_rules.py, compresser.py
├── evaluation/ baseline.py, metrics.py
├── visualization/ graph_render.py, explain.py
└── pipeline.py shared orchestration for the UI, demo and tests
app.py Streamlit UI
scripts/ day1_demo.py, day2_demo.py, day3_demo.py
tests/ test_day1.py, test_day2.py, test_day3.py

## Day 1 — Prompt Intelligence
Cleans and segments a prompt into information units, classifies each into
one of `role, instruction, context, constraint, requirement, example,
output_format, other`, embeds units with `all-MiniLM-L6-v2`, and builds a
semantic similarity graph: an edge means two units have similar
embeddings, not a factual or causal relationship. Importance combines type
weight, graph connectivity and centrality.

## Day 2 — Quality-Aware Compression + Evaluation
`protection_rules.py` assigns each unit type a protection level and a
priority score. `compresser.py` removes redundant near-duplicate units
(keeping the higher-priority one as representative) and then trims to a
target node count set by the retention ratio, locking protected types from
removal. The compressed prompt is rebuilt in original order, never by
importance. Every retain/remove decision carries a concrete, traceable
reason. `evaluation/` reports token reduction (via `tiktoken`, `cl100k_base`),
compression ratio, node retention, whole-prompt semantic similarity, and
instruction fidelity (how many protected units survived).

## Day 3 — Interactive Interface
A Streamlit app (`app.py`) that runs the full pipeline on a user-entered
prompt and shows:
- An interactive Plotly graph (zoom, pan, click-to-select). Green = retained,
  red = removed, orange = removed as a near-duplicate; diamonds mark
  protected unit types.
- A details panel with Day 1 attributes (importance, connectivity,
  centrality, degree) and the Day 2 decision, reason and reason codes for
  any selected node.
- Original vs. compressed prompt (side-by-side and as a unit-level diff).
- An evaluation dashboard (token, graph and quality metrics).
- Highlight/search filters (UI-only; never change the underlying result).
- Downloads for the compressed prompt, graph JSON, compression JSON and
  evaluation JSON.
- An optional "Load sample prompt" button (the financial-analyst prompt
  used for the demo numbers below); the main text area accepts any prompt.

## Compression Approach
Deterministic and explainable — no ML training, no LLM calls. Two stages:
(1) collapse near-duplicate units to one representative, chosen by priority;
(2) trim the remainder to a target node count (`ceil(retention_ratio × N)`),
protecting strongly-typed units from this stage. The retention ratio is a
*target*, not a guarantee — locked units can raise the actual ratio, and
duplicate removal can lower it.

## Evaluation Metrics
- Token reduction / compression ratio — via `tiktoken` (`cl100k_base`),
  with a labelled approximate fallback if unavailable.
- Node retention % — retained / original nodes.
- Semantic similarity — cosine similarity between whole-prompt
  embeddings of the original and compressed prompt. A similarity score, not
  a guarantee of equivalent model output.
- Instruction fidelity — fraction of protected units (instruction,
  constraint, requirement, output_format) retained or covered by a retained
  near-duplicate representative.

## Installation
```powershell
pip install -r requirements.txt
```
First run downloads `all-MiniLM-L6-v2` and the `cl100k_base` tokenizer file
(internet required once; both are cached afterward).

## Running

```powershell
# Interactive app
streamlit run app.py

# Backend-only demo (no UI)
python scripts/day3_demo.py
python scripts/day3_demo.py --file my_prompt.txt --retention 0.5 --top 5
python scripts/day3_demo.py --export data/day3_result.json

# Tests
pytest -q
```

## Example Workflow
Demo prompt (financial-analyst sample, `scripts/day1_demo.py`), default
settings (similarity threshold 0.30, retention ratio 0.70):

Original Nodes: 9 Retained Nodes: 7 Removed Nodes: 2
Original Tokens: 110 Compressed Tokens: 74
Token Reduction: 36 (32.7%) Compression Ratio: 1.49x
Semantic Similarity: 0.857
Instruction Fidelity: 1.00 (5/5 protected units preserved)

This is a single example result, not a universal benchmark — actual numbers
depend on the prompt and the chosen settings.

## Limitations
- Semantic similarity and instruction fidelity are heuristic signals; they
  do not guarantee an LLM would respond equivalently to the compressed
  prompt.
- The embedding model reads only ~256 word pieces, so similarity mainly
  reflects the start of long prompts.
- The retention ratio is a target, subject to protection locks and
  duplicate removal.
- Reconstructed list items use a fixed `"- "` marker; numbered-list
  formatting is not preserved exactly.
- This is a semantic-similarity graph, not a factual/world-knowledge graph.

## Future Scope
- Support for additional/custom unit-type taxonomies.
- Alternative compression strategies (e.g. summarization-aware merging)
  alongside the current deterministic rule-based approach.
- Multi-prompt / batch analysis and comparison view.