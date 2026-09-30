"""Interactive Knowledge Graph Interface for Prompt Compression (Day 3).

Run from the project root:
    streamlit run app.py

Presentation + orchestration only. All analysis comes from the existing Day 1
and Day 2 modules through src.pipeline.run_analysis(); nothing is recalculated
here. The graph is a semantic-similarity graph: an edge means two units have
similar embeddings, not that a factual relationship exists between them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import streamlit as st

from scripts.day1_demo import SAMPLE_PROMPT
from src.compression.compresser import Decision
from src.embeddings.embedder import get_model
from src.graph.builder import graph_to_dict
from src.pipeline import PIPELINE_ERRORS, AnalysisResult, AnalysisSettings, run_analysis
from src.visualization.explain import (
    ExplanationError,
    NodeExplanation,
    explain_all,
    search_nodes,
)
from src.visualization.graph_render import (
    FILTER_OPTIONS,
    PLOTLY_CONFIG,
    GraphRenderError,
    build_figure,
    compute_layout,
    filter_node_ids,
    selected_node_ids,
)

LIMITATIONS = (
    "Academic prototype. Semantic similarity is an embedding-based score and instruction "
    "fidelity measures retention of protected units; neither guarantees that a language "
    "model would respond equivalently to the compressed prompt. Results shown are for the "
    "prompt you entered only."
)

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#$~|])")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _md_escape(text: Any) -> str:
    """Escape Markdown/LaTeX specials (e.g. '$') and flatten whitespace."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def _na(value: Optional[float], fmt: str = "{:.3f}") -> str:
    return "N/A" if value is None else fmt.format(value)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):          # numpy arrays and scalars
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=_json_default)


def _md_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    lines.extend(f"| {_md_escape(k)} | {_md_escape(v)} |" for k, v in rows)
    return "\n".join(lines)


def _diff_view(compression: Any) -> str:
    """Original prompt unit by unit; removed units are prefixed with '-'."""
    lines: list[str] = []
    for d in compression.decisions:
        prefix = "-" if d.decision is Decision.REMOVE else " "
        for i, line in enumerate(d.text.splitlines() or [""]):
            tag = f"{d.node_id}  " if i == 0 else "     "
            lines.append(f"{prefix} {tag}{line}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cached backend calls
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (the first run may download it)...")
def _embedding_model() -> Any:
    return get_model()


@st.cache_resource(show_spinner="Analyzing prompt...", max_entries=8)
def _analyze_cached(
    prompt: str,
    threshold: float,
    retention: float,
    protect_strong: bool,
    remove_duplicates: bool,
):
    settings = AnalysisSettings(threshold, retention, protect_strong, remove_duplicates)
    analysis = run_analysis(prompt, settings, model=_embedding_model())
    return analysis, compute_layout(analysis.graph)   # layout is deterministic (seeded)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("prompt_text", "")
    st.session_state.setdefault("analyzed_prompt", None)
    st.session_state.setdefault("analysis_id", 0)
    st.session_state.setdefault("selected_node", None)
    st.session_state.setdefault("_last_click", None)


def _sidebar() -> tuple[float, float, bool, bool]:
    st.sidebar.header("Controls")
    threshold = st.sidebar.slider(
        "Similarity threshold", 0.05, 0.90, 0.30, 0.05,
        help="Minimum cosine similarity between two units to draw an edge. Lower = more edges.",
    )
    retention = st.sidebar.slider(
        "Retention ratio", 0.10, 1.00, 0.70, 0.05,
        help="Target fraction of units to keep. It is a target, not a guarantee: protected "
             "units can raise the actual ratio and near-duplicate removal can lower it.",
    )
    protect = st.sidebar.checkbox(
        "Protect instructions, constraints, requirements and output format", value=True
    )
    dedupe = st.sidebar.checkbox("Remove near duplicates", value=True)
    st.sidebar.caption(
        "Near-duplicate threshold is fixed at 0.90 (Day 1 default). "
        "Changing a control re-runs the analysis on the last analyzed prompt."
    )
    return round(threshold, 2), round(retention, 2), protect, dedupe


def _load_sample() -> None:
    st.session_state["prompt_text"] = SAMPLE_PROMPT


def _prompt_input() -> None:
    st.subheader("Prompt")
    st.text_area(
        "Prompt text", key="prompt_text", height=220,
        placeholder="Enter your prompt...", label_visibility="collapsed",
    )
    col_analyze, col_sample, _ = st.columns([1, 1, 3])
    analyze = col_analyze.button("Analyze prompt", type="primary")
    col_sample.button("Load sample prompt", on_click=_load_sample)

    if analyze:
        text = st.session_state["prompt_text"]
        if not text.strip():
            st.warning("Please enter a prompt before analyzing.")
        else:
            st.session_state["analyzed_prompt"] = text
            st.session_state["analysis_id"] += 1
            st.session_state["selected_node"] = None
            st.session_state["_last_click"] = None


# --------------------------------------------------------------------------
# Result sections
# --------------------------------------------------------------------------
def _render_overview(a: AnalysisResult) -> None:
    gm, ev = a.graph_metrics, a.evaluation
    st.subheader("Prompt overview")
    cols = st.columns(4)
    cols[0].metric("Original nodes", gm.num_nodes)
    cols[1].metric("Edges", gm.num_edges)
    cols[2].metric("Connected components", gm.num_components)
    cols[3].metric("Original tokens", ev.original_tokens)
    st.caption("Unit types: " + ", ".join(f"{t} ({n})" for t, n in gm.type_counts.items()))
    if gm.isolated_nodes:
        st.caption(
            "Isolated nodes (no edge at this threshold): " + ", ".join(gm.isolated_nodes)
            + ". Lowering the similarity threshold may connect them."
        )


def _render_graph(
    a: AnalysisResult,
    positions: dict[str, tuple[float, float]],
    settings_key: tuple[Any, ...],
) -> set[str]:
    """Draw the graph; returns the set of node ids currently visible (not dimmed)."""
    st.subheader("Interactive semantic graph")
    st.caption(
        "Scroll to zoom, drag to pan, click a node to inspect it. Green = retained, "
        "red = removed, orange = removed as a near-duplicate. Diamonds are protected unit "
        "types; larger markers have higher importance. Edges are embedding similarity "
        "(semantic_similarity), not factual relations; hover an edge midpoint for its weight."
    )
    col_filter, col_search = st.columns([1, 2])
    mode = col_filter.selectbox("Highlight", FILTER_OPTIONS, key="node_filter")
    query = col_search.text_input(
        "Search node text", key="node_search", placeholder="Search node text..."
    )

    all_ids = [d.node_id for d in a.compression.decisions]
    visible = filter_node_ids(a.compression, mode)
    if query.strip():
        visible = visible & set(search_nodes(a.compression, query))
    visible_ids = None if len(visible) == len(all_ids) else visible
    if not visible:
        st.info("No nodes match the current filter/search.")

    selected = st.session_state.get("selected_node")
    if selected not in all_ids:
        selected = None
        st.session_state["selected_node"] = None

    fig = build_figure(
        a.graph, a.compression, positions=positions,
        selected_id=selected, visible_ids=visible_ids,
    )
    # A new key whenever the figure content changes, so a stale click selection
    # is never mapped onto a different node.
    chart_key = "graph|" + "|".join(
        str(x) for x in (st.session_state["analysis_id"], *settings_key, mode, query.strip())
    )
    event = st.plotly_chart(
        fig, width="stretch", key=chart_key, on_select="rerun",
        selection_mode="points", config=PLOTLY_CONFIG,
    )

    clicked = selected_node_ids(event, fig)
    click_id = clicked[0] if clicked else None
    if click_id != st.session_state["_last_click"]:
        st.session_state["_last_click"] = click_id
        if click_id is not None and click_id != selected:
            st.session_state["selected_node"] = click_id
            st.rerun()
    return visible


def _render_details(
    a: AnalysisResult,
    explanations: dict[str, NodeExplanation],
    visible: set[str],
) -> None:
    st.subheader("Selected node details")
    current = st.session_state.get("selected_node")
    options = [""] + [n for n in explanations if n in visible or n == current]
    index = options.index(current) if current in options else 0

    def _label(nid: str) -> str:
        if not nid:
            return "-- select a node --"
        e = explanations[nid]
        return f"{nid} | {e.unit_type} | {' '.join(e.text.split())[:60]}"

    choice = st.selectbox(
        "Inspect a node (or click one in the graph)", options, index=index, format_func=_label
    )
    picked = choice or None
    if picked != current:
        st.session_state["selected_node"] = picked
        st.rerun()

    if current is None:
        st.info("Click a node in the graph, or choose one above, to see why it was kept or removed.")
        return

    exp = explanations[current]
    callout = {"Retained": st.success, "Removed": st.error}.get(exp.status, st.warning)
    callout(f"{exp.node_id}: {exp.status}")
    st.code(exp.text, language=None, wrap_lines=True)

    left, right = st.columns(2)
    with left:
        st.markdown(_md_table(exp.details_rows()))
    with right:
        st.markdown(f"**{exp.headline}**")
        for bullet in exp.bullets:
            st.markdown(f"- {_md_escape(bullet)}")
        st.caption("Reason from the compression engine: " + _md_escape(exp.reason))


def _render_compression(a: AnalysisResult) -> None:
    c = a.compression
    st.subheader("Compression results")
    st.caption(
        f"Requested retention ratio {c.requested_retention_ratio:.2f}; actual "
        f"{c.actual_retention_ratio:.3f} ({c.retained_node_count} of {c.original_node_count} "
        f"nodes retained; target {c.target_node_count})."
    )
    tab_side, tab_diff, tab_reasons = st.tabs(
        ["Original vs compressed", "Unit-level diff", "Removal decisions"]
    )
    with tab_side:
        left, right = st.columns(2)
        left.markdown("**Original prompt**")
        left.code(c.original_prompt, language=None, wrap_lines=True)
        right.markdown("**Compressed prompt**")
        right.code(c.compressed_prompt, language=None, wrap_lines=True)
    with tab_diff:
        st.caption("Original prompt unit by unit. Lines starting with '-' were removed.")
        st.code(_diff_view(c), language="diff", wrap_lines=True)
    with tab_reasons:
        if not c.removed_units:
            st.info("No units were removed at these settings.")
        for d in c.removed_units:
            st.markdown(
                f"**{d.node_id}** ({_md_escape(d.unit_type)}): {_md_escape(d.reason)}"
            )


def _render_evaluation(a: AnalysisResult) -> None:
    ev, gm = a.evaluation, a.graph_metrics
    st.subheader("Evaluation")

    st.markdown("**Token metrics**")
    t = st.columns(5)
    t[0].metric("Original tokens", ev.original_tokens)
    t[1].metric("Compressed tokens", ev.compressed_tokens)
    t[2].metric("Token reduction", ev.token_reduction)
    t[3].metric("Token reduction %", f"{ev.token_reduction_percent:.1f}%")
    t[4].metric("Compression ratio", _na(ev.compression_ratio, "{:.2f}x"))
    st.caption(
        f"Tokenizer: {ev.tokenizer}" + ("" if ev.tokenizer_exact else " (approximate counts)")
    )

    st.markdown("**Graph metrics**")
    g1 = st.columns(4)
    g1[0].metric("Original nodes", ev.original_nodes)
    g1[1].metric("Retained nodes", ev.retained_nodes)
    g1[2].metric("Removed nodes", ev.removed_nodes)
    g1[3].metric("Edges", gm.num_edges)
    g2 = st.columns(4)
    g2[0].metric("Connected components", gm.num_components)
    g2[1].metric("Density", f"{gm.density:.3f}")
    g2[2].metric("Average degree", f"{gm.average_degree:.2f}")

    st.markdown("**Quality signals**")
    q = st.columns(3)
    q[0].metric("Semantic similarity", _na(ev.semantic_similarity))
    q[1].metric("Instruction fidelity", _na(ev.instruction_fidelity, "{:.2f}"))
    q[2].metric("Node retention", f"{ev.node_retention_percent:.1f}%")
    if ev.instruction_fidelity is None:
        st.caption("Instruction fidelity: N/A (no protected units in this prompt).")
    else:
        preserved = ev.protected_retained + ev.protected_covered_by_duplicate
        text = (
            f"Instruction fidelity: {preserved}/{ev.protected_total} protected units preserved "
            f"({ev.protected_retained} retained, {ev.protected_covered_by_duplicate} via a "
            "near-duplicate representative)."
        )
        if ev.protected_lost_ids:
            text += " Lost: " + ", ".join(ev.protected_lost_ids) + "."
        st.caption(text)
    st.caption(
        "Semantic similarity compares whole-prompt embeddings; it is not a guarantee of "
        "equivalent model output."
    )

    if a.warnings:
        with st.expander(f"Warnings ({len(a.warnings)})"):
            for w in a.warnings:
                st.markdown("- " + _md_escape(w))


def _render_exports(a: AnalysisResult) -> None:
    st.subheader("Export")
    cols = st.columns(4)
    cols[0].download_button(
        "Compressed prompt (.txt)", a.compression.compressed_prompt,
        file_name="compressed_prompt.txt", mime="text/plain",
    )
    cols[1].download_button(
        "Graph (.json)", _dumps(graph_to_dict(a.graph)),
        file_name="graph.json", mime="application/json",
    )
    cols[2].download_button(
        "Evaluation (.json)", _dumps(a.evaluation.to_dict()),
        file_name="evaluation.json", mime="application/json",
    )
    cols[3].download_button(
        "Compression (.json)", _dumps(a.compression.to_dict()),
        file_name="compression.json", mime="application/json",
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Interactive KG for Prompt Compression", layout="wide")
    _init_state()

    st.title("Interactive Knowledge Graph for Prompt Compression")
    st.caption(
        "An interactive semantic graph interface for analyzing and compressing prompts, "
        "with explainable retain/remove decisions."
    )

    threshold, retention, protect, dedupe = _sidebar()
    _prompt_input()

    prompt = st.session_state["analyzed_prompt"]
    if prompt is None:
        st.info("Enter or paste a prompt (or load the sample), then press Analyze prompt.")
        return

    try:
        analysis, positions = _analyze_cached(prompt, threshold, retention, protect, dedupe)
    except PIPELINE_ERRORS as exc:
        st.error(f"Analysis failed ({type(exc).__name__}): {exc}")
        return

    try:
        explanations = explain_all(analysis.graph, analysis.compression)
        _render_overview(analysis)
        visible = _render_graph(analysis, positions, (threshold, retention, protect, dedupe))
        _render_details(analysis, explanations, visible)
        _render_compression(analysis)
        _render_evaluation(analysis)
        _render_exports(analysis)
    except (GraphRenderError, ExplanationError) as exc:
        st.error(f"Could not display the results ({type(exc).__name__}): {exc}")
        return

    st.caption(LIMITATIONS)


if __name__ == "__main__":
    main()