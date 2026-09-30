"""Day 3 tests: pipeline orchestration, graph rendering, explanations.

Run from the project root:   pytest -q
Offline: uses a local FakeModel, no Streamlit process is started (app.py's
functions are imported and unit-tested directly where practical).
"""

from __future__ import annotations

import json
import zlib

import networkx as nx
import numpy as np
import pytest

from src.compression.compresser import CompressionConfig, Decision, compress_graph
from src.graph.builder import graph_to_dict
from src.pipeline import (
    AnalysisSettings,
    PIPELINE_ERRORS,
    run_analysis,
)
from src.preprocessing.cleaner import EmptyPromptError
from src.visualization.explain import (
    ExplanationError,
    explain_all,
    explain_node,
    search_nodes,
)
from src.visualization.graph_render import (
    FILTER_PROTECTED,
    FILTER_REMOVED,
    GraphRenderError,
    build_figure,
    compute_layout,
    filter_node_ids,
    node_status,
    selected_node_ids,
    STATUS_DUPLICATE,
    STATUS_REMOVED,
    STATUS_RETAINED,
)

SAMPLE_PROMPT = (
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


class FakeModel:
    """Deterministic stand-in for a SentenceTransformer (no download needed)."""

    DIM = 16

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, zlib.crc32(word.encode("utf-8")) % self.DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


@pytest.fixture(scope="module")
def analysis():
    return run_analysis(SAMPLE_PROMPT, AnalysisSettings(similarity_threshold=0.05), model=FakeModel())


# ==========================================================================
# 1. PIPELINE
# ==========================================================================
def test_run_analysis_end_to_end(analysis):
    assert analysis.compression.original_node_count == 7
    assert analysis.compression.retained_node_count + analysis.compression.removed_node_count == 7
    assert analysis.evaluation.original_nodes == 7
    assert set(analysis.graph.nodes) == {d.node_id for d in analysis.compression.decisions}


def test_run_analysis_respects_settings():
    settings = AnalysisSettings(similarity_threshold=0.05, retention_ratio=1.0)
    a = run_analysis(SAMPLE_PROMPT, settings, model=FakeModel())
    assert a.compression.retained_node_count == a.compression.original_node_count


def test_run_analysis_empty_prompt_raises():
    with pytest.raises(EmptyPromptError):
        run_analysis("   ", model=FakeModel())


def test_run_analysis_rejects_non_string():
    with pytest.raises(TypeError):
        run_analysis(None, model=FakeModel())  # type: ignore[arg-type]


def test_pipeline_errors_tuple_is_usable_as_except_clause():
    with pytest.raises(PIPELINE_ERRORS):
        run_analysis("", model=FakeModel())


def test_run_analysis_is_deterministic():
    a1 = run_analysis(SAMPLE_PROMPT, model=FakeModel())
    a2 = run_analysis(SAMPLE_PROMPT, model=FakeModel())
    assert a1.compression.retained_node_ids == a2.compression.retained_node_ids
    assert a1.evaluation.semantic_similarity == pytest.approx(a2.evaluation.semantic_similarity)


# ==========================================================================
# 2. GRAPH RENDERING
# ==========================================================================
def test_compute_layout_places_every_node(analysis):
    pos = compute_layout(analysis.graph)
    assert set(pos) == set(analysis.graph.nodes)


def test_compute_layout_isolated_nodes_do_not_collide():
    g = nx.Graph()
    g.add_node("A", index=0)
    g.add_node("B", index=1)   # no edges: both isolated
    pos = compute_layout(g)
    assert pos["A"] != pos["B"]


def test_build_figure_matches_node_count(analysis):
    fig = build_figure(analysis.graph, analysis.compression)
    node_traces = [t for t in fig.data if t.customdata is not None]
    total = sum(len(t.customdata) for t in node_traces)
    assert total == analysis.compression.original_node_count


def test_build_figure_rejects_mismatched_result(analysis):
    g2 = nx.Graph()
    g2.add_node("Z01", node_id="Z01", type="role", importance=0.5, text="x", index=0)
    bad_result = compress_graph(g2, CompressionConfig(retention_ratio=1.0))
    with pytest.raises(GraphRenderError):
        build_figure(analysis.graph, bad_result)


def test_build_figure_rejects_unknown_selected_id(analysis):
    with pytest.raises(GraphRenderError):
        build_figure(analysis.graph, analysis.compression, selected_id="N99")


def test_node_status_matches_decision(analysis):
    for d in analysis.compression.decisions:
        status = node_status(d)
        if d.decision is Decision.RETAIN:
            assert status == STATUS_RETAINED
        elif d.duplicate_of is not None:
            assert status == STATUS_DUPLICATE
        else:
            assert status == STATUS_REMOVED


def test_filter_node_ids_removed_and_protected(analysis):
    removed = filter_node_ids(analysis.compression, FILTER_REMOVED)
    assert removed == {d.node_id for d in analysis.compression.removed_units}
    protected = filter_node_ids(analysis.compression, FILTER_PROTECTED)
    assert protected == {d.node_id for d in analysis.compression.decisions if d.protected}


def test_filter_node_ids_rejects_unknown_mode(analysis):
    with pytest.raises(ValueError):
        filter_node_ids(analysis.compression, "not a real mode")


def test_selected_node_ids_maps_click_to_node(analysis):
    fig = build_figure(analysis.graph, analysis.compression)
    retained_curve = next(i for i, t in enumerate(fig.data) if t.name == "Retained")
    retained_ids = list(fig.data[retained_curve].customdata)
    click = {"points": [{"curve_number": retained_curve, "point_index": 0}]}
    assert selected_node_ids(click, fig) == [retained_ids[0]]


def test_selected_node_ids_handles_no_selection(analysis):
    fig = build_figure(analysis.graph, analysis.compression)
    assert selected_node_ids(None, fig) == []
    assert selected_node_ids({"points": []}, fig) == []


# ==========================================================================
# 3. EXPLANATIONS
# ==========================================================================
def test_explain_all_covers_every_node(analysis):
    explanations = explain_all(analysis.graph, analysis.compression)
    assert set(explanations) == set(analysis.graph.nodes)


def test_explain_node_reason_matches_decision(analysis):
    d = analysis.compression.decisions[0]
    exp = explain_node(analysis.graph, analysis.compression, d.node_id)
    assert exp.reason == d.reason
    assert exp.decision == d.decision.value


def test_explain_node_unknown_id_raises(analysis):
    with pytest.raises(ExplanationError):
        explain_node(analysis.graph, analysis.compression, "N99")


def test_search_nodes_matches_text_and_type(analysis):
    assert set(search_nodes(analysis.compression, "tokyo")) <= set(analysis.graph.nodes)
    assert search_nodes(analysis.compression, "   ") == []


# ==========================================================================
# 4. JSON SERIALIZATION (for the export buttons)
# ==========================================================================
def test_graph_to_dict_is_json_safe(analysis):
    json.dumps(graph_to_dict(analysis.graph))


def test_compression_and_evaluation_to_dict_are_json_safe(analysis):
    json.dumps(analysis.compression.to_dict())
    json.dumps(analysis.evaluation.to_dict())


def test_explanations_to_dict_are_json_safe(analysis):
    explanations = explain_all(analysis.graph, analysis.compression)
    json.dumps([e.to_dict() for e in explanations.values()])