"""Day 1 tests: preprocessing, extraction, embeddings, graph, importance, metrics.

Run from the project root:   pytest -q
Most tests use a deterministic FakeModel or hand-made vectors (fast, offline).
The two 'real_model' tests use all-MiniLM-L6-v2 and SKIP if it cannot be loaded.
"""

from __future__ import annotations

import json
import zlib

import networkx as nx
import numpy as np
import pytest

from src.embeddings.embedder import (
    EmbeddingError,
    EmbeddingResult,
    embed_texts,
    embed_units,
    get_model,
    similarity_matrix,
)
from src.extraction.unit_classifier import (
    ClassificationError,
    ClassifiedUnit,
    classify_unit,
    classify_units,
    type_counts,
)
from src.graph.builder import (
    GraphBuildError,
    GraphConfig,
    build_semantic_graph,
    graph_to_dict,
)
from src.graph.importance import (
    ImportanceConfig,
    ImportanceError,
    annotate_graph,
    compute_importance,
)
from src.graph.metrics import GraphMetricsError, compute_graph_metrics
from src.preprocessing.cleaner import (
    CleanerConfig,
    EmptyPromptError,
    PromptCleaningError,
    clean_prompt,
)
from src.preprocessing.segmenter import InformationUnit, segment_prompt

# --------------------------------------------------------------------------
# Shared sample data and helpers
# --------------------------------------------------------------------------
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

SAMPLE_TYPES = [
    "role", "instruction", "instruction", "requirement",
    "constraint", "constraint", "output_format",
]


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


def make_classified(specs):
    """Build ClassifiedUnit objects from (text, type) pairs."""
    units = []
    for i, (text, utype) in enumerate(specs):
        info = InformationUnit(f"N{i + 1:02d}", i, text, "sentence", i + 1)
        units.append(ClassifiedUnit(info, utype, 1.0, ("test",), {utype: 1.0}))
    return tuple(units)


def make_embeddings(classified, vectors):
    """Wrap hand-made vectors in an EmbeddingResult (same ids, same order)."""
    return EmbeddingResult(
        unit_ids=tuple(c.unit_id for c in classified),
        texts=tuple(c.text for c in classified),
        vectors=np.array(vectors, dtype=np.float32),
        model_name="test-model",
    )


# Path graph N01 - N02 - N03 plus isolated N04, with cosine sims 0.8 and 0.6.
PATH_SPECS = [
    ("Plan a trip.", "instruction"),
    ("Budget under $500.", "constraint"),
    ("The trip is to Tokyo.", "context"),
    ("Blue skies.", "other"),
]
PATH_VECTORS = [[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]]


def weighted_path_graph(with_isolated: bool = True) -> nx.Graph:
    """Same shape as the path graph, built directly (for module-independent tests)."""
    g = nx.Graph()
    nodes = [("N01", "instruction"), ("N02", "constraint"), ("N03", "context")]
    if with_isolated:
        nodes.append(("N04", "other"))
    for nid, t in nodes:
        g.add_node(nid, node_id=nid, text=f"demo {nid}", type=t)
    g.add_edge("N01", "N02", weight=0.8)
    g.add_edge("N02", "N03", weight=0.6)
    return g


@pytest.fixture(scope="module")
def cleaned():
    return clean_prompt(SAMPLE_PROMPT)


@pytest.fixture(scope="module")
def segmented(cleaned):
    return segment_prompt(cleaned)


@pytest.fixture(scope="module")
def classified(segmented):
    return classify_units(segmented)


@pytest.fixture
def path_units():
    return make_classified(PATH_SPECS)


@pytest.fixture
def path_graph(path_units):
    return build_semantic_graph(
        path_units,
        make_embeddings(path_units, PATH_VECTORS),
        GraphConfig(similarity_threshold=0.5),
    )


# ==========================================================================
# 1. CLEANING
# ==========================================================================
def test_clean_normalizes_whitespace_quotes_and_line_endings():
    raw = (
        "  You are a   senior data analyst.\r\n\r\n\r\n\r\n"
        "Task:   Summarize the report.\t\r\n"
        "- Use \u201cbullet points\u201d\n"
        "   - Keep it short  \n"
    )
    expected = (
        "You are a senior data analyst.\n\n"
        "Task: Summarize the report.\n"
        '- Use "bullet points"\n'
        "   - Keep it short"
    )
    result = clean_prompt(raw)
    assert result.text == expected
    assert result.original == raw
    assert result.warnings == ()


def test_clean_preserves_code_fence_content():
    raw = "Run this:\n```python\nx  =  1\n\ty = 2   \n```\nThen stop."
    result = clean_prompt(raw)
    assert "```python\nx  =  1\n\ty = 2\n```" in result.text


def test_clean_removes_invisible_and_unusual_characters():
    raw = "Hello\x07\u200b\u00a0\u00a0world\u2003again"
    assert clean_prompt(raw).text == "Hello world again"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t \n", "\u200b"])
def test_clean_rejects_empty_prompt(raw):
    with pytest.raises(EmptyPromptError):
        clean_prompt(raw)


def test_clean_rejects_non_string():
    with pytest.raises(TypeError):
        clean_prompt(None)  # type: ignore[arg-type]


def test_clean_rejects_oversized_prompt():
    with pytest.raises(PromptCleaningError):
        clean_prompt("word " * 10, CleanerConfig(max_chars=10))


def test_clean_warns_on_very_short_prompt():
    result = clean_prompt("Hello")
    assert len(result.warnings) == 1
    assert "Very short" in result.warnings[0]


# ==========================================================================
# 2. SEGMENTATION
# ==========================================================================
def test_segment_sample_prompt_structure(segmented):
    assert len(segmented.units) == 7
    assert [u.unit_id for u in segmented.units] == [f"N0{i}" for i in range(1, 8)]
    assert [u.kind for u in segmented.units] == (
        ["sentence"] * 4 + ["list_item"] * 2 + ["code_block"]
    )
    assert [u.text for u in segmented.units] == [
        "You are an expert travel planner.",
        "Your goal is to help users plan trips.",
        "Task: Create a 3-day itinerary for Tokyo.",
        "Include e.g. museums and parks.",       # 'e.g.' must not cause a split
        "Budget must not exceed $500.",
        "Avoid crowded places.",
        '```json\n{"days": []}\n```',
    ]
    assert segmented.warnings == ()


def test_segment_attaches_headings_and_line_numbers(segmented):
    assert [u.parent_heading for u in segmented.units] == [
        None, None, None, None, "Constraints", "Constraints", "Output format",
    ]
    assert [u.line_start for u in segmented.units] == [1, 1, 3, 3, 6, 7, 10]


def test_segment_code_block_is_one_unit(segmented):
    blocks = [u for u in segmented.units if u.kind == "code_block"]
    assert len(blocks) == 1
    assert blocks[0].text.startswith("```json") and blocks[0].text.endswith("```")


def test_segment_list_item_continuation():
    result = segment_prompt("- first item that wraps\n  onto the next line\n- second item here")
    assert [u.text for u in result.units] == [
        "first item that wraps onto the next line",
        "second item here",
    ]
    assert all(u.kind == "list_item" for u in result.units)


def test_segment_joins_soft_wrapped_lines():
    result = segment_prompt("Write a summary of the\nreport for the team.")
    assert [u.text for u in result.units] == ["Write a summary of the report for the team."]


def test_segment_markdown_heading():
    result = segment_prompt("# Rules\nBe brief.\nBe clear.")
    assert [u.text for u in result.units] == ["Be brief.", "Be clear."]
    assert all(u.parent_heading == "Rules" for u in result.units)


def test_segment_warns_on_duplicates():
    result = segment_prompt("Be concise.\nBe concise.")
    assert len(result.units) == 2                       # kept, not removed
    assert any("duplicates" in w for w in result.warnings)


def test_segment_warns_on_single_unit():
    result = segment_prompt("Summarize this document carefully.")
    assert len(result.units) == 1
    assert any("Only one information unit" in w for w in result.warnings)


def test_segment_rejects_empty_and_invalid_input():
    with pytest.raises(EmptyPromptError):
        segment_prompt("   ")
    with pytest.raises(TypeError):
        segment_prompt(123)  # type: ignore[arg-type]


# ==========================================================================
# 3. CLASSIFICATION
# ==========================================================================
def _unit(text, kind="sentence", heading=None):
    return InformationUnit("N01", 0, text, kind, 1, heading)


def test_classify_sample_prompt_types(classified):
    assert [c.unit_type for c in classified] == SAMPLE_TYPES
    assert all(0.0 < c.confidence <= 1.0 for c in classified)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Act as a senior lawyer.", "role"),
        ("For example, use short words.", "example"),
        ("What is the capital of France?", "instruction"),
        ("Blue skies over Pune.", "other"),
        ("You must cite sources.", "requirement"),
        ("Do not reveal the answer.", "constraint"),
        ("Respond in JSON.", "output_format"),
    ],
)
def test_classify_rule_examples(text, expected):
    assert classify_unit(_unit(text)).unit_type == expected


def test_classify_explanation_lists_rules_and_scores(classified):
    text = classified[0].explain()
    assert "N01" in text and "role" in text and "you_are_role" in text


def test_classify_unmatched_text_is_other_with_zero_confidence():
    result = classify_unit(_unit("Blue skies over Pune."))
    assert result.unit_type == "other"
    assert result.confidence == 0.0
    assert "no rule matched" in result.explain()


def test_type_counts(classified):
    assert type_counts(classified) == {
        "role": 1, "instruction": 2, "constraint": 2,
        "requirement": 1, "output_format": 1,
    }


def test_classify_units_rejects_empty_input():
    with pytest.raises(ClassificationError):
        classify_units([])


def test_classify_unit_rejects_wrong_type():
    with pytest.raises(TypeError):
        classify_unit("not a unit")  # type: ignore[arg-type]


def test_classify_code_block_uses_fence_language():
    assert classify_unit(_unit("```json\n{}\n```", kind="code_block")).unit_type == "output_format"
    assert classify_unit(_unit("```python\nx = 1\n```", kind="code_block")).unit_type == "context"


# ==========================================================================
# 4. EMBEDDINGS
# ==========================================================================
def test_embed_texts_shape_dtype_and_normalization():
    vectors = embed_texts(["a b", "c d e"], model=FakeModel())
    assert vectors.shape == (2, FakeModel.DIM)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embed_units_preserves_order_and_ids(classified):
    result = embed_units(classified, model=FakeModel())
    assert result.unit_ids == tuple(c.unit_id for c in classified)
    assert result.texts == tuple(c.text for c in classified)
    assert result.vectors.shape == (7, FakeModel.DIM)
    assert result.dim == FakeModel.DIM


def test_embed_units_vector_lookup(classified):
    result = embed_units(classified, model=FakeModel())
    assert np.array_equal(result.vector("N03"), result.vectors[2])
    with pytest.raises(KeyError):
        result.vector("N99")


def test_embed_units_accepts_segmentation_result(segmented):
    assert len(embed_units(segmented, model=FakeModel())) == 7


def test_embed_is_deterministic(classified):
    a = embed_units(classified, model=FakeModel())
    b = embed_units(classified, model=FakeModel())
    assert np.allclose(a.vectors, b.vectors)


def test_embed_rejects_empty_batch():
    with pytest.raises(EmbeddingError, match="No texts"):
        embed_texts([], model=FakeModel())


def test_embed_rejects_blank_text():
    with pytest.raises(EmbeddingError, match="position 1"):
        embed_texts(["ok text", "   "], model=FakeModel())


def test_embed_rejects_string_input():
    with pytest.raises(TypeError):
        embed_texts("hello", model=FakeModel())  # type: ignore[arg-type]


def test_embed_rejects_duplicate_unit_ids():
    units = make_classified([("a b", "other"), ("c d", "other")])
    with pytest.raises(EmbeddingError, match="Duplicate"):
        embed_units((units[0], units[0]), model=FakeModel())


def test_embed_wraps_model_failure():
    class BrokenModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(EmbeddingError, match="boom") as exc_info:
        embed_texts(["hello world"], model=BrokenModel())
    assert isinstance(exc_info.value.__cause__, RuntimeError)   # not swallowed


def test_embed_detects_non_finite_output():
    class NanModel:
        def encode(self, texts, **kwargs):
            return np.full((len(texts), 4), np.nan)

    with pytest.raises(EmbeddingError, match="NaN"):
        embed_texts(["hello world"], model=NanModel())


def test_similarity_matrix_properties():
    vectors = embed_texts(["red apple", "green apple", "fast car"], model=FakeModel())
    sim = similarity_matrix(vectors)
    assert sim.shape == (3, 3)
    assert np.allclose(sim, sim.T, atol=1e-6)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    assert sim.min() >= -1.0 and sim.max() <= 1.0


def test_similarity_matrix_rejects_zero_vector():
    with pytest.raises(EmbeddingError):
        similarity_matrix(np.zeros((2, 3), dtype=np.float32))


def test_embed_units_rejects_objects_without_text():
    with pytest.raises(TypeError):
        embed_units([object()], model=FakeModel())


def _real_model_or_skip():
    try:
        return get_model()
    except EmbeddingError as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")


def test_real_model_embeddings_shape_norm_and_cache(classified):
    model = _real_model_or_skip()
    assert get_model() is model                      # loaded once, then reused
    first = embed_units(classified)
    second = embed_units(classified)
    assert first.vectors.shape == (7, 384)
    assert first.model_name == "all-MiniLM-L6-v2"
    assert np.allclose(np.linalg.norm(first.vectors, axis=1), 1.0, atol=1e-4)
    assert np.allclose(first.vectors, second.vectors, atol=1e-5)


def test_real_model_semantic_ordering():
    _real_model_or_skip()
    texts = [
        "A cat sat on the mat.",
        "A kitten rested on the rug.",
        "Quarterly revenue increased by ten percent.",
    ]
    sim = similarity_matrix(embed_texts(texts))
    assert sim[0, 1] > sim[0, 2]


# ==========================================================================
# 5. GRAPH CONSTRUCTION
# ==========================================================================
def test_graph_nodes_have_required_attributes(path_graph):
    assert path_graph.number_of_nodes() == 4
    node = path_graph.nodes["N02"]
    assert node["node_id"] == "N02"
    assert node["text"] == "Budget under $500."
    assert node["type"] == "constraint"
    assert node["embedding"].shape == (3,)
    assert node["confidence"] == 1.0


def test_graph_edges_follow_threshold_and_have_required_attributes(path_graph):
    edges = {frozenset(e) for e in path_graph.edges}
    assert edges == {frozenset({"N01", "N02"}), frozenset({"N02", "N03"})}
    data = path_graph["N01"]["N02"]
    assert data["source"] == "N01" and data["target"] == "N02"
    assert data["weight"] == pytest.approx(0.8, abs=1e-5)
    assert data["relationship"] == "semantic_similarity"
    assert data["near_duplicate"] is False
    assert path_graph["N02"]["N03"]["weight"] == pytest.approx(0.6, abs=1e-5)


def test_graph_has_no_self_loops(path_graph):
    assert nx.number_of_selfloops(path_graph) == 0
    assert path_graph.graph["near_duplicate_pairs"] == []


def test_graph_threshold_is_configurable(path_units):
    emb = make_embeddings(path_units, PATH_VECTORS)
    strict = build_semantic_graph(path_units, emb, GraphConfig(similarity_threshold=0.7))
    loose = build_semantic_graph(path_units, emb, GraphConfig(similarity_threshold=0.5))
    assert strict.number_of_edges() == 1
    assert loose.number_of_edges() == 2


def test_graph_warns_when_no_edges(path_units):
    emb = make_embeddings(path_units, PATH_VECTORS)
    g = build_semantic_graph(path_units, emb, GraphConfig(similarity_threshold=0.9))
    assert g.number_of_edges() == 0
    assert any("No semantic edges" in w for w in g.graph["warnings"])


def test_graph_flags_near_duplicates():
    units = make_classified([("Be concise.", "requirement"), ("Be concise.", "requirement")])
    g = build_semantic_graph(units, make_embeddings(units, [[1, 0, 0], [1, 0, 0]]))
    assert g.number_of_edges() == 1
    assert g["N01"]["N02"]["near_duplicate"] is True
    assert len(g.graph["near_duplicate_pairs"]) == 1
    assert any("near-duplicates" in w for w in g.graph["warnings"])
    assert g.number_of_nodes() == 2                      # nothing merged or removed


def test_graph_single_unit():
    units = make_classified([("Summarize this.", "instruction")])
    g = build_semantic_graph(units, make_embeddings(units, [[1, 0, 0]]))
    assert g.number_of_nodes() == 1 and g.number_of_edges() == 0
    assert any("Only one information unit" in w for w in g.graph["warnings"])


def test_graph_rejects_mismatched_embeddings(path_units):
    reversed_emb = make_embeddings(tuple(reversed(path_units)), PATH_VECTORS)
    with pytest.raises(GraphBuildError, match="differ"):
        build_semantic_graph(path_units, reversed_emb)


def test_graph_rejects_no_units():
    with pytest.raises(GraphBuildError, match="zero"):
        build_semantic_graph((), None)  # type: ignore[arg-type]


def test_graph_rejects_wrong_types(path_units):
    emb = make_embeddings(path_units, PATH_VECTORS)
    with pytest.raises(TypeError):
        build_semantic_graph(["x"], emb)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        build_semantic_graph(path_units, "not embeddings")  # type: ignore[arg-type]


def test_graph_config_validation():
    with pytest.raises(GraphBuildError):
        GraphConfig(similarity_threshold=0.0)
    with pytest.raises(GraphBuildError):
        GraphConfig(similarity_threshold=1.5)
    with pytest.raises(GraphBuildError):
        GraphConfig(similarity_threshold=0.6, near_duplicate_threshold=0.5)


def test_graph_to_dict_is_json_safe(path_graph):
    data = graph_to_dict(path_graph)
    json.dumps(data)
    assert len(data["nodes"]) == 4 and len(data["edges"]) == 2
    assert "embedding" not in data["nodes"][0]
    with_emb = graph_to_dict(path_graph, include_embeddings=True)
    json.dumps(with_emb)
    assert len(with_emb["nodes"][0]["embedding"]) == 3


# ==========================================================================
# 6. IMPORTANCE
# ==========================================================================
def test_importance_matches_hand_calculation(path_graph):
    result = compute_importance(path_graph)
    # N02: 0.40*0.9 + 0.25*1.0 + 0.15*1.0 + 0.20*1.0 = 0.96
    assert result["N02"].importance == pytest.approx(0.96, abs=1e-3)
    assert result["N02"].connectivity == pytest.approx(1.0)
    assert result["N02"].centrality == pytest.approx(1.0)
    # N01: 0.40*1.0 + 0.25*(0.8/1.4) = 0.5429
    assert result["N01"].importance == pytest.approx(0.5429, abs=1e-3)
    # N03: 0.40*0.5 + 0.25*(0.6/1.4) = 0.3071
    assert result["N03"].importance == pytest.approx(0.3071, abs=1e-3)
    # N04 (isolated, 'other'): 0.40*0.2 = 0.08
    assert result["N04"].importance == pytest.approx(0.08, abs=1e-3)


def test_importance_ranking_order(path_graph):
    ranked = compute_importance(path_graph).ranked()
    assert [r.node_id for r in ranked] == ["N02", "N01", "N03", "N04"]


def test_importance_top_k(path_graph):
    result = compute_importance(path_graph)
    assert [r.node_id for r in result.top(2)] == ["N02", "N01"]
    with pytest.raises(ImportanceError):
        result.top(0)


def test_importance_values_are_in_unit_interval(path_graph):
    for r in compute_importance(path_graph).scores.values():
        assert 0.0 <= r.importance <= 1.0
        assert 0.0 <= r.connectivity <= 1.0
        assert 0.0 <= r.centrality <= 1.0


def test_importance_is_deterministic(path_graph):
    a = compute_importance(path_graph)
    b = compute_importance(path_graph)
    assert {k: v.importance for k, v in a.scores.items()} == {
        k: v.importance for k, v in b.scores.items()
    }


def test_importance_explain_reports_fields(path_graph):
    text = compute_importance(path_graph)["N02"].explain()
    assert "Node N02" in text
    assert "Type: constraint" in text
    assert "Importance: 0.96" in text
    assert "Protected: yes" in text


def test_importance_protection_flags(path_graph):
    result = compute_importance(path_graph)
    assert result["N02"].protection == 1.0       # constraint
    assert result["N01"].protection == 0.0       # instruction


def test_importance_single_node_graph():
    g = nx.Graph()
    g.add_node("A", text="x", type="instruction")
    r = compute_importance(g)["A"]
    assert r.importance == pytest.approx(0.40)   # type weight only
    assert r.connectivity == 0.0 and r.centrality == 0.0


def test_importance_rejects_empty_graph():
    with pytest.raises(ImportanceError, match="empty graph"):
        compute_importance(nx.Graph())


def test_importance_rejects_missing_or_unknown_type():
    g = nx.Graph()
    g.add_node("A", text="x")
    with pytest.raises(ImportanceError, match="no 'type'"):
        compute_importance(g)
    g2 = nx.Graph()
    g2.add_node("A", text="x", type="banana")
    with pytest.raises(ImportanceError, match="unknown type"):
        compute_importance(g2)


def test_importance_config_validation():
    with pytest.raises(ImportanceError):
        ImportanceConfig(w_type=0.5)                         # weights sum to 1.10
    with pytest.raises(ImportanceError):
        ImportanceConfig(type_weights={"instruction": 1.0})  # missing types


def test_annotate_graph_writes_attributes(path_graph):
    result = compute_importance(path_graph)
    annotated = annotate_graph(path_graph, result)
    assert annotated is path_graph
    node = path_graph.nodes["N02"]
    assert node["importance"] == pytest.approx(0.96, abs=1e-3)
    assert {"connectivity", "centrality", "protection"} <= set(node)


# ==========================================================================
# 7. GRAPH METRICS
# ==========================================================================
def test_metrics_on_path_graph_with_isolated_node():
    m = compute_graph_metrics(weighted_path_graph())
    assert m.num_nodes == 4 and m.num_edges == 2
    assert m.density == pytest.approx(1 / 3)
    assert m.average_degree == pytest.approx(1.0)
    assert m.num_components == 2
    assert m.component_sizes == (3, 1)
    assert m.largest_component_size == 3
    assert m.is_connected is False
    assert m.isolated_nodes == ("N04",)
    assert m.average_clustering == 0.0
    assert m.average_edge_weight == pytest.approx(0.7)
    assert m.average_shortest_path_length is None        # not connected
    assert m.diameter is None
    assert m.type_counts == {"instruction": 1, "constraint": 1, "context": 1, "other": 1}


def test_metrics_top_centrality():
    m = compute_graph_metrics(weighted_path_graph())
    assert [n for n, _ in m.top_by_degree] == ["N02", "N01", "N03"]
    assert m.top_by_degree[0][1] == pytest.approx(2 / 3)
    assert m.top_by_betweenness[0][0] == "N02"
    assert m.top_by_betweenness[0][1] == pytest.approx(1 / 3)


def test_metrics_connected_graph_has_path_metrics():
    m = compute_graph_metrics(weighted_path_graph(with_isolated=False))
    assert m.is_connected is True and m.num_components == 1
    assert m.average_shortest_path_length == pytest.approx(4 / 3)
    assert m.diameter == 2


def test_metrics_complete_graph():
    g = nx.Graph()
    for n in ("A", "B", "C"):
        g.add_node(n, type="context")
    g.add_edge("A", "B", weight=0.5)
    g.add_edge("B", "C", weight=0.5)
    g.add_edge("A", "C", weight=0.5)
    m = compute_graph_metrics(g)
    assert m.density == pytest.approx(1.0)
    assert m.average_degree == pytest.approx(2.0)
    assert m.average_clustering == pytest.approx(1.0)
    assert m.average_shortest_path_length == pytest.approx(1.0)
    assert m.diameter == 1


def test_metrics_single_node():
    g = nx.Graph()
    g.add_node("A", type="instruction")
    m = compute_graph_metrics(g)
    assert m.num_nodes == 1 and m.num_edges == 0
    assert m.is_connected is True
    assert m.average_shortest_path_length is None and m.diameter is None
    assert m.degree_centrality["A"] == 0.0
    assert m.average_degree == 0.0


def test_metrics_graph_without_edges():
    g = nx.Graph()
    for n in ("A", "B", "C"):
        g.add_node(n, type="context")
    m = compute_graph_metrics(g)
    assert m.num_components == 3 and m.component_sizes == (1, 1, 1)
    assert m.isolated_nodes == ("A", "B", "C")
    assert m.average_edge_weight is None
    assert m.density == 0.0


def test_metrics_summary_and_dict():
    m = compute_graph_metrics(weighted_path_graph())
    summary = m.summary()
    assert "Nodes: 4" in summary
    assert "Components: 2 (sizes: 3, 1)" in summary
    assert "Isolated nodes: N04" in summary
    data = m.to_dict()
    json.dumps(data)
    assert data["num_edges"] == 2


def test_metrics_reject_invalid_input():
    with pytest.raises(GraphMetricsError, match="empty"):
        compute_graph_metrics(nx.Graph())
    with pytest.raises(TypeError):
        compute_graph_metrics("not a graph")  # type: ignore[arg-type]
    with pytest.raises(GraphMetricsError):
        compute_graph_metrics(weighted_path_graph(), top_k=0)
    g = nx.Graph()
    g.add_edge("A", "B")                                  # edge without 'weight'
    with pytest.raises(GraphMetricsError, match="weight"):
        compute_graph_metrics(g)


# ==========================================================================
# 8. END-TO-END (fake model, so it runs offline)
# ==========================================================================
def test_full_pipeline_with_fake_model():
    classified_units = classify_units(segment_prompt(clean_prompt(SAMPLE_PROMPT)))
    embeddings = embed_units(classified_units, model=FakeModel())
    graph = build_semantic_graph(
        classified_units, embeddings, GraphConfig(similarity_threshold=0.05)
    )
    importance = compute_importance(graph)
    metrics = compute_graph_metrics(graph)

    assert metrics.num_nodes == 7
    assert metrics.num_edges == graph.number_of_edges()
    assert set(importance.scores) == set(graph.nodes)
    assert all(0.0 <= r.importance <= 1.0 for r in importance.scores.values())
    assert metrics.type_counts == type_counts(classified_units)


def test_pipeline_single_unit_prompt():
    units = classify_units(segment_prompt(clean_prompt("Summarize this document carefully.")))
    assert len(units) == 1 and units[0].unit_type == "instruction"
    graph = build_semantic_graph(units, embed_units(units, model=FakeModel()))
    assert graph.number_of_nodes() == 1 and graph.number_of_edges() == 0
    assert len(compute_importance(graph)) == 1
    assert compute_graph_metrics(graph).num_nodes == 1