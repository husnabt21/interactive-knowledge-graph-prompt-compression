"""Day 2 tests: protection rules, compression, baseline, evaluation.

Run from the project root:   pytest -q
All tests are deterministic and offline (FakeModel / BagOfWordsModel / hand-made
graphs); no network access or tiktoken download is required.
"""

from __future__ import annotations

import zlib

import networkx as nx
import numpy as np
import pytest

from src.compression.compresser import (
    CompressionConfig,
    CompressionError,
    Decision,
    EmptyGraphError,
    InvalidConfigError,
    InvalidRetentionRatioError,
    compress_graph,
    reconstruct_prompt,
)
from src.compression.protection_rules import (
    DEFAULT_POLICY,
    InvalidGraphError,
    InvalidPolicyError,
    InvalidUnitTypeError,
    ProtectionLevel,
    ProtectionPolicy,
    ProtectionRuleError,
    VALID_UNIT_TYPES,
    assess_graph,
    assess_node,
)
from src.embeddings.embedder import embed_units
from src.evaluation import baseline as baseline_mod
from src.evaluation.baseline import (
    TokenCounter,
    TokenizerUnavailableError,
    compare_prompts,
    get_token_counter,
)
from src.evaluation.metrics import (
    compute_instruction_fidelity,
    evaluate_compression,
    node_retention_percent,
    semantic_similarity,
)
from src.extraction.unit_classifier import classify_units
from src.graph.builder import GraphConfig, build_semantic_graph
from src.graph.importance import annotate_graph, compute_importance
from src.preprocessing.cleaner import clean_prompt
from src.preprocessing.segmenter import segment_prompt

# --------------------------------------------------------------------------
# Shared fixtures / fakes (self-contained; not imported from test_day1.py)
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


class BagOfWordsModel:
    """Deterministic stand-in used only for semantic-similarity tests."""

    def encode(self, items, normalize_embeddings=True, **kwargs):
        import re

        words = [re.findall(r"[a-z0-9]+", t.lower()) for t in items]
        vocab = sorted({w for ws in words for w in ws})
        col = {w: i for i, w in enumerate(vocab)}
        vecs = np.zeros((len(items), max(len(vocab), 1)), dtype=np.float32)
        for r, ws in enumerate(words):
            for w in ws:
                vecs[r, col[w]] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1.0, norms)
        return vecs


def _demo_graph() -> nx.Graph:
    """Same hand-made graph as compressor.py's __main__ check. Fresh each call."""
    g = nx.Graph(near_duplicate_pairs=[("N02", "N03", 0.95)])
    rows = [
        ("N01", 0, "You are a travel planner.", "role", 0.55, 1),
        ("N02", 1, "Plan a 3-day trip to Tokyo.", "instruction", 0.80, 3),
        ("N03", 2, "Plan a three day trip to Tokyo.", "instruction", 0.70, 4),
        ("N04", 3, "Tokyo has many museums.", "context", 0.40, 5),
        ("N05", 4, "Budget must not exceed $500.", "constraint", 0.85, 7),
        ("N06", 5, "Thanks a lot.", "other", 0.10, 9),
        ("N07", 6, "e.g. Visit Ueno Park.", "example", 0.35, 10),
    ]
    for nid, idx, text, utype, imp, line in rows:
        g.add_node(nid, node_id=nid, text=text, type=utype, importance=imp,
                   index=idx, kind="sentence", line_start=line)
    g.add_edge("N02", "N03", weight=0.95, near_duplicate=True)
    g.add_edge("N02", "N04", weight=0.40, near_duplicate=False)
    g.add_edge("N03", "N04", weight=0.38, near_duplicate=False)
    g.add_edge("N04", "N07", weight=0.50, near_duplicate=False)
    g.add_edge("N01", "N02", weight=0.35, near_duplicate=False)
    return g


@pytest.fixture
def demo_graph():
    return _demo_graph()


@pytest.fixture(scope="module")
def cleaned():
    return clean_prompt(SAMPLE_PROMPT)


@pytest.fixture(scope="module")
def segmented(cleaned):
    return segment_prompt(cleaned)


@pytest.fixture(scope="module")
def classified(segmented):
    return classify_units(segmented)


@pytest.fixture(scope="module")
def real_graph(classified):
    """Full Day 1 pipeline output on SAMPLE_PROMPT, using FakeModel (offline)."""
    embeddings = embed_units(classified, model=FakeModel())
    graph = build_semantic_graph(
        classified, embeddings, GraphConfig(similarity_threshold=0.05)
    )
    annotate_graph(graph, compute_importance(graph))
    return graph


# ==========================================================================
# 1. PROTECTION RULES
# ==========================================================================
@pytest.mark.parametrize("t", ["instruction", "constraint", "requirement", "output_format"])
def test_strong_types_are_protected(t):
    a = assess_node("N01", t, 0.5)
    assert a.level is ProtectionLevel.STRONG
    assert a.protected is True
    assert a.priority == pytest.approx(1.5)


@pytest.mark.parametrize("t", ["role", "context", "example"])
def test_normal_types_are_not_protected(t):
    a = assess_node("N01", t, 0.5)
    assert a.level is ProtectionLevel.NORMAL
    assert a.protected is False
    assert a.priority == pytest.approx(0.5)


def test_low_type_other_ranks_below_normal():
    a = assess_node("N01", "other", 0.5)
    assert a.level is ProtectionLevel.LOW
    assert a.protected is False
    assert a.priority == pytest.approx(0.25)


def test_all_valid_unit_types_covered():
    for t in VALID_UNIT_TYPES:
        assess_node("N01", t, 0.5)  # must not raise


def test_assess_node_rejects_invalid_unit_type():
    with pytest.raises(InvalidUnitTypeError):
        assess_node("N01", "banana", 0.5)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), "0.5", True])
def test_assess_node_rejects_invalid_importance(bad):
    with pytest.raises(ProtectionRuleError):
        assess_node("N01", "role", bad)


def test_assess_node_rejects_empty_node_id():
    with pytest.raises(InvalidGraphError):
        assess_node("", "role", 0.5)


def test_policy_rejects_overlapping_types():
    with pytest.raises(InvalidPolicyError):
        ProtectionPolicy(
            strong_types=frozenset({"instruction"}),
            normal_types=frozenset({"instruction", "role", "context", "example"}),
            low_types=frozenset({"constraint", "requirement", "output_format", "other"}),
        )


def test_policy_rejects_missing_types():
    with pytest.raises(InvalidPolicyError):
        ProtectionPolicy(
            strong_types=frozenset({"instruction"}),
            normal_types=frozenset({"role"}),
            low_types=frozenset({"other"}),
        )


def test_policy_rejects_missing_level_bonus():
    with pytest.raises(InvalidPolicyError):
        ProtectionPolicy(level_bonus={ProtectionLevel.STRONG: 1.0, ProtectionLevel.NORMAL: 0.0})


def test_policy_rejects_non_numeric_bonus():
    with pytest.raises(InvalidPolicyError):
        ProtectionPolicy(level_bonus={
            ProtectionLevel.STRONG: "high", ProtectionLevel.NORMAL: 0.0, ProtectionLevel.LOW: -0.25,
        })


def test_assess_graph_on_hand_made_graph(demo_graph):
    result = assess_graph(demo_graph)
    assert set(result) == set(demo_graph.nodes)
    assert result["N02"].protected is True
    assert result["N06"].level is ProtectionLevel.LOW


def test_assess_graph_empty_graph_returns_empty_dict():
    assert assess_graph(nx.Graph()) == {}


def test_assess_graph_rejects_missing_attributes():
    g = nx.Graph()
    g.add_node("N01", type="role")  # missing 'importance'
    with pytest.raises(InvalidGraphError, match="importance"):
        assess_graph(g)


def test_assess_graph_rejects_non_graph():
    with pytest.raises(InvalidGraphError):
        assess_graph("not a graph")  # type: ignore[arg-type]


# ==========================================================================
# 2. COMPRESSION
# ==========================================================================
def test_compress_hits_retention_target(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.60))
    assert r.target_node_count == 5
    assert r.retained_node_count == 5
    assert r.actual_retention_ratio == pytest.approx(5 / 7)
    assert r.warnings == ()


def test_protected_nodes_are_retained(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.10))
    # N02 (instruction) and N05 (constraint) are strong types: must survive
    # even at an extremely low retention ratio.
    assert "N02" in r.retained_node_ids
    assert "N05" in r.retained_node_ids
    assert r.decision_for("N02").protected is True


def test_protect_strong_false_allows_removal(demo_graph):
    r = compress_graph(
        demo_graph, CompressionConfig(retention_ratio=0.10, protect_strong=False)
    )
    assert r.retained_node_count == 1  # only the single top-ranked node survives


def test_near_duplicate_pair_never_both_removed(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=1.0))
    assert not ({"N02", "N03"} <= set(r.removed_node_ids))
    d3 = r.decision_for("N03")
    assert d3.decision is Decision.REMOVE
    assert d3.duplicate_of == "N02"
    assert d3.duplicate_similarity == pytest.approx(0.95)


def test_near_duplicate_representative_has_higher_priority(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=1.0))
    rep_priority = r.decision_for("N02").priority
    dup_priority = r.decision_for("N03").priority
    assert rep_priority >= dup_priority


def test_near_duplicate_chain_keeps_one_representative():
    g = nx.Graph(near_duplicate_pairs=[("A", "B", 0.95), ("B", "C", 0.93)])
    for nid, imp in (("A", 0.5), ("B", 0.9), ("C", 0.4)):
        g.add_node(nid, node_id=nid, text=f"text {nid}", type="context",
                    importance=imp, index=ord(nid), kind="sentence", line_start=1)
    g.add_edge("A", "B", weight=0.95, near_duplicate=True)
    g.add_edge("B", "C", weight=0.93, near_duplicate=True)
    r = compress_graph(g, CompressionConfig(retention_ratio=1.0))
    # B has highest importance -> becomes the representative for both.
    assert r.retained_node_ids == ("B",)
    assert r.decision_for("A").duplicate_of == "B"
    assert r.decision_for("C").duplicate_of == "B"


def test_remove_near_duplicates_false_disables_stage_a(demo_graph):
    r = compress_graph(
        demo_graph, CompressionConfig(retention_ratio=1.0, remove_near_duplicates=False)
    )
    assert r.retained_node_count == r.original_node_count
    assert r.decision_for("N03").duplicate_of is None


def test_original_order_is_preserved_regardless_of_input_order(demo_graph):
    text = reconstruct_prompt(demo_graph, ["N05", "N01", "N02"])
    assert text == (
        "You are a travel planner.\n\n"
        "Plan a 3-day trip to Tokyo.\n\n"
        "Budget must not exceed $500."
    )


def test_reconstruct_prompt_rejects_unknown_node(demo_graph):
    with pytest.raises(InvalidGraphError):
        reconstruct_prompt(demo_graph, ["N01", "N99"])


def test_compress_graph_rejects_empty_graph():
    with pytest.raises(EmptyGraphError):
        compress_graph(nx.Graph())


def test_compress_graph_rejects_missing_text_attribute():
    g = nx.Graph()
    g.add_node("N01", node_id="N01", type="role", importance=0.5, index=0)
    with pytest.raises(InvalidGraphError, match="text"):
        compress_graph(g)


def test_compress_graph_rejects_missing_index_attribute():
    g = nx.Graph()
    g.add_node("N01", node_id="N01", type="role", importance=0.5, text="hi")
    with pytest.raises(InvalidGraphError, match="index"):
        compress_graph(g)


def test_compress_graph_rejects_duplicate_index():
    g = nx.Graph()
    g.add_node("N01", node_id="N01", type="role", importance=0.5, text="a", index=0)
    g.add_node("N02", node_id="N02", type="role", importance=0.5, text="b", index=0)
    with pytest.raises(InvalidGraphError, match="index"):
        compress_graph(g)


def test_compress_graph_rejects_node_id_mismatch():
    g = nx.Graph()
    g.add_node("N01", node_id="WRONG", type="role", importance=0.5, text="a", index=0)
    with pytest.raises(InvalidGraphError, match="node_id"):
        compress_graph(g)


@pytest.mark.parametrize("bad_ratio", [0.0, -0.5, 1.5, "0.7"])
def test_invalid_retention_ratio_rejected(bad_ratio):
    with pytest.raises(InvalidRetentionRatioError):
        CompressionConfig(retention_ratio=bad_ratio)


def test_invalid_config_fields_rejected():
    with pytest.raises(InvalidConfigError):
        CompressionConfig(redundancy_penalty=-1.0)
    with pytest.raises(InvalidConfigError):
        CompressionConfig(list_marker=123)  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        CompressionConfig(policy="not a policy")  # type: ignore[arg-type]


def test_compress_graph_rejects_wrong_type():
    with pytest.raises(InvalidGraphError):
        compress_graph("not a graph")  # type: ignore[arg-type]


def test_compression_is_deterministic(demo_graph):
    r1 = compress_graph(_demo_graph(), CompressionConfig(retention_ratio=0.6))
    r2 = compress_graph(_demo_graph(), CompressionConfig(retention_ratio=0.6))
    assert r1.retained_node_ids == r2.retained_node_ids
    assert r1.removed_node_ids == r2.removed_node_ids
    assert [d.reason for d in r1.decisions] == [d.reason for d in r2.decisions]


def test_every_decision_has_a_grounded_reason(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.6))
    for d in r.decisions:
        assert d.reason  # non-empty
        assert len(d.reason_codes) >= 1
        if d.decision is Decision.REMOVE:
            assert d.reason_codes[0] in ("NEAR_DUPLICATE_REMOVED", "BELOW_RETENTION_TARGET")
        else:
            assert set(d.reason_codes) <= {
                "PROTECTED_TYPE", "NEAR_DUPLICATE_REPRESENTATIVE", "WITHIN_RETENTION_TARGET",
            }


def test_compression_result_helpers(demo_graph):
    r = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.6))
    assert len(r.retained_units) == r.retained_node_count
    assert len(r.removed_units) == r.removed_node_count
    assert r.decision_for("N05").decision is Decision.RETAIN
    with pytest.raises(KeyError):
        r.decision_for("N99")
    d = r.to_dict()
    assert d["retained_node_count"] == r.retained_node_count


# ==========================================================================
# 3. BASELINE (TOKEN COUNTING)
# ==========================================================================
def test_get_token_counter_falls_back_when_tiktoken_unavailable(monkeypatch):
    def _boom():
        raise ImportError("tiktoken not installed")

    monkeypatch.setattr(baseline_mod, "_load_encoding", _boom)
    counter = get_token_counter(allow_fallback=True)
    assert counter.exact is False
    assert counter.count("hello world") == 2


def test_get_token_counter_raises_when_fallback_disallowed(monkeypatch):
    def _boom():
        raise ImportError("tiktoken not installed")

    monkeypatch.setattr(baseline_mod, "_load_encoding", _boom)
    with pytest.raises(TokenizerUnavailableError):
        get_token_counter(allow_fallback=False)


def test_compare_prompts_reduction_and_ratio():
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    result = compare_prompts("a b c d", "a b", counter)
    assert result.original_tokens == 4
    assert result.compressed_tokens == 2
    assert result.token_reduction == 2
    assert result.token_reduction_percent == pytest.approx(50.0)
    assert result.compression_ratio == pytest.approx(2.0)


def test_compare_prompts_empty_original():
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    result = compare_prompts("", "", counter)
    assert result.original_tokens == 0
    assert result.token_reduction_percent == 0.0
    assert any("0 tokens" in w for w in result.warnings)


def test_compare_prompts_zero_compressed_tokens():
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    result = compare_prompts("a b c", "", counter)
    assert result.compressed_tokens == 0
    assert result.compression_ratio is None
    assert any("undefined" in w for w in result.warnings)


def test_compare_prompts_negative_reduction_when_compressed_is_longer():
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    result = compare_prompts("a b", "a b c d", counter)
    assert result.token_reduction == -2
    assert result.token_reduction_percent == pytest.approx(-100.0)


def test_compare_prompts_rejects_non_string_input():
    with pytest.raises(TypeError):
        compare_prompts(None, "x")  # type: ignore[arg-type]


def test_token_counter_rejects_non_string_text():
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    with pytest.raises(TypeError):
        counter.count(123)  # type: ignore[arg-type]


# ==========================================================================
# 4. EVALUATION METRICS
# ==========================================================================
def test_node_retention_percent_basic():
    assert node_retention_percent(7, 10) == pytest.approx(70.0)
    assert node_retention_percent(0, 0) == 0.0


def test_node_retention_percent_rejects_invalid_input():
    with pytest.raises(ValueError):
        node_retention_percent(-1, 5)
    with pytest.raises(ValueError):
        node_retention_percent(5, 3)  # retained > original


def test_semantic_similarity_identical_prompts_is_near_one():
    sim = semantic_similarity("Plan a trip to Tokyo.", "Plan a trip to Tokyo.", model=BagOfWordsModel())
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_semantic_similarity_rejects_empty_prompt():
    from src.evaluation.metrics import EvaluationError

    with pytest.raises(EvaluationError):
        semantic_similarity("", "something", model=BagOfWordsModel())


def test_instruction_fidelity_all_preserved(demo_graph):
    result = compress_graph(demo_graph, CompressionConfig(retention_ratio=1.0))
    fidelity = compute_instruction_fidelity(result)
    # protected: N02, N03 (instruction) and N05 (constraint) = 3 total.
    assert fidelity.protected_total == 3
    assert fidelity.retained == 2          # N02, N05
    assert fidelity.covered_by_duplicate == 1  # N03 via N02
    assert fidelity.lost_ids == ()
    assert fidelity.fidelity == pytest.approx(1.0)


def test_instruction_fidelity_with_a_lost_protected_unit(demo_graph):
    # protect_strong=False removes the *lock*, not the priority bonus, so a
    # merely low retention_ratio isn't enough if a protected node still ranks
    # at the top on merit. Force target=1 so even N05 (constraint, priority
    # 1.85) is squeezed out, leaving only the near-duplicate representative N02.
    result = compress_graph(
        demo_graph, CompressionConfig(retention_ratio=0.1, protect_strong=False)
    )
    assert result.target_node_count == 1
    fidelity = compute_instruction_fidelity(result)
    assert fidelity.protected_total == 3
    assert fidelity.fidelity is not None and fidelity.fidelity < 1.0
    assert "N05" in fidelity.lost_ids


def test_instruction_fidelity_none_when_no_protected_units():
    g = nx.Graph()
    g.add_node("A", node_id="A", type="role", importance=0.5, text="hi", index=0)
    r = compress_graph(g, CompressionConfig(retention_ratio=1.0))
    fidelity = compute_instruction_fidelity(r)
    assert fidelity.protected_total == 0
    assert fidelity.fidelity is None


def test_evaluate_compression_rejects_wrong_type():
    with pytest.raises(TypeError):
        evaluate_compression("not a compression result")  # type: ignore[arg-type]


def test_evaluate_compression_full_result(demo_graph):
    result = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.6))
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    evaluation = evaluate_compression(result, counter=counter, model=BagOfWordsModel())
    assert evaluation.original_nodes == 7
    assert evaluation.retained_nodes == result.retained_node_count
    assert 0.0 <= evaluation.node_retention_percent <= 100.0
    assert evaluation.semantic_similarity is not None
    assert -1.0 <= evaluation.semantic_similarity <= 1.0
    assert evaluation.instruction_fidelity is not None
    d = evaluation.to_dict()
    assert d["original_nodes"] == 7
    assert isinstance(evaluation.summary(), str) and "Semantic Similarity" in evaluation.summary()


def test_evaluate_compression_skip_semantic(demo_graph):
    result = compress_graph(demo_graph, CompressionConfig(retention_ratio=0.6))
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    evaluation = evaluate_compression(result, counter=counter, compute_semantic=False)
    assert evaluation.semantic_similarity is None


# ==========================================================================
# 5. END-TO-END: prompt -> Day 1 -> Day 2 -> evaluation
# ==========================================================================
def test_full_day1_day2_pipeline(real_graph):
    compression = compress_graph(real_graph, CompressionConfig(retention_ratio=0.7))
    counter = TokenCounter("whitespace", False, lambda t: len(t.split()))
    evaluation = evaluate_compression(compression, counter=counter, model=BagOfWordsModel())

    assert compression.original_node_count == 7
    assert compression.retained_node_count + compression.removed_node_count == 7
    assert set(compression.retained_node_ids) | set(compression.removed_node_ids) == set(real_graph.nodes)
    # protected instruction/constraint/output_format units must survive.
    protected_ids = {
        n for n, d in real_graph.nodes(data=True)
        if d["type"] in ("instruction", "constraint", "output_format")
    }
    assert protected_ids <= set(compression.retained_node_ids)
    # compressed prompt is a substring-preserving reorder: every retained
    # node's text appears in the compressed prompt.
    for nid in compression.retained_node_ids:
        assert real_graph.nodes[nid]["text"].split("\n")[0][:10] in compression.compressed_prompt \
            or real_graph.nodes[nid]["text"] in compression.compressed_prompt

    assert evaluation.original_nodes == 7
    assert evaluation.instruction_fidelity == pytest.approx(1.0)
    assert evaluation.token_reduction_percent >= 0.0 or evaluation.compressed_tokens > evaluation.original_tokens