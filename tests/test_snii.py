"""SNII tests: determinism, score range, ranking, edge cases."""

from __future__ import annotations

import networkx as nx
import pytest

from src.graph.importance import ImportanceError, compute_importance
from src.graph.snii import SNIIError, compute_snii, explain_snii


def _path_graph() -> nx.Graph:
    g = nx.Graph()
    for nid, t in (("N01", "instruction"), ("N02", "constraint"),
                   ("N03", "context"), ("N04", "other")):
        g.add_node(nid, node_id=nid, text=f"demo {nid}", type=t)
    g.add_edge("N01", "N02", weight=0.8)
    g.add_edge("N02", "N03", weight=0.6)
    return g


def test_snii_matches_known_importance_values():
    result = compute_snii(compute_importance(_path_graph()))
    assert result["N02"].snii == pytest.approx(0.96, abs=1e-3)
    assert result["N01"].snii == pytest.approx(0.5429, abs=1e-3)
    assert result["N03"].snii == pytest.approx(0.3071, abs=1e-3)
    assert result["N04"].snii == pytest.approx(0.08, abs=1e-3)


def test_snii_is_deterministic():
    g = _path_graph()
    r1 = compute_snii(compute_importance(g))
    r2 = compute_snii(compute_importance(g))
    assert [s.snii for s in r1.ranked()] == [s.snii for s in r2.ranked()]


def test_snii_score_range():
    result = compute_snii(compute_importance(_path_graph()))
    for s in result.scores.values():
        assert 0.0 <= s.snii <= 1.0


def test_snii_ranking_order():
    result = compute_snii(compute_importance(_path_graph()))
    assert [s.node_id for s in result.ranked()] == ["N02", "N01", "N03", "N04"]
    assert [s.rank for s in result.ranked()] == [1, 2, 3, 4]


def test_snii_top_k():
    result = compute_snii(compute_importance(_path_graph()))
    assert [s.node_id for s in result.top(2)] == ["N02", "N01"]
    with pytest.raises(SNIIError):
        result.top(0)


def test_snii_contributions_sum_to_snii():
    result = compute_snii(compute_importance(_path_graph()))
    for s in result.scores.values():
        assert sum(s.weighted_contributions.values()) == pytest.approx(s.snii, abs=1e-9)
        if s.snii > 0:
            assert sum(s.percent_contributions.values()) == pytest.approx(100.0, abs=1e-6)


def test_explain_snii_contains_key_fields():
    result = compute_snii(compute_importance(_path_graph()))
    text = explain_snii(result["N02"])
    assert "N02" in text and "rank 1 of 4" in text


def test_compute_snii_rejects_wrong_type():
    with pytest.raises(TypeError):
        compute_snii("not an importance result")  # type: ignore[arg-type]


def test_compute_snii_single_node_graph():
    g = nx.Graph()
    g.add_node("A", node_id="A", text="x", type="instruction")
    result = compute_snii(compute_importance(g))
    assert result["A"].rank == 1
    assert result["A"].total_nodes == 1


def test_compute_snii_all_same_importance_ties_by_node_id():
    g = nx.Graph()
    for nid in ("N03", "N01", "N02"):
        g.add_node(nid, node_id=nid, text="x", type="other")   # isolated, identical
    result = compute_snii(compute_importance(g))
    assert [s.node_id for s in result.ranked()] == ["N01", "N02", "N03"]


def test_compute_importance_rejects_empty_graph_before_snii():
    with pytest.raises(ImportanceError):
        compute_importance(nx.Graph())