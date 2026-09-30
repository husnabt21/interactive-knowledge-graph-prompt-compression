"""Interactive semantic-graph figure for the Day 3 UI (Step 1).

Presentation only: reads the annotated Day 1 graph and a Day 2
CompressionResult and draws them. It never recalculates importance or
compression decisions.

Visual conventions
------------------
green  circle/diamond : retained
red                   : removed (budget / lower priority)
orange                : removed as a near-duplicate of another node
diamond               : strongly protected unit type
marker size           : scales with Day 1 importance
solid grey edge       : semantic_similarity edge (hover shows weight)
dashed orange edge    : near-duplicate pair
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Mapping, Optional

import networkx as nx
import plotly.graph_objects as go

from src.compression.compresser import CompressionResult, Decision, NodeDecision


class GraphRenderError(ValueError):
    """The graph/result cannot be rendered (empty graph, mismatched ids, ...)."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
STATUS_RETAINED = "retained"
STATUS_REMOVED = "removed"
STATUS_DUPLICATE = "duplicate_removed"
STATUS_ORDER = (STATUS_RETAINED, STATUS_REMOVED, STATUS_DUPLICATE)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    STATUS_RETAINED: ("Retained", "#2e7d32"),
    STATUS_REMOVED: ("Removed", "#c62828"),
    STATUS_DUPLICATE: ("Removed (near-duplicate)", "#ef6c00"),
}

UNIT_TYPE_ORDER = (
    "role", "instruction", "context", "constraint",
    "requirement", "example", "output_format", "other",
)
FILTER_ALL = "All nodes"
FILTER_RETAINED = "Retained only"
FILTER_REMOVED = "Removed only"
FILTER_PROTECTED = "Protected only"
FILTER_OPTIONS = (FILTER_ALL, FILTER_RETAINED, FILTER_REMOVED, FILTER_PROTECTED, *UNIT_TYPE_ORDER)

DIM_ALPHA = 0.30
EDGE_COLOR = "rgba(120,120,120,0.55)"
DUP_EDGE_COLOR = "rgba(239,108,0,0.85)"

# Pass to st.plotly_chart(..., config=PLOTLY_CONFIG).
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def node_status(decision: NodeDecision) -> str:
    """Map a NodeDecision to one of the three display statuses."""
    if decision.decision is Decision.RETAIN:
        return STATUS_RETAINED
    return STATUS_DUPLICATE if decision.duplicate_of is not None else STATUS_REMOVED


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _shorten(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _node_size(importance: float) -> float:
    return 16.0 + 26.0 * max(0.0, min(1.0, float(importance)))


def _hover(graph: nx.Graph, d: NodeDecision) -> str:
    lines = [
        f"<b>{html.escape(d.node_id)}</b> · {html.escape(d.unit_type)}",
        f"Decision: {d.decision.value}",
        f"Importance: {d.importance:.3f} · Priority: {d.priority:.3f}",
    ]
    data = graph.nodes[d.node_id]
    if "connectivity" in data and "centrality" in data:
        lines.append(
            f"Connectivity: {float(data['connectivity']):.3f} · "
            f"Centrality: {float(data['centrality']):.3f}"
        )
    if d.protected:
        lines.append("Protected unit type")
    if d.duplicate_of is not None:
        sim = "" if d.duplicate_similarity is None else f" (similarity {d.duplicate_similarity:.3f})"
        lines.append(f"Near-duplicate of {html.escape(d.duplicate_of)}{sim}")
    lines.append(html.escape(_shorten(d.text)))
    return "<br>".join(lines)


def _validate_inputs(graph: nx.Graph, result: CompressionResult) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"graph must be a networkx.Graph, got {type(graph).__name__}")
    if not isinstance(result, CompressionResult):
        raise TypeError(f"result must be a CompressionResult, got {type(result).__name__}")
    if graph.number_of_nodes() == 0:
        raise GraphRenderError("cannot render an empty graph")
    if {d.node_id for d in result.decisions} != set(graph.nodes):
        raise GraphRenderError(
            "graph nodes and compression decisions do not match; compress the same graph you render"
        )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def compute_layout(graph: nx.Graph, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Deterministic layout that stops isolated nodes from crushing the cluster.

    Connected nodes get a seeded spring layout (higher edge weight = closer).
    Isolated nodes (no edges) would otherwise be pushed to the far edge of the
    canvas and squash everything else, so they are placed in an evenly spaced
    row below the connected cluster, in prompt order. If no node has an edge,
    all nodes are placed on a circle.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"graph must be a networkx.Graph, got {type(graph).__name__}")
    if graph.number_of_nodes() == 0:
        raise GraphRenderError("cannot lay out an empty graph")

    connected = [n for n in graph.nodes if graph.degree(n) > 0]
    isolated = sorted(
        (n for n in graph.nodes if graph.degree(n) == 0),
        key=lambda n: (graph.nodes[n].get("index", 0), str(n)),
    )
    if not connected:
        circle = nx.circular_layout(graph)
        return {n: (float(p[0]), float(p[1])) for n, p in circle.items()}

    spring = nx.spring_layout(graph.subgraph(connected), weight="weight", seed=seed)
    pos = {n: (float(p[0]), float(p[1])) for n, p in spring.items()}
    if isolated:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        centre = (min(xs) + max(xs)) / 2
        span = max(max(xs) - min(xs), 1.0)
        y = min(ys) - 0.6
        k = len(isolated)
        for i, n in enumerate(isolated):
            x = centre if k == 1 else centre - span / 2 + span * i / (k - 1)
            pos[n] = (x, y)
    return pos


def _resolve_positions(
    graph: nx.Graph, positions: Optional[Mapping[str, tuple[float, float]]]
) -> Mapping[str, tuple[float, float]]:
    if positions is None:
        return compute_layout(graph)
    missing = set(graph.nodes) - set(positions)
    if missing:
        raise GraphRenderError(f"positions missing for nodes: {sorted(missing)}")
    return positions


# --------------------------------------------------------------------------
# Filters (UI only; never change the CompressionResult)
# --------------------------------------------------------------------------
def filter_node_ids(result: CompressionResult, mode: str) -> set[str]:
    """Return node ids matching a UI filter mode from FILTER_OPTIONS."""
    if not isinstance(result, CompressionResult):
        raise TypeError(f"result must be a CompressionResult, got {type(result).__name__}")
    decisions = result.decisions
    if mode == FILTER_ALL:
        return {d.node_id for d in decisions}
    if mode == FILTER_RETAINED:
        return {d.node_id for d in decisions if d.decision is Decision.RETAIN}
    if mode == FILTER_REMOVED:
        return {d.node_id for d in decisions if d.decision is Decision.REMOVE}
    if mode == FILTER_PROTECTED:
        return {d.node_id for d in decisions if d.protected}
    if mode in UNIT_TYPE_ORDER:
        return {d.node_id for d in decisions if d.unit_type == mode}
    raise ValueError(f"unknown filter mode {mode!r}; expected one of {FILTER_OPTIONS}")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _edge_traces(graph: nx.Graph, pos: Mapping[str, tuple[float, float]]) -> list[go.Scatter]:
    reg_x: list[Optional[float]] = []
    reg_y: list[Optional[float]] = []
    dup_x: list[Optional[float]] = []
    dup_y: list[Optional[float]] = []
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_text: list[str] = []

    for a, b, data in graph.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        is_dup = bool(data.get("near_duplicate"))
        xs, ys = (dup_x, dup_y) if is_dup else (reg_x, reg_y)
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        weight = data.get("weight")
        w_txt = "n/a" if weight is None else f"{float(weight):.3f}"
        tag = " (near-duplicate)" if is_dup else ""
        mid_text.append(f"{html.escape(str(a))} — {html.escape(str(b))}<br>similarity {w_txt}{tag}")

    traces: list[go.Scatter] = []
    if reg_x:
        traces.append(go.Scatter(
            x=reg_x, y=reg_y, mode="lines", name="Semantic edge", hoverinfo="skip",
            line=dict(width=1.5, color=EDGE_COLOR),
        ))
    if dup_x:
        traces.append(go.Scatter(
            x=dup_x, y=dup_y, mode="lines", name="Near-duplicate edge", hoverinfo="skip",
            line=dict(width=2.5, color=DUP_EDGE_COLOR, dash="dash"),
        ))
    if mid_x:
        traces.append(go.Scatter(
            x=mid_x, y=mid_y, mode="markers", name="Edge weight", showlegend=False,
            hoverinfo="text", hovertext=mid_text, marker=dict(size=10, opacity=0),
        ))
    return traces


def build_figure(
    graph: nx.Graph,
    result: CompressionResult,
    *,
    positions: Optional[Mapping[str, tuple[float, float]]] = None,
    selected_id: Optional[str] = None,
    visible_ids: Optional[Iterable[str]] = None,
    height: int = 620,
) -> go.Figure:
    """Build the interactive graph figure.

    Args:
        graph: annotated Day 1 graph.
        result: Day 2 CompressionResult for the same graph.
        positions: node -> (x, y); computed with compute_layout() if omitted.
        selected_id: node to ring-highlight.
        visible_ids: if given, other nodes are dimmed (filters/search).
        height: figure height in pixels.
    """
    _validate_inputs(graph, result)
    pos = _resolve_positions(graph, positions)
    if selected_id is not None and selected_id not in graph:
        raise GraphRenderError(f"selected node {selected_id!r} is not in the graph")
    visible = None if visible_ids is None else set(visible_ids)
    if visible is not None and not visible <= set(graph.nodes):
        raise GraphRenderError(f"visible_ids has unknown nodes: {sorted(visible - set(graph.nodes))}")

    fig = go.Figure()
    for trace in _edge_traces(graph, pos):
        fig.add_trace(trace)

    by_status: dict[str, list[NodeDecision]] = {s: [] for s in STATUS_ORDER}
    for d in result.decisions:
        by_status[node_status(d)].append(d)

    for status in STATUS_ORDER:
        group = by_status[status]
        if not group:
            continue
        label, color = STATUS_STYLE[status]
        alphas = [1.0 if visible is None or d.node_id in visible else DIM_ALPHA for d in group]
        fig.add_trace(go.Scatter(
            x=[pos[d.node_id][0] for d in group],
            y=[pos[d.node_id][1] for d in group],
            mode="markers+text",
            name=label,
            text=[d.node_id for d in group],
            textposition="top center",
            textfont=dict(size=12),
            customdata=[d.node_id for d in group],
            hoverinfo="text",
            hovertext=[_hover(graph, d) for d in group],
            marker=dict(
                size=[_node_size(d.importance) for d in group],
                symbol=["diamond" if d.protected else "circle" for d in group],
                color=[_rgba(color, a) for a in alphas],
                line=dict(
                    width=[2.5 if d.protected else 1 for d in group],
                    color=[_rgba("#263238" if d.protected else "#ffffff", a) for a in alphas],
                ),
            ),
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
        ))

    if selected_id is not None:
        sel = next(d for d in result.decisions if d.node_id == selected_id)
        fig.add_trace(go.Scatter(
            x=[pos[selected_id][0]], y=[pos[selected_id][1]], mode="markers",
            name="Selected", showlegend=False, hoverinfo="skip",
            marker=dict(size=_node_size(sel.importance) + 16, color="rgba(0,0,0,0)",
                        line=dict(width=3, color="#1565c0")),
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        hovermode="closest",
        dragmode="pan",
        clickmode="event+select",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


# --------------------------------------------------------------------------
# Click -> node id
# --------------------------------------------------------------------------
def selected_node_ids(selection: Any, fig: go.Figure) -> list[str]:
    """Map a Streamlit plotly selection to node ids.

    Pass ``event.selection`` (or the whole event) from
    ``st.plotly_chart(fig, on_select="rerun")``. Points on non-node traces
    (edges, highlight ring) are ignored.
    """
    if selection is None:
        return []
    try:
        if "selection" in selection:
            selection = selection["selection"]
        points = selection["points"]
    except (KeyError, TypeError):
        return []

    ids: list[str] = []
    for p in points:
        curve = p.get("curve_number")
        idx = p.get("point_index", p.get("point_number"))
        if not isinstance(curve, int) or not isinstance(idx, int):
            continue
        if not 0 <= curve < len(fig.data):
            continue
        custom = fig.data[curve].customdata
        if custom is None or not 0 <= idx < len(custom):
            continue
        ids.append(str(custom[idx]))
    return ids


# --------------------------------------------------------------------------
# Manual check:  python -m src.visualization.graph_render
# Hand-made graph, no models needed. Writes an interactive preview you can
# open in a browser to check zoom / pan / hover.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from src.compression.compresser import CompressionConfig, compress_graph

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

    result = compress_graph(g, CompressionConfig(retention_ratio=0.60))
    figure = build_figure(g, result)

    print("Traces:", [t.name for t in figure.data])
    for t in figure.data:
        if t.customdata is not None:
            ids = list(t.customdata)
            print(f"{t.name} ({len(ids)}): {ids}")

    retained_curve = next(i for i, t in enumerate(figure.data) if t.name == "Retained")
    retained_ids = list(figure.data[retained_curve].customdata)
    click = {"points": [{"curve_number": retained_curve, "point_index": retained_ids.index("N05")}]}
    print("Selected via fake click:", selected_node_ids(click, figure))

    with_ring = build_figure(g, result, selected_id="N05")
    print("Extra traces when a node is selected:", len(with_ring.data) - len(figure.data))
    print("Filter 'Removed only':", sorted(filter_node_ids(result, FILTER_REMOVED)))
    print("Filter 'Protected only':", sorted(filter_node_ids(result, FILTER_PROTECTED)))

    out = Path("data/graph_preview.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    with_ring.write_html(out, include_plotlyjs=True)
    print(f"Wrote {out}")