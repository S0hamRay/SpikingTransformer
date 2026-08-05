"""Tests for knowledge-graph visualization helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from db.viz import (
    GraphSnapshot,
    build_plotly_figure,
    fetch_community_graph,
    fetch_knowledge_graph,
    format_graph_stats,
    render_graph_view,
)


def _session_with_rows(*row_batches):
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    # Each session.run(... ) returns an iterable / .single()
    run_results = []
    for batch in row_batches:
        result = MagicMock()
        if isinstance(batch, dict) and "single" in batch:
            result.single.return_value = batch["single"]
            result.__iter__ = MagicMock(return_value=iter(batch.get("rows", [])))
        else:
            result.__iter__ = MagicMock(return_value=iter(batch))
            result.single.return_value = {"n": 0}
        run_results.append(result)
    session.run.side_effect = run_results
    return session


def test_build_plotly_figure_empty_shows_hint() -> None:
    fig = build_plotly_figure(GraphSnapshot())
    assert fig.layout.title.text
    # annotation present for empty graph
    assert fig.layout.annotations


def test_build_plotly_figure_colors_by_kind() -> None:
    snapshot = GraphSnapshot(
        nodes=[
            {"id": "Paper:1", "label": "Demo Paper", "kind": "Paper", "community": None},
            {
                "id": "Concept:1",
                "label": "Leiden",
                "kind": "Concept",
                "community": 0,
            },
        ],
        edges=[
            {
                "source": "Paper:1",
                "target": "Concept:1",
                "rel": "USES_METHOD",
                "weight": 1.0,
            }
        ],
    )
    fig = build_plotly_figure(snapshot, color_by_community=False)
    assert len(fig.data) >= 2  # edges + node traces


def test_format_graph_stats() -> None:
    text = format_graph_stats(
        GraphSnapshot(
            nodes=[{"id": "a"}],
            edges=[],
            stats={"shown_nodes": 1, "shown_edges": 0, "communities": 0},
        )
    )
    assert "Shown" in text
    assert "Run Leiden" in text


def test_fetch_knowledge_graph_handles_db_error() -> None:
    store = MagicMock()
    store.connect.side_effect = RuntimeError("bolt down")
    snap = fetch_knowledge_graph(store=store)
    assert snap.error
    assert "Neo4j unavailable" in snap.error


def test_fetch_community_graph_parses_rows() -> None:
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    count_result = MagicMock()
    count_result.single.return_value = {"n": 2}

    co_rows = [
        {
            "a_name": "Attention",
            "a_id": "c1",
            "a_comm": 0,
            "b_name": "Transformer",
            "b_id": "c2",
            "b_comm": 0,
            "weight": 3.0,
        }
    ]
    paper_rows = [{"title": "Demo", "id": "p1", "comm": 0}]
    concept_rows = []

    co_result = MagicMock()
    co_result.__iter__ = MagicMock(return_value=iter(co_rows))
    paper_result = MagicMock()
    paper_result.__iter__ = MagicMock(return_value=iter(paper_rows))
    concept_result = MagicMock()
    concept_result.__iter__ = MagicMock(return_value=iter(concept_rows))

    session.run.side_effect = [count_result, co_result, paper_result, concept_result]

    store = MagicMock()
    store.connect.return_value.session.return_value = session
    snap = fetch_community_graph(store=store, max_nodes=20)
    assert snap.error is None
    assert len(snap.nodes) >= 2
    assert snap.stats["communities"] == 2


def test_render_graph_view_community_mode() -> None:
    snap = GraphSnapshot(
        nodes=[
            {
                "id": "Concept:1",
                "label": "A",
                "kind": "Concept",
                "community": 1,
            }
        ],
        edges=[],
        stats={"shown_nodes": 1, "shown_edges": 0, "communities": 1},
    )
    with patch("db.viz.fetch_community_graph", return_value=snap):
        fig, md = render_graph_view("Communities", max_nodes=40)
    assert fig.layout.title.text
    assert "Communities" in md or "Shown" in md
