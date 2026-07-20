"""Tests for Leiden community detection pipeline (mocked Neo4j / GDS)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from db.config import leiden_gamma
from db.leiden import (
    GRAPH_NAME,
    LeidenError,
    LeidenStats,
    cascade_papers_to_communities,
    cleanup_previous_clustering,
    drop_graph_if_exists,
    materialize_communities,
    project_concept_graph,
    rebuild_co_occurrence,
    run_leiden,
    run_leiden_write,
)


def test_leiden_gamma_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEIDEN_GAMMA", raising=False)
    assert leiden_gamma() == 1.0


def test_leiden_gamma_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEIDEN_GAMMA", "1.5")
    assert leiden_gamma() == 1.5


def test_stats_format_summary() -> None:
    stats = LeidenStats(
        community_count=3,
        modularity=0.431,
        size_min=2,
        size_median=8.0,
        size_max=47,
        gamma=1.0,
        concept_count=20,
        co_occur_edges=40,
        papers_assigned=5,
    )
    text = stats.format_summary()
    assert "Leiden communities: 3" in text
    assert "0.431" in text
    assert "min: 2" in text
    assert "median: 8" in text
    assert "max: 47" in text
    assert "gamma: 1.0" in text


def test_rebuild_co_occurrence_deletes_then_merges() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {"n": 4}

    n = rebuild_co_occurrence(session)
    assert n == 4
    assert session.run.call_count == 4
    first = session.run.call_args_list[0].args[0]
    assert "CO_OCCURS_WITH" in first and "DELETE" in first
    assert "USES_METHOD" in session.run.call_args_list[1].args[0]
    assert "MENTIONS" in session.run.call_args_list[2].args[0]


def test_cleanup_previous_clustering() -> None:
    session = MagicMock()
    cleanup_previous_clustering(session)
    queries = [c.args[0] for c in session.run.call_args_list]
    assert any("Community" in q and "DETACH DELETE" in q for q in queries)
    assert any("BELONGS_TO" in q for q in queries)
    assert any("leidenCommunity" in q for q in queries)


def test_drop_graph_if_exists_when_present() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {"exists": True}
    drop_graph_if_exists(session)
    assert session.run.call_count == 2
    assert GRAPH_NAME in session.run.call_args_list[0].kwargs.get(
        "name", session.run.call_args_list[0].args[1:]
    ) or session.run.call_args_list[0].kwargs.get("name") == GRAPH_NAME


def test_drop_graph_if_exists_when_absent() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {"exists": False}
    drop_graph_if_exists(session)
    assert session.run.call_count == 1


def test_run_leiden_write_passes_gamma() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {
        "communityCount": 2,
        "modularity": 0.5,
        "modularities": [0.5],
        "nodePropertiesWritten": 10,
    }
    out = run_leiden_write(session, gamma=1.25)
    assert out["communityCount"] == 2
    kwargs = session.run.call_args.kwargs
    assert kwargs["gamma"] == 1.25
    assert kwargs["name"] == GRAPH_NAME
    assert "gds.leiden.write" in session.run.call_args.args[0]


def test_project_concept_graph() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {
        "graphName": GRAPH_NAME,
        "nodeCount": 5,
        "relationshipCount": 8,
    }
    out = project_concept_graph(session)
    assert out["nodeCount"] == 5
    assert "gds.graph.project" in session.run.call_args.args[0]


def test_materialize_and_cascade() -> None:
    session = MagicMock()
    session.run.return_value.single.return_value = {"n": 3}
    assert materialize_communities(session) == 3
    assert "Community" in session.run.call_args.args[0]
    assert "BELONGS_TO" in session.run.call_args.args[0]

    session.run.return_value.single.return_value = {"n": 2}
    assert cascade_papers_to_communities(session) == 2
    assert "majority" not in session.run.call_args.args[0].lower()
    assert "BELONGS_TO" in session.run.call_args.args[0]


def _mock_store_for_pipeline(
    *,
    concept_count: int = 5,
    edge_count: int = 4,
    community_count: int = 2,
    modularity: float = 0.4,
    papers: int = 1,
) -> MagicMock:
    store = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session
    store.connect.return_value = driver

    # Sequence of session.run().single() / list results used by run_leiden
    call_results: list = []

    def _run(query: str, **kwargs):  # noqa: ANN001
        result = MagicMock()
        q = query.strip()
        call_results.append(q)

        if "count(c) AS n" in q and "Concept" in q:
            result.single.return_value = {"n": concept_count}
        elif "count(r) AS n" in q and "CO_OCCURS_WITH" in q:
            result.single.return_value = {"n": edge_count}
        elif "gds.graph.exists" in q:
            result.single.return_value = {"exists": False}
        elif "gds.graph.project" in q:
            result.single.return_value = {
                "graphName": GRAPH_NAME,
                "nodeCount": concept_count,
                "relationshipCount": edge_count,
            }
        elif "gds.leiden.write" in q:
            result.single.return_value = {
                "communityCount": community_count,
                "modularity": modularity,
                "modularities": [modularity],
                "nodePropertiesWritten": concept_count,
            }
        elif "gds.graph.drop" in q:
            result.single.return_value = {"graphName": GRAPH_NAME}
        elif "count(DISTINCT comm)" in q or (
            "MERGE (comm:Community" in q and "RETURN count" in q
        ):
            result.single.return_value = {"n": community_count}
        elif "count(DISTINCT p) AS n" in q:
            result.single.return_value = {"n": papers}
        elif "comm.size AS size" in q:
            return iter([{"size": 2}, {"size": 4}, {"size": 6}])
        else:
            result.single.return_value = {"n": 0}
        return result

    session.run.side_effect = _run
    store._session_queries = call_results
    store._session = session
    return store


def test_run_leiden_pipeline_order_and_gamma() -> None:
    store = _mock_store_for_pipeline()

    with patch("db.leiden.print"):
        stats = run_leiden(gamma=1.7, store=store)

    assert stats.gamma == 1.7
    assert stats.community_count == 2
    assert abs(stats.modularity - 0.4) < 1e-9
    assert stats.co_occur_edges == 4

    queries = store._session_queries
    # Relative order of key steps
    def first_idx(pred) -> int:
        for i, q in enumerate(queries):
            if pred(q):
                return i
        raise AssertionError(f"query not found: {pred}")

    i_rebuild = first_idx(lambda q: "DELETE r" in q and "CO_OCCURS_WITH" in q)
    i_cleanup = first_idx(lambda q: "DETACH DELETE" in q and "Community" in q)
    i_project = first_idx(lambda q: "gds.graph.project" in q)
    i_leiden = first_idx(lambda q: "gds.leiden.write" in q)
    i_materialize = first_idx(lambda q: "MERGE (comm:Community" in q)
    i_cascade = first_idx(lambda q: "count(DISTINCT p) AS n" in q)

    assert i_rebuild < i_cleanup < i_project < i_leiden < i_materialize < i_cascade

    # gamma passed to leiden write
    leiden_calls = [
        c
        for c in store._session.run.call_args_list
        if c.args and "gds.leiden.write" in c.args[0]
    ]
    assert leiden_calls
    assert leiden_calls[0].kwargs["gamma"] == 1.7

    store.init_schema.assert_called()


def test_run_leiden_raises_when_too_few_concepts() -> None:
    store = _mock_store_for_pipeline(concept_count=1)
    with pytest.raises(LeidenError, match="at least 2 Concept"):
        run_leiden(gamma=1.0, store=store)


def test_run_leiden_raises_when_no_cooccurrence() -> None:
    store = _mock_store_for_pipeline(concept_count=5, edge_count=0)
    with pytest.raises(LeidenError, match="CO_OCCURS_WITH"):
        run_leiden(gamma=1.0, store=store)
