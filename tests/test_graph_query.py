"""Tests for read-only Neo4j graph query helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from db.graph_query import (
    GraphQueryError,
    assert_read_only,
    execute_cypher,
    looks_like_cypher,
    run_graph_query,
)


def test_looks_like_cypher() -> None:
    assert looks_like_cypher("MATCH (p:Paper) RETURN p.title")
    assert looks_like_cypher("  optional match (a:Author) return a")
    assert not looks_like_cypher("Which papers use Leiden clustering?")


def test_assert_read_only_allows_match() -> None:
    q = assert_read_only("MATCH (c:Concept) RETURN c.name LIMIT 5")
    assert q.startswith("MATCH")


def test_assert_read_only_rejects_writes() -> None:
    with pytest.raises(GraphQueryError, match="read-only"):
        assert_read_only("MATCH (p:Paper) SET p.title = 'x' RETURN p")
    with pytest.raises(GraphQueryError, match="read-only"):
        assert_read_only("CREATE (p:Paper {title: 'x'})")
    with pytest.raises(GraphQueryError, match="read-only"):
        assert_read_only("CALL gds.leiden.write('g', {})")


def test_assert_read_only_strips_fences() -> None:
    q = assert_read_only("```cypher\nMATCH (n) RETURN n\n```")
    assert q == "MATCH (n) RETURN n"


def test_execute_cypher_returns_rows() -> None:
    store = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    record = MagicMock()
    record.keys.return_value = ["title"]
    record.__getitem__ = lambda self, k: "Attention Is All You Need"
    session.run.return_value = [record]

    driver = MagicMock()
    driver.session.return_value = session
    store.connect.return_value = driver

    rows = execute_cypher(
        "MATCH (p:Paper) RETURN p.title AS title",
        store=store,
    )
    assert rows == [{"title": "Attention Is All You Need"}]


def test_run_graph_query_cypher_path() -> None:
    with patch("db.graph_query.execute_cypher", return_value=[{"n": 1}]) as exec_mock:
        out = run_graph_query("MATCH (p:Paper) RETURN count(p) AS n")
    assert out["source"] == "cypher"
    assert out["results"] == [{"n": 1}]
    exec_mock.assert_called_once()


def test_run_graph_query_natural_language_path() -> None:
    with (
        patch(
            "db.graph_query.natural_language_to_cypher",
            return_value="MATCH (c:Concept) RETURN c.name LIMIT 5",
        ),
        patch("db.graph_query.execute_cypher", return_value=[{"c.name": "Leiden"}]),
    ):
        out = run_graph_query("list concept names")
    assert out["source"] == "natural_language"
    assert out["query"].startswith("MATCH")
    assert out["results"][0]["c.name"] == "Leiden"


def test_run_graph_query_empty() -> None:
    with pytest.raises(GraphQueryError, match="Provide"):
        run_graph_query("   ")
