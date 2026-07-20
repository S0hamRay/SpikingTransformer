"""Unit tests for Neo4j / Postgres academic graph helpers (mocked drivers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from db.config import Neo4jConfig, PostgresConfig
from db.neo4j_store import Neo4jGraphStore


def test_postgres_config_dsn() -> None:
    cfg = PostgresConfig(host="db", port=5433, user="u", password="p", database="d")
    assert cfg.dsn == "postgresql://u:p@db:5433/d"


def test_neo4j_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://neo:7687")
    monkeypatch.setenv("NEO4J_USER", "neo")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    cfg = Neo4jConfig.from_env()
    assert cfg.uri == "bolt://neo:7687"
    assert cfg.user == "neo"
    assert cfg.password == "secret"


def test_neo4j_upsert_paper_runs_cypher() -> None:
    store = Neo4jGraphStore(Neo4jConfig())
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value.single.return_value = {"id": "abc"}

    driver = MagicMock()
    driver.session.return_value = session

    with patch.object(store, "connect", return_value=driver):
        paper_id = store.upsert_paper(
            {"id": uuid4(), "title": "Attention Is All You Need", "year": 2017}
        )

    assert paper_id == "abc"
    session.run.assert_called()
    cypher = session.run.call_args.args[0]
    assert "MERGE (p:Paper" in cypher


def test_neo4j_link_reports_metric_sets_value() -> None:
    store = Neo4jGraphStore(Neo4jConfig())
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session

    with patch.object(store, "connect", return_value=driver):
        store.link_reports_metric("p1", "c1", value="92.1")

    cypher = session.run.call_args.args[0]
    assert "REPORTS_METRIC" in cypher
    assert session.run.call_args.kwargs["value"] == "92.1"


def test_schema_without_vector_strips_extension() -> None:
    from db.postgres_store import _schema_without_vector

    sql = 'CREATE EXTENSION IF NOT EXISTS "vector";\nembedding vector(768)'
    out = _schema_without_vector(sql)
    assert "vector(768)" not in out
    assert "embedding TEXT" in out
    assert 'CREATE EXTENSION IF NOT EXISTS "vector"' not in out
