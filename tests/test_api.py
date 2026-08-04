"""Tests for the FastAPI service layer and MCP operation ids."""

from __future__ import annotations

from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.server import MCP_OPERATIONS, create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    application = create_app(mount_gradio=False, mount_mcp=False)
    return TestClient(application)


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_rag_query_operation_id_and_body(client: TestClient) -> None:
    with patch("api.server.rag_service.query", return_value="Spiking uses addition.") as mock:
        res = client.post(
            "/rag/query",
            json={"question": "What is spiking attention?"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Spiking uses addition."
    assert body["session_id"] == "default"
    mock.assert_called_once_with("What is spiking attention?", "default")


def test_rag_ingest_operation_id(client: TestClient) -> None:
    with patch(
        "api.server.rag_service.ingest",
        return_value="Indexed 1 file(s) (3 chunks): paper.pdf",
    ) as mock:
        res = client.post(
            "/rag/ingest",
            json={"file_path": "/tmp/paper.pdf", "session_id": "s1"},
        )
    assert res.status_code == 200
    assert "Indexed" in res.json()["status"]
    mock.assert_called_once_with("/tmp/paper.pdf", "s1")


def test_graph_query_endpoint(client: TestClient) -> None:
    payload = {
        "query": "MATCH (p:Paper) RETURN p.title AS title",
        "source": "cypher",
        "results": [{"title": "Demo"}],
    }
    with patch("api.server.graph_service.query", return_value=payload):
        res = client.post(
            "/graph/query",
            json={
                "cypher_or_natural_language": "MATCH (p:Paper) RETURN p.title AS title"
            },
        )
    assert res.status_code == 200
    assert res.json()["results"][0]["title"] == "Demo"


def test_graph_query_bad_request(client: TestClient) -> None:
    from db.graph_query import GraphQueryError

    with patch(
        "api.server.graph_service.query",
        side_effect=GraphQueryError("Only read-only Cypher is allowed"),
    ):
        res = client.post(
            "/graph/query",
            json={"cypher_or_natural_language": "CREATE (n:X)"},
        )
    assert res.status_code == 400


def test_run_community_detection_endpoint(client: TestClient) -> None:
    stats = {
        "community_count": 3,
        "modularity": 0.42,
        "size_min": 2,
        "size_median": 4.0,
        "size_max": 10,
        "gamma": 1.5,
        "concept_count": 20,
        "co_occur_edges": 30,
        "papers_assigned": 5,
        "summary": "Leiden communities: 3",
    }
    with patch(
        "api.server.graph_service.run_community_detection",
        return_value=stats,
    ) as mock:
        res = client.post("/graph/community-detection", json={"gamma": 1.5})
    assert res.status_code == 200
    assert res.json()["modularity"] == 0.42
    mock.assert_called_once_with(1.5)


def test_openapi_exposes_mcp_operation_ids(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    op_ids = {
        path_item[method].get("operationId")
        for path_item in schema["paths"].values()
        for method in path_item
        if method in {"get", "post", "put", "delete", "patch"}
    }
    assert set(MCP_OPERATIONS).issubset(op_ids)


def test_chat_reply_missing_checkpoint(client: TestClient) -> None:
    with patch(
        "api.server.chat_service.reply",
        side_effect=FileNotFoundError("No checkpoint"),
    ):
        res = client.post(
            "/chat/reply",
            json={"message": "hello", "attn_type": "spiking"},
        )
    assert res.status_code == 404


def test_mcp_mount_exposes_path() -> None:
    pytest.importorskip("fastapi_mcp")
    application = create_app(mount_gradio=False, mount_mcp=True)
    paths = {getattr(route, "path", None) for route in application.routes}
    assert "/mcp" in paths
