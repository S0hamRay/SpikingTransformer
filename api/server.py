"""FastAPI application: engines as HTTP, Gradio UI sub-app, MCP tools.

Run with:
    python -m api.server
    # or
    uvicorn api.server:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    POST /chat/reply
    POST /chat/reset
    POST /rag/query                 (MCP: rag_query)
    POST /rag/ingest                (MCP: rag_ingest)
    POST /graph/query               (MCP: graph_query)
    POST /graph/community-detection (MCP: run_community_detection)
    /                               Gradio chatbot
    /mcp                            auto-generated MCP server (fastapi-mcp)
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from api.schemas import (
    ChatReplyRequest,
    ChatReplyResponse,
    ChatResetRequest,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthResponse,
    RAGIngestRequest,
    RAGIngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from api.services import (
    GraphQueryError,
    LeidenError,
    chat_service,
    graph_service,
    rag_service,
)

MCP_OPERATIONS = [
    "rag_query",
    "rag_ingest",
    "graph_query",
    "run_community_detection",
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv()
    yield


def create_app(*, mount_gradio: bool = True, mount_mcp: bool = True) -> FastAPI:
    """Build the FastAPI app with optional Gradio and MCP mounts."""
    app = FastAPI(
        title="SpikingTransformer",
        description=(
            "Chat (spiking/standard), corrective RAG, and academic Graph RAG "
            "over Paper/Author/Concept. MCP tools are auto-generated from the "
            "RAG/graph routes via fastapi-mcp."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/chat/reply", response_model=ChatReplyResponse, tags=["chat"])
    def chat_reply(body: ChatReplyRequest) -> ChatReplyResponse:
        try:
            reply = chat_service.reply(body.message, body.attn_type)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ChatReplyResponse(reply=reply, attn_type=body.attn_type)

    @app.post("/chat/reset", tags=["chat"])
    def chat_reset(body: ChatResetRequest) -> dict[str, str]:
        chat_service.reset(body.attn_type)
        return {"status": "ok", "attn_type": body.attn_type}

    @app.post(
        "/rag/query",
        response_model=RAGQueryResponse,
        tags=["rag"],
        operation_id="rag_query",
        summary="Corrective RAG answer",
        description=(
            "Run the LangGraph corrective RAG pipeline (retrieve → grade → "
            "optional web search → generate) and return an answer grounded in "
            "indexed papers for the given session."
        ),
    )
    def rag_query(body: RAGQueryRequest) -> RAGQueryResponse:
        answer = rag_service.query(body.question, body.session_id)
        return RAGQueryResponse(answer=answer, session_id=body.session_id)

    @app.post(
        "/rag/ingest",
        response_model=RAGIngestResponse,
        tags=["rag"],
        operation_id="rag_ingest",
        summary="Index a new paper",
        description=(
            "Index a .txt or .pdf paper into the session Chroma store and sync "
            "Paper/Author/Concept entities into Postgres + Neo4j when enabled."
        ),
    )
    def rag_ingest(body: RAGIngestRequest) -> RAGIngestResponse:
        status = rag_service.ingest(body.file_path, body.session_id)
        return RAGIngestResponse(
            status=status,
            session_id=body.session_id,
            file_path=body.file_path,
        )

    @app.post(
        "/graph/query",
        response_model=GraphQueryResponse,
        tags=["graph"],
        operation_id="graph_query",
        summary="Traverse the Paper/Author/Concept graph",
        description=(
            "Run a read-only Cypher query, or a natural-language question that "
            "is translated to Cypher via Ollama, against the Neo4j academic graph."
        ),
    )
    def graph_query(body: GraphQueryRequest) -> GraphQueryResponse:
        try:
            result = graph_service.query(body.cypher_or_natural_language)
        except GraphQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface Neo4j connectivity issues
            raise HTTPException(
                status_code=503,
                detail=f"Graph query failed: {exc}",
            ) from exc
        return GraphQueryResponse(**result)

    @app.post(
        "/graph/community-detection",
        response_model=CommunityDetectionResponse,
        tags=["graph"],
        operation_id="run_community_detection",
        summary="Run Leiden community detection",
        description=(
            "Trigger Neo4j GDS Leiden on Concept co-occurrence and return "
            "modularity plus community size statistics."
        ),
    )
    def run_community_detection(
        body: CommunityDetectionRequest,
    ) -> CommunityDetectionResponse:
        try:
            stats = graph_service.run_community_detection(body.gamma)
        except LeidenError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Community detection failed: {exc}",
            ) from exc
        return CommunityDetectionResponse(**stats)

    if mount_mcp:
        _mount_mcp(app)

    if mount_gradio:
        _mount_gradio(app)

    return app


def _mount_mcp(app: FastAPI) -> None:
    """Auto-generate an MCP server from the FastAPI RAG/graph routes."""
    try:
        from fastapi_mcp import FastApiMCP
    except ImportError as exc:  # pragma: no cover - optional until deps installed
        raise ImportError(
            "fastapi-mcp is required for MCP mounting. "
            "Install with: pip install fastapi-mcp"
        ) from exc

    try:
        mcp = FastApiMCP(
            app,
            name="SpikingTransformer Graph RAG",
            description=(
                "MCP tools for corrective RAG, paper ingest, Neo4j graph "
                "traversal, and Leiden community detection."
            ),
            include_operations=list(MCP_OPERATIONS),
        )
    except TypeError as exc:
        raise ImportError(
            "fastapi-mcp is incompatible with this mcp package version. "
            "Install pinned deps: pip install 'fastapi-mcp>=0.3,<0.5' 'mcp>=1.6,<2'"
        ) from exc

    # fastapi-mcp >=0.3 prefers mount_http(); older versions use mount().
    if hasattr(mcp, "mount_http"):
        mcp.mount_http()
    else:
        mcp.mount()


def _mount_gradio(app: FastAPI) -> None:
    """Mount the Gradio chatbot UI at the site root."""
    import gradio as gr

    from app import build_demo

    demo = build_demo()
    # Mount last so API/MCP routes keep priority; browser hits to ``/`` get the UI.
    gr.mount_gradio_app(app, demo, path="/")


_app: FastAPI | None = None


def __getattr__(name: str):
    """Lazy ``app`` so ``from api.server import create_app`` skips Gradio/MCP mounts."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """CLI entry: serve FastAPI + Gradio + MCP on one port."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="SpikingTransformer API (engines + Gradio UI + MCP)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-gradio",
        action="store_true",
        help="Skip mounting the Gradio UI at /.",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip mounting the auto-generated MCP server.",
    )
    args = parser.parse_args()

    import uvicorn

    application = create_app(
        mount_gradio=not args.no_gradio,
        mount_mcp=not args.no_mcp,
    )
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
