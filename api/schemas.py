"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatReplyRequest(BaseModel):
    message: str = Field(..., description="User message to the character-level LM.")
    attn_type: Literal["spiking", "standard"] = Field(
        "spiking", description="Attention variant checkpoint to use."
    )


class ChatReplyResponse(BaseModel):
    reply: str
    attn_type: str


class ChatResetRequest(BaseModel):
    attn_type: Literal["spiking", "standard"] = "spiking"


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Question for the corrective RAG pipeline.")
    session_id: str = Field(
        "default",
        description="RAG session id (vector store + history under rag_data/<id>/).",
    )


class RAGQueryResponse(BaseModel):
    answer: str
    session_id: str


class RAGIngestRequest(BaseModel):
    file_path: str = Field(
        ..., description="Filesystem path to a .txt or .pdf paper to index."
    )
    session_id: str = Field(
        "default",
        description="RAG session id that will own the indexed vectors.",
    )


class RAGIngestResponse(BaseModel):
    status: str
    session_id: str
    file_path: str


class GraphQueryRequest(BaseModel):
    cypher_or_natural_language: str = Field(
        ...,
        description=(
            "Read-only Cypher, or a natural-language question about the "
            "Paper/Author/Concept graph (translated to Cypher via Ollama)."
        ),
    )


class GraphQueryResponse(BaseModel):
    query: str = Field(..., description="Cypher that was executed.")
    source: Literal["cypher", "natural_language"]
    results: list[dict[str, Any]]


class CommunityDetectionRequest(BaseModel):
    gamma: float | None = Field(
        None,
        description=(
            "Leiden resolution parameter (higher → more communities). "
            "Defaults to LEIDEN_GAMMA env / 1.0 when omitted."
        ),
    )


class CommunityDetectionResponse(BaseModel):
    community_count: int
    modularity: float
    size_min: int
    size_median: float
    size_max: int
    gamma: float
    concept_count: int
    co_occur_edges: int
    papers_assigned: int
    summary: str


class HealthResponse(BaseModel):
    status: str = "ok"
