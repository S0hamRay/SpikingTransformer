"""Service wrappers around ChatEngine, RAGEngine, and graph helpers."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from db.graph_query import GraphQueryError, run_graph_query
from db.leiden import LeidenError, LeidenStats, run_leiden

if TYPE_CHECKING:
    from chatbot.engine import ChatEngine
    from rag.engine import RAGEngine

CHECKPOINT_ROOT = "checkpoints"
DEFAULT_RAG_SESSION = "default"


@lru_cache(maxsize=4)
def get_chat_engine(attn_type: str) -> "ChatEngine":
    """Load and cache a character-level chat engine for an attention variant."""
    from chatbot.engine import ChatEngine, default_checkpoint_for

    path = default_checkpoint_for(attn_type, CHECKPOINT_ROOT)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train it first:\n"
            f"  python train.py --config configs/shakespeare.yaml "
            f"--attn-type {attn_type} --checkpoint-dir {path.parent}"
        )
    return ChatEngine.from_checkpoint(path)


@lru_cache(maxsize=16)
def get_rag_engine(session_id: str) -> "RAGEngine":
    """Load and cache a session-scoped RAG engine."""
    from rag.engine import RAGEngine

    return RAGEngine.from_session(session_id or DEFAULT_RAG_SESSION)


class ChatService:
    """Thin facade over :class:`ChatEngine`."""

    def reply(self, message: str, attn_type: str = "spiking") -> str:
        return get_chat_engine(attn_type).generate_reply(message)

    def reset(self, attn_type: str = "spiking") -> None:
        try:
            get_chat_engine(attn_type).reset()
        except FileNotFoundError:
            pass


class RAGService:
    """Thin facade over :class:`RAGEngine` for corrective RAG."""

    def query(self, question: str, session_id: str = DEFAULT_RAG_SESSION) -> str:
        return get_rag_engine(session_id).generate_reply(question)

    def ingest(self, file_path: str, session_id: str = DEFAULT_RAG_SESSION) -> str:
        path = Path(file_path).expanduser()
        if not path.is_file():
            return f"File not found: {path}"
        return get_rag_engine(session_id).ingest_files([path])


class GraphService:
    """Paper/Author/Concept graph traversal and Leiden community detection."""

    def query(self, cypher_or_natural_language: str) -> dict[str, Any]:
        return run_graph_query(cypher_or_natural_language)

    def run_community_detection(self, gamma: float | None = None) -> dict[str, Any]:
        stats: LeidenStats = run_leiden(gamma=gamma)
        payload = asdict(stats)
        payload["summary"] = stats.format_summary()
        return payload


chat_service = ChatService()
rag_service = RAGService()
graph_service = GraphService()

__all__ = [
    "ChatService",
    "GraphQueryError",
    "GraphService",
    "LeidenError",
    "RAGService",
    "chat_service",
    "get_chat_engine",
    "get_rag_engine",
    "graph_service",
    "rag_service",
]
