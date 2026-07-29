"""RAG engine facade: ingest documents and answer questions via LangGraph."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag.graph import build_rag_graph, initial_rag_state
from rag.ingest import SUPPORTED_SUFFIXES, ingest_files
from rag.store import VectorStore

DEFAULT_RAG_DATA_ROOT = "rag_data"


def default_rag_dir(session_id: str, root: str | Path = DEFAULT_RAG_DATA_ROOT) -> Path:
    """Return the persistence directory for a RAG session."""
    return Path(root) / session_id


class RAGEngine:
    """Document-grounded chat powered by LangGraph + Ollama."""

    def __init__(
        self,
        store: VectorStore,
        history_path: Path | None = None,
        sync_graph_db: bool = True,
    ) -> None:
        self.store = store
        self.graph = build_rag_graph(store)
        self.history: list[tuple[str, str]] = []
        self.history_path = history_path
        self.sync_graph_db = sync_graph_db
        if self.history_path is not None and self.history_path.exists():
            self._load_history()

    @classmethod
    def from_session(
        cls,
        session_id: str,
        root: str | Path = DEFAULT_RAG_DATA_ROOT,
    ) -> "RAGEngine":
        """Build a session-scoped RAG engine."""
        rag_dir = default_rag_dir(session_id, root)
        history_path = rag_dir / "history.json"
        store = VectorStore(persist_dir=rag_dir / "chroma")
        return cls(store=store, history_path=history_path)

    def ingest_files(self, file_paths: list[str | Path]) -> str:
        """Index ``.txt`` / ``.pdf`` files and optionally sync to Postgres/Neo4j."""
        if not file_paths:
            return "No files provided."

        valid = [p for p in file_paths if Path(p).suffix.lower() in SUPPORTED_SUFFIXES]
        if not valid:
            return "Only .txt and .pdf files are supported."

        try:
            added, indexed, papers = ingest_files(self.store, valid)
        except Exception as exc:  # noqa: BLE001 - surface Ollama/network errors in UI
            return (
                f"Indexing failed: {exc}. "
                "Is Ollama running? Try: ollama serve"
            )

        if not indexed:
            return "No readable content found in the uploaded files."

        sync_notes: list[str] = []
        if self.sync_graph_db:
            try:
                from db.sync import sync_paper_ingest

                for paper in papers:
                    result = sync_paper_ingest(paper)
                    if result.get("synced"):
                        sync_notes.append(
                            f"{paper['filename']}→paper {result['paper_id'][:8]}…"
                        )
                    elif result.get("reason"):
                        sync_notes.append(f"{paper['filename']}: {result['reason']}")
            except Exception as exc:  # noqa: BLE001
                sync_notes.append(f"graph sync skipped: {exc}")

        names = ", ".join(indexed)
        msg = f"Indexed {len(indexed)} file(s) ({added} chunks): {names}"
        if sync_notes:
            msg += "\n\nGraph/DB: " + "; ".join(sync_notes)
        return msg

    def generate_reply(self, user_text: str) -> str:
        """Run the RAG graph and return an answer."""
        user_text = user_text.strip()
        if not user_text:
            return ""

        try:
            result = self.graph.invoke(initial_rag_state(user_text))
            reply = str(result.get("answer", "")).strip()
        except Exception as exc:  # noqa: BLE001 - surface Ollama/network errors in UI
            reply = (
                f"Generation failed: {exc}. "
                "Is Ollama running with the chat model pulled?"
            )

        self.history.append((user_text, reply))
        self._save_history()
        return reply

    def reset(self) -> None:
        """Clear chat history and the vector index."""
        self.history = []
        self.store.reset()
        self._save_history()

    def _load_history(self) -> None:
        if self.history_path is None:
            return
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            self.history = [(str(u), str(b)) for u, b in data]
        except (OSError, ValueError, TypeError):
            self.history = []

    def _save_history(self) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([list(turn) for turn in self.history]),
            encoding="utf-8",
        )
        os.replace(tmp, self.history_path)
