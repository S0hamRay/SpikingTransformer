"""Chroma vector store wrapper for RAG document chunks."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 6
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
# Overview questions often match appendix prompt text ("summary of the paper")
# more strongly than the abstract — bias retrieval toward early chunks.
_OVERVIEW_QUERY = re.compile(
    r"\b("
    r"about|summary|summarize|abstract|overview|title|contribution|contributions|"
    r"propose|proposed|main\s+idea|what\s+is\s+this\s+paper|paper\s+about"
    r")\b",
    re.IGNORECASE,
)
# nomic-embed-text trains at 2048; keep batches small so the Ollama runner
# does not crash mid-/tokenize (connection reset → opaque 400).
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_EMBED_NUM_CTX = 2048
DEFAULT_EMBED_RETRIES = 3


def _embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def _embed_batch_size() -> int:
    return int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))


class Embeddings(Protocol):
    """Minimal embedding interface used by :class:`VectorStore`."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaBatchedEmbeddings:
    """Ollama embeddings with truncation, batching, and retries.

    LangChain's ``OllamaEmbeddings`` does not pass ``truncate=True``. Oversized
    or bursty batches can crash Ollama's runner (``read: connection reset by
    peer`` on ``/tokenize``), which surfaces as a confusing 400.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        batch_size: int | None = None,
        num_ctx: int = DEFAULT_EMBED_NUM_CTX,
        retries: int = DEFAULT_EMBED_RETRIES,
        keep_alive: str | float | None = "10m",
    ) -> None:
        self.model = model or _embed_model()
        self.base_url = base_url or _ollama_base_url()
        self.batch_size = max(1, batch_size or _embed_batch_size())
        self.num_ctx = num_ctx
        self.retries = max(1, retries)
        self.keep_alive = keep_alive
        self._client = None

    def _get_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.base_url)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._get_client().embed(
                    model=self.model,
                    input=texts,
                    truncate=True,
                    options={"num_ctx": self.num_ctx},
                    keep_alive=self.keep_alive,
                )
                embeddings = response["embeddings"]
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Ollama returned {len(embeddings)} embeddings "
                        f"for {len(texts)} texts"
                    )
                return list(embeddings)
            except Exception as exc:  # noqa: BLE001 - retry transient runner crashes
                last_exc = exc
                if attempt + 1 >= self.retries:
                    break
                # Give the Ollama runner a moment to respawn after a reset.
                time.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise RuntimeError(
            f"Ollama embedding failed after {self.retries} attempts "
            f"({self.model} @ {self.base_url}): {last_exc}"
        ) from last_exc


class VectorStore:
    """Session-scoped Chroma collection backed by Ollama embeddings."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "rag_docs",
        embeddings: Embeddings | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection: Collection | None = None
        self._embeddings: Embeddings = embeddings or OllamaBatchedEmbeddings()

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, documents: list[Document]) -> int:
        """Embed and index documents. Returns the number of chunks added."""
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self._embeddings.embed_documents(texts)
        ids = [f"chunk_{self.collection.count() + i}" for i in range(len(texts))]
        metadatas = [
            doc.metadata if doc.metadata else {"source": "unknown"}
            for doc in documents
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(documents)

    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K) -> list[Document]:
        """Return the top-k most similar document chunks for a query.

        Fetches a wider candidate pool, deduplicates, and applies a position
        prior so title/abstract chunks win over appendix prompt text for
        overview questions like "what is the paper about?".
        """
        if self.collection.count() == 0:
            return []

        overview = bool(_OVERVIEW_QUERY.search(query or ""))
        fetch_k = min(max(k * 5, 20), self.collection.count())
        query_embedding = self._embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        scored: list[tuple[float, Document]] = []
        seen: set[str] = set()
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            key = " ".join((text or "").split())
            if not key or key in seen:
                continue
            seen.add(key)
            metadata = meta or {}
            pos = _chunk_position(metadata)
            # Cosine distance: lower is better. Prefer front-matter chunks.
            bias = 0.0
            if pos <= 2:
                bias -= 0.40
            elif pos <= 8:
                bias -= 0.18
            if overview and pos <= 5:
                bias -= 0.25
            scored.append(
                (
                    float(dist) + bias,
                    Document(page_content=text, metadata=metadata),
                )
            )

        scored.sort(key=lambda item: item[0])
        docs = [doc for _, doc in scored[:k]]

        if overview:
            docs = self._merge_front_matter(docs, k=k, front_n=3)

        return docs

    def _merge_front_matter(
        self, docs: list[Document], *, k: int, front_n: int = 3
    ) -> list[Document]:
        """Ensure the earliest paper chunks (title/abstract) are in context."""
        front = self._front_matter_chunks(limit=front_n)
        if not front:
            return docs

        merged: list[Document] = []
        seen: set[str] = set()
        for doc in front + docs:
            key = " ".join(doc.page_content.split())
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= k:
                break
        return merged

    def _front_matter_chunks(self, limit: int = 3) -> list[Document]:
        """Return unique lowest-position chunks (typically title/abstract)."""
        raw = self.collection.get(include=["documents", "metadatas"])
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        items = sorted(
            (
                (_chunk_position(meta), text, meta or {})
                for text, meta in zip(documents, metadatas, strict=True)
            ),
            key=lambda item: item[0],
        )
        out: list[Document] = []
        seen_pos: set[int] = set()
        for pos, text, meta in items:
            if pos in seen_pos or not (text or "").strip():
                continue
            seen_pos.add(pos)
            out.append(Document(page_content=text, metadata=meta))
            if len(out) >= limit:
                break
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete the collection and its persisted data."""
        try:
            self._client.delete_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
        self._collection = None


def _chunk_position(metadata: dict) -> int:
    try:
        return int(metadata.get("position", 10**9))
    except (TypeError, ValueError):
        return 10**9
