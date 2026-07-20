"""Chroma vector store wrapper for RAG document chunks."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


class VectorStore:
    """Session-scoped Chroma collection backed by Ollama embeddings."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "rag_docs",
        embeddings: OllamaEmbeddings | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection: Collection | None = None
        self._embeddings = embeddings or OllamaEmbeddings(
            model=_embed_model(),
            base_url=_ollama_base_url(),
        )

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
        """Return the top-k most similar document chunks for a query."""
        if self.collection.count() == 0:
            return []

        query_embedding = self._embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs: list[Document] = []
        for text, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
            strict=True,
        ):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete the collection and its persisted data."""
        try:
            self._client.delete_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
        self._collection = None
