"""Tests for the RAG ingestion and LangGraph pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.engine import RAGEngine, default_rag_dir
from rag.graph import build_rag_graph
from rag.ingest import (
    chunk_text,
    ingest_files,
    ingest_plaintext_files,
    read_document,
    read_plaintext,
)
from rag.store import VectorStore


class FakeEmbeddings:
    """Deterministic embeddings for tests (no Ollama required)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.5, 0.0] for i, _ in enumerate(texts)]

    def embed_query(self, query: str) -> list[float]:
        if "spiking" in query.lower():
            return [0.0, 0.5, 0.0]
        return [1.0, 0.5, 0.0]


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text(
        "Spiking transformers use addition-only attention.\n"
        "Standard transformers use softmax attention.\n",
        encoding="utf-8",
    )
    return path


def _make_store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "chroma", embeddings=FakeEmbeddings())


def test_read_plaintext(sample_txt: Path) -> None:
    text = read_plaintext(sample_txt)
    assert "Spiking transformers" in text


def test_chunk_text_adds_source_metadata() -> None:
    docs = chunk_text(
        "alpha beta gamma delta",
        source="demo.txt",
        chunk_size=10,
        chunk_overlap=2,
    )
    assert docs
    assert all(doc.metadata["source"] == "demo.txt" for doc in docs)
    assert all("position" in doc.metadata for doc in docs)


def test_ingest_plaintext_files_indexes_chunks(
    tmp_path: Path, sample_txt: Path
) -> None:
    store = _make_store(tmp_path)
    added, indexed = ingest_plaintext_files(store, [sample_txt])
    assert added > 0
    assert indexed == ["notes.txt"]
    assert store.count() == added


def test_ingest_files_returns_paper_records(
    tmp_path: Path, sample_txt: Path
) -> None:
    store = _make_store(tmp_path)
    added, indexed, papers = ingest_files(store, [sample_txt])
    assert added > 0
    assert indexed == ["notes.txt"]
    assert len(papers) == 1
    assert papers[0]["title"] == "notes"
    assert papers[0]["chunks"]


def test_read_document_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("rag.ingest.read_pdf", return_value="Abstract\nSpiking attention works."):
        text = read_document(pdf_path)
    assert "Spiking" in text


def test_ingest_pdf_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    store = _make_store(tmp_path)

    with (
        patch(
            "rag.ingest.read_pdf",
            return_value="Abstract\nSpiking transformers use addition-only attention.",
        ),
        patch(
            "rag.ingest.pdf_metadata",
            return_value={"title": "My Paper", "author": "Ada Lovelace"},
        ),
    ):
        added, indexed, papers = ingest_files(store, [pdf_path])

    assert indexed == ["paper.pdf"]
    assert added > 0
    assert papers[0]["title"] == "My Paper"
    assert papers[0]["author"] == "Ada Lovelace"


def test_similarity_search_returns_documents(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.add_documents(
        [
            Document(page_content="spiking attention", metadata={"source": "a.txt"}),
            Document(page_content="softmax attention", metadata={"source": "b.txt"}),
        ]
    )
    results = store.similarity_search("spiking", k=1)
    assert len(results) == 1
    assert "spiking" in results[0].page_content


def test_rag_graph_generate_without_context(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    graph = build_rag_graph(store)
    result = graph.invoke({"question": "What is spiking?", "context": "", "answer": ""})
    assert "No documents have been indexed" in result["answer"]


def test_retrieve_node_finds_relevant_context(tmp_path: Path) -> None:
    from rag.graph import build_retrieve_node

    store = _make_store(tmp_path)
    store.add_documents(
        [
            Document(page_content="spiking attention details", metadata={"source": "a.txt"}),
            Document(page_content="softmax attention details", metadata={"source": "b.txt"}),
        ]
    )
    retrieve = build_retrieve_node(store, top_k=1)
    result = retrieve({"question": "Tell me about spiking", "context": "", "answer": ""})
    assert "spiking" in result["context"].lower()


def test_rag_engine_ingest_and_reply(tmp_path: Path, sample_txt: Path) -> None:
    store = _make_store(tmp_path)
    engine = RAGEngine(
        store=store,
        history_path=tmp_path / "history.json",
        sync_graph_db=False,
    )
    status = engine.ingest_files([sample_txt])
    assert "Indexed 1 file" in status

    with patch.object(
        engine.graph,
        "invoke",
        return_value={"answer": "They differ in attention mechanism."},
    ):
        reply = engine.generate_reply("How do they differ?")

    assert "attention" in reply.lower()
    assert len(engine.history) == 1


def test_rag_engine_reset_clears_history_and_store(
    tmp_path: Path, sample_txt: Path
) -> None:
    store = _make_store(tmp_path)
    engine = RAGEngine(
        store=store,
        history_path=tmp_path / "history.json",
        sync_graph_db=False,
    )
    engine.ingest_files([sample_txt])
    engine.history.append(("hello", "world"))
    engine.reset()

    assert engine.history == []
    assert store.count() == 0


def test_default_rag_dir() -> None:
    assert default_rag_dir("abc") == Path("rag_data/abc")


def test_sync_paper_ingest_mocked() -> None:
    from langchain_core.documents import Document

    from db.sync import sync_paper_ingest

    paper = {
        "filename": "demo.pdf",
        "title": "Demo Paper",
        "author": "Ada Lovelace",
        "text": "Abstract\nHello world.",
        "chunks": [
            Document(page_content="Hello world.", metadata={"position": 0}),
        ],
    }

    pg = MagicMock()
    pg.upsert_paper.return_value = "11111111-1111-1111-1111-111111111111"
    pg.upsert_author.return_value = "22222222-2222-2222-2222-222222222222"
    pg.replace_chunks.return_value = ["33333333-3333-3333-3333-333333333333"]

    graph = MagicMock()

    with patch("db.sync.sync_enabled", return_value=True):
        result = sync_paper_ingest(paper, postgres=pg, neo4j=graph)

    assert result["synced"] is True
    pg.upsert_paper.assert_called_once()
    graph.upsert_paper.assert_called_once()
    graph.upsert_chunk.assert_called_once()
