"""Tests for the RAG ingestion and LangGraph pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.engine import RAGEngine, default_rag_dir
from rag.graph import (
    build_rag_graph,
    build_retrieve_node,
    initial_rag_state,
    route_after_grade,
)
from rag.ingest import (
    chunk_text,
    ingest_files,
    ingest_plaintext_files,
    read_document,
    read_plaintext,
)
from rag.store import VectorStore
from rag.web_search import format_tavily_results, tavily_configured


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


def test_web_search_without_key_explains_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.graph import build_web_search_node

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch("rag.graph.tavily_configured", return_value=False):
        out = build_web_search_node()(initial_rag_state("What is spiking?"))
    assert "TAVILY_API_KEY" in out["answer"]


def test_crag_graph_routes_to_web_when_docs_irrelevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_store(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    with (
        patch(
            "rag.graph.build_grade_documents_node",
            return_value=lambda state: {
                "documents": [],
                "context": "",
                "needs_web": "yes",
            },
        ),
        patch(
            "rag.graph.build_rewrite_query_node",
            return_value=lambda state: {"search_query": "rewritten query"},
        ),
        patch(
            "rag.graph.tavily_search",
            return_value="Web: spiking transformers use event-driven spikes.",
        ),
        patch(
            "rag.graph.build_generate_node",
            return_value=lambda state: {
                "answer": f"Answer from: {state.get('context', '')}"
            },
        ),
    ):
        graph = build_rag_graph(store)
        result = graph.invoke(initial_rag_state("What is spiking?"))

    assert "event-driven" in result["answer"].lower()


def test_retrieve_node_finds_relevant_context(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.add_documents(
        [
            Document(page_content="spiking attention details", metadata={"source": "a.txt"}),
            Document(page_content="softmax attention details", metadata={"source": "b.txt"}),
        ]
    )
    retrieve = build_retrieve_node(store, top_k=1)
    result = retrieve(initial_rag_state("Tell me about spiking"))
    assert "spiking" in result["context"].lower()


def test_route_after_grade() -> None:
    assert route_after_grade({"needs_web": "yes"}) == "rewrite_query"
    assert route_after_grade({"needs_web": "no"}) == "generate"


def test_format_tavily_results() -> None:
    text = format_tavily_results(
        {
            "results": [
                {
                    "title": "Spiking nets",
                    "url": "https://example.com",
                    "content": "Binary spikes.",
                }
            ]
        }
    )
    assert "Spiking nets" in text
    assert "Binary spikes." in text


def test_tavily_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert tavily_configured() is False
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert tavily_configured() is True


def test_web_search_node_uses_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag.graph import build_web_search_node

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    with patch("rag.graph.tavily_search", return_value="web snippet about spikes"):
        node = build_web_search_node()
        out = node(initial_rag_state("What are spikes?"))
    assert "web snippet" in out["context"]


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
