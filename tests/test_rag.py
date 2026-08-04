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


def test_ollama_batched_embeddings_batches_and_truncates() -> None:
    from rag.store import OllamaBatchedEmbeddings

    emb = OllamaBatchedEmbeddings(batch_size=2, retries=2)
    client = MagicMock()
    client.embed.side_effect = [
        {"embeddings": [[0.1], [0.2]]},
        {"embeddings": [[0.3]]},
    ]
    emb._client = client

    out = emb.embed_documents(["a", "b", "c"])
    assert out == [[0.1], [0.2], [0.3]]
    assert client.embed.call_count == 2
    first_kwargs = client.embed.call_args_list[0].kwargs
    assert first_kwargs["truncate"] is True
    assert first_kwargs["options"]["num_ctx"] == 2048


def test_ollama_batched_embeddings_retries_then_succeeds() -> None:
    from rag.store import OllamaBatchedEmbeddings

    emb = OllamaBatchedEmbeddings(batch_size=8, retries=3)
    client = MagicMock()
    client.embed.side_effect = [
        ConnectionError("connection reset by peer"),
        {"embeddings": [[0.5]]},
    ]
    emb._client = client

    with patch("rag.store.time.sleep"):
        out = emb.embed_documents(["hello"])
    assert out == [[0.5]]
    assert client.embed.call_count == 2


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


def test_overview_query_prefers_front_matter(tmp_path: Path) -> None:
    """Appendix prompts mentioning 'summary of the paper' must not beat the abstract."""
    store = _make_store(tmp_path)
    store.add_documents(
        [
            Document(
                page_content=(
                    "Effective Strategies for Agents. Abstract: We propose CAID, "
                    "a multi-agent framework for asynchronous software engineering."
                ),
                metadata={"source": "p.pdf", "position": 0, "title": "CAID"},
            ),
            Document(
                page_content=(
                    "Provide engineers with a detailed summary of the paper based "
                    "on your exploration of the overall structure of the paper."
                ),
                metadata={"source": "p.pdf", "position": 170, "title": "CAID"},
            ),
        ]
    )
    results = store.similarity_search("What is the paper about?", k=2)
    assert results
    assert "CAID" in results[0].page_content
    assert "Abstract" in results[0].page_content


def test_web_search_without_key_explains_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.graph import build_web_search_node

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with patch("rag.graph.tavily_configured", return_value=False):
        out = build_web_search_node()(initial_rag_state("What is spiking?"))
    assert "TAVILY_API_KEY" in out["answer"]
    assert "Upload" in out["answer"]


def test_grade_keeps_retrieval_when_tavily_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag.graph import build_grade_documents_node

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fake_response = MagicMock()
    fake_response.content = "no"
    with (
        patch("rag.graph.tavily_configured", return_value=False),
        patch("rag.graph._chat_model") as chat_model,
    ):
        chat_model.return_value.invoke.return_value = fake_response
        # GRADE_PROMPT | model uses model as runnable; patch the chain invoke path
        with patch("rag.graph.GRADE_PROMPT") as prompt:
            chain = MagicMock()
            chain.invoke.return_value = fake_response
            prompt.__or__.return_value = chain
            out = build_grade_documents_node()(
                {
                    **initial_rag_state("What is spiking?"),
                    "documents": ["spiking transformers use spikes"],
                }
            )
    assert out["needs_web"] == "no"
    assert "spiking transformers" in out["context"]


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


def test_rag_engine_reset_keeps_index_by_default(
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
    before = store.count()
    engine.reset()

    assert engine.history == []
    assert store.count() == before > 0


def test_rag_engine_reset_can_clear_index(
    tmp_path: Path, sample_txt: Path
) -> None:
    store = _make_store(tmp_path)
    engine = RAGEngine(
        store=store,
        history_path=tmp_path / "history.json",
        sync_graph_db=False,
    )
    engine.ingest_files([sample_txt])
    engine.reset(clear_index=True)

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
