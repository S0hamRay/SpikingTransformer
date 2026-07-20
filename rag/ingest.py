"""Document ingestion: read plaintext/PDF files, chunk, and index."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.pdf import pdf_metadata, read_pdf
from rag.store import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, VectorStore

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def read_plaintext(path: str | Path) -> str:
    """Read a plaintext file as UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def read_document(path: str | Path) -> str:
    """Read a supported document (``.txt`` or ``.pdf``) into text."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return read_plaintext(p)
    if suffix == ".pdf":
        return read_pdf(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """Split text into overlapping chunks with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    base = {"source": source}
    if extra_metadata:
        base.update(extra_metadata)
    return [
        Document(
            page_content=chunk,
            metadata={**base, "position": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def ingest_files(
    store: VectorStore,
    file_paths: list[str | Path],
) -> tuple[int, list[str], list[dict]]:
    """Read, chunk, and index supported document files.

    Returns:
        ``(chunks_added, indexed_filenames, paper_records)`` where each
        paper record is a dict suitable for Postgres/Neo4j sync:
        ``{filename, title, author, text, chunks}``.
    """
    all_docs: list[Document] = []
    indexed: list[str] = []
    papers: list[dict] = []

    for path in file_paths:
        p = Path(path)
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if not p.exists():
            continue

        try:
            text = read_document(p)
        except (ValueError, OSError):
            continue
        if not text.strip():
            continue

        meta: dict = {"file_type": p.suffix.lower().lstrip(".")}
        title = p.stem
        author = None
        if p.suffix.lower() == ".pdf":
            pdf_meta = pdf_metadata(p)
            if pdf_meta.get("title"):
                title = pdf_meta["title"]  # type: ignore[assignment]
            author = pdf_meta.get("author")
            meta["title"] = title

        docs = chunk_text(text, source=p.name, extra_metadata=meta)
        all_docs.extend(docs)
        indexed.append(p.name)
        papers.append(
            {
                "filename": p.name,
                "path": str(p),
                "title": title,
                "author": author,
                "text": text,
                "chunks": docs,
            }
        )

    added = store.add_documents(all_docs)
    return added, indexed, papers


# Backwards-compatible alias used by older tests / callers.
def ingest_plaintext_files(
    store: VectorStore,
    file_paths: list[str | Path],
) -> tuple[int, list[str]]:
    """Read, chunk, and index plaintext files (``.txt`` only)."""
    txt_only = [p for p in file_paths if Path(p).suffix.lower() == ".txt"]
    added, indexed, _ = ingest_files(store, txt_only)
    return added, indexed
