"""Sync ingested documents into Postgres + Neo4j for Graph RAG."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from db.config import sync_enabled
from db.neo4j_store import Neo4jGraphStore
from db.postgres_store import PostgresStore


def sync_paper_ingest(
    paper: dict[str, Any],
    *,
    postgres: PostgresStore | None = None,
    neo4j: Neo4jGraphStore | None = None,
) -> dict[str, Any]:
    """Write a paper + chunks from a RAG ingest record into both stores.

    ``paper`` is the dict produced by ``rag.ingest.ingest_files``:
    ``{filename, title, author, text, chunks}``.

    Returns:
        ``{paper_id, chunk_ids, synced}`` or ``{synced: False, reason: ...}``.
    """
    if not sync_enabled():
        return {"synced": False, "reason": "RAG_SYNC_GRAPH_DB disabled"}

    pg = postgres or PostgresStore()
    graph = neo4j or Neo4jGraphStore()

    try:
        pg.init_schema()
        graph.init_schema()
    except Exception as exc:  # noqa: BLE001 - optional infra; don't break RAG
        return {"synced": False, "reason": f"DB unavailable: {exc}"}

    try:
        abstract = _guess_abstract(paper.get("text") or "")
        paper_id = pg.upsert_paper(
            title=paper.get("title") or paper.get("filename") or "Untitled",
            abstract=abstract,
            source_filename=paper.get("filename"),
        )

        author_id: UUID | None = None
        if paper.get("author"):
            # PDF author metadata may be a comma-separated list.
            names = [n.strip() for n in str(paper["author"]).split(",") if n.strip()]
            for i, name in enumerate(names):
                aid = pg.upsert_author(name=name)
                pg.link_author(paper_id, aid, author_order=i)
                if i == 0:
                    author_id = aid

        chunk_payload = [
            {
                "text": doc.page_content,
                "section_type": _guess_section(doc.page_content),
                "position": doc.metadata.get("position", i),
            }
            for i, doc in enumerate(paper.get("chunks") or [])
        ]
        chunk_ids = pg.replace_chunks(paper_id, chunk_payload)

        # Mirror into Neo4j
        graph.upsert_paper(
            {
                "id": paper_id,
                "title": paper.get("title") or paper.get("filename"),
                "abstract": abstract,
                "source_filename": paper.get("filename"),
            }
        )
        if author_id is not None and paper.get("author"):
            names = [n.strip() for n in str(paper["author"]).split(",") if n.strip()]
            for i, name in enumerate(names):
                # Re-fetch author ids by re-upserting (idempotent).
                aid = pg.upsert_author(name=name)
                graph.upsert_author({"id": aid, "name": name})
                graph.link_authored_by(paper_id, aid)

        for cid, payload in zip(chunk_ids, chunk_payload, strict=True):
            graph.upsert_chunk({"id": cid, **payload}, paper_id)

        return {
            "synced": True,
            "paper_id": str(paper_id),
            "chunk_ids": [str(c) for c in chunk_ids],
        }
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": str(exc)}
    finally:
        if postgres is None:
            pg.close()
        if neo4j is None:
            graph.close()


def _guess_abstract(text: str, max_chars: int = 1200) -> str:
    lower = text.lower()
    if "abstract" in lower:
        idx = lower.find("abstract")
        snippet = text[idx : idx + max_chars]
        return snippet.strip()
    return text[:max_chars].strip()


def _guess_section(chunk: str) -> str:
    head = chunk[:80].lower()
    for label in (
        "abstract",
        "introduction",
        "method",
        "methods",
        "related work",
        "experiment",
        "results",
        "discussion",
        "conclusion",
        "references",
    ):
        if label in head:
            return label.replace(" ", "_")
    return "body"
