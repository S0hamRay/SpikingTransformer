"""Sync ingested documents into Postgres + Neo4j for Graph RAG."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from db.config import sync_enabled
from db.entity_extraction import ExtractedEntity, extract_entities_from_chunks
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

    Also runs basic entity extraction on chunks and links Concepts
    (``MENTIONS`` / ``USES_METHOD`` / ``EVALUATES_ON`` / ``REPORTS_METRIC``).

    Returns:
        ``{paper_id, chunk_ids, concept_ids, synced}`` or
        ``{synced: False, reason: ...}``.
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

        concept_ids = _sync_extracted_entities(
            pg,
            graph,
            paper_id=paper_id,
            chunk_ids=chunk_ids,
            chunk_texts=[c["text"] for c in chunk_payload],
        )

        return {
            "synced": True,
            "paper_id": str(paper_id),
            "chunk_ids": [str(c) for c in chunk_ids],
            "concept_ids": [str(c) for c in concept_ids],
        }
    except Exception as exc:  # noqa: BLE001
        return {"synced": False, "reason": str(exc)}
    finally:
        if postgres is None:
            pg.close()
        if neo4j is None:
            graph.close()


def _sync_extracted_entities(
    pg: PostgresStore,
    graph: Neo4jGraphStore,
    *,
    paper_id: UUID,
    chunk_ids: list[UUID],
    chunk_texts: list[str],
) -> list[UUID]:
    """Extract entities from chunks and write Concept links to both stores."""
    per_chunk = extract_entities_from_chunks(chunk_texts)
    concept_cache: dict[tuple[str, str], UUID] = {}
    paper_linked: set[tuple[str, str]] = set()

    def _ensure_concept(entity: ExtractedEntity) -> UUID:
        key = (entity.name.lower(), entity.concept_type)
        if key in concept_cache:
            return concept_cache[key]
        concept_id = pg.upsert_concept(
            name=entity.name,
            concept_type=entity.concept_type,
            aliases=list(entity.aliases),
        )
        graph.upsert_concept(
            {
                "id": concept_id,
                "name": entity.name,
                "type": entity.concept_type,
                "aliases": list(entity.aliases),
            }
        )
        concept_cache[key] = concept_id
        return concept_id

    for chunk_id, entities in zip(chunk_ids, per_chunk, strict=True):
        for entity in entities:
            concept_id = _ensure_concept(entity)
            pg.link_chunk_concept(chunk_id, concept_id)
            graph.link_mentions(chunk_id, concept_id)

            paper_key = (entity.name.lower(), entity.concept_type)
            if paper_key in paper_linked:
                # Still refresh metric value if a later chunk has one.
                if entity.concept_type == "metric" and entity.metric_value:
                    pg.link_metric(paper_id, concept_id, value=entity.metric_value)
                    graph.link_reports_metric(
                        paper_id, concept_id, value=entity.metric_value
                    )
                continue
            paper_linked.add(paper_key)

            if entity.concept_type == "method":
                pg.link_method(paper_id, concept_id)
                graph.link_uses_method(paper_id, concept_id)
            elif entity.concept_type == "dataset":
                dataset_id = pg.upsert_dataset(name=entity.name)
                pg.link_dataset(paper_id, dataset_id)
                graph.upsert_dataset({"id": dataset_id, "name": entity.name})
                graph.link_evaluates_on(paper_id, dataset_id)
            elif entity.concept_type == "metric":
                pg.link_metric(paper_id, concept_id, value=entity.metric_value)
                graph.link_reports_metric(
                    paper_id, concept_id, value=entity.metric_value
                )
            elif entity.concept_type in {"architecture", "task", "other"}:
                # Architectures / tasks are paper-level methods-ish for Leiden.
                pg.link_method(paper_id, concept_id)
                graph.link_uses_method(paper_id, concept_id)

    return list(concept_cache.values())


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
