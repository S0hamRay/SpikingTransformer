"""Postgres client and repository for academic paper metadata."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from db.config import PostgresConfig

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class PostgresStore:
    """Thin psycopg wrapper around the academic paper schema."""

    def __init__(self, config: PostgresConfig | None = None) -> None:
        self.config = config or PostgresConfig.from_env()
        self._conn = None

    def connect(self):
        import psycopg
        from psycopg.rows import dict_row

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.config.dsn, row_factory=dict_row)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        conn = self.connect()
        with conn.cursor() as cur:
            yield cur
            conn.commit()

    def init_schema(self) -> None:
        """Apply ``schema.sql``. Requires ``pgvector`` when available."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # Fall back without pgvector if the extension is unavailable.
        try:
            with self.cursor() as cur:
                cur.execute(sql)
        except Exception:
            # Reconnect after a failed transaction, then create tables without vector.
            self.close()
            fallback = _schema_without_vector(sql)
            with self.cursor() as cur:
                cur.execute(fallback)

    def upsert_paper(
        self,
        *,
        title: str,
        abstract: str | None = None,
        year: int | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        url: str | None = None,
        citation_count: int = 0,
        source_filename: str | None = None,
        venue_id: UUID | None = None,
    ) -> UUID:
        with self.cursor() as cur:
            if doi:
                cur.execute("SELECT id FROM papers WHERE doi = %s", (doi,))
                row = cur.fetchone()
                if row:
                    paper_id = row["id"]
                    cur.execute(
                        """
                        UPDATE papers
                        SET title = %s, abstract = COALESCE(%s, abstract),
                            year = COALESCE(%s, year),
                            arxiv_id = COALESCE(%s, arxiv_id),
                            url = COALESCE(%s, url),
                            citation_count = %s,
                            source_filename = COALESCE(%s, source_filename),
                            venue_id = COALESCE(%s, venue_id),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            title,
                            abstract,
                            year,
                            arxiv_id,
                            url,
                            citation_count,
                            source_filename,
                            venue_id,
                            paper_id,
                        ),
                    )
                    return paper_id

            cur.execute(
                """
                INSERT INTO papers (
                    title, abstract, year, doi, arxiv_id, url,
                    citation_count, source_filename, venue_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    title,
                    abstract,
                    year,
                    doi,
                    arxiv_id,
                    url,
                    citation_count,
                    source_filename,
                    venue_id,
                ),
            )
            return cur.fetchone()["id"]

    def upsert_author(
        self,
        *,
        name: str,
        affiliation: str | None = None,
        orcid: str | None = None,
    ) -> UUID:
        with self.cursor() as cur:
            if orcid:
                cur.execute("SELECT id FROM authors WHERE orcid = %s", (orcid,))
                row = cur.fetchone()
                if row:
                    return row["id"]
            cur.execute(
                """
                INSERT INTO authors (name, affiliation, orcid)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (name, affiliation, orcid),
            )
            return cur.fetchone()["id"]

    def link_author(
        self, paper_id: UUID, author_id: UUID, author_order: int = 0
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_authors (paper_id, author_id, author_order)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (paper_id, author_id, author_order),
            )

    def replace_chunks(
        self,
        paper_id: UUID,
        chunks: list[dict[str, Any]],
    ) -> list[UUID]:
        """Replace all chunks for a paper. Each chunk: text, section_type, position."""
        with self.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
            ids: list[UUID] = []
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (paper_id, text, section_type, position)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        paper_id,
                        chunk["text"],
                        chunk.get("section_type", "body"),
                        chunk.get("position", 0),
                    ),
                )
                ids.append(cur.fetchone()["id"])
            return ids

    def upsert_venue(
        self, *, name: str, venue_type: str | None = None, year: int | None = None
    ) -> UUID:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO venues (name, type, year)
                VALUES (%s, %s, %s)
                ON CONFLICT (name, year) DO UPDATE SET type = COALESCE(EXCLUDED.type, venues.type)
                RETURNING id
                """,
                (name, venue_type, year),
            )
            return cur.fetchone()["id"]

    def upsert_concept(
        self, *, name: str, concept_type: str, aliases: list[str] | None = None
    ) -> UUID:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO concepts (name, type, aliases)
                VALUES (%s, %s, %s)
                ON CONFLICT (name, type) DO UPDATE
                  SET aliases = COALESCE(EXCLUDED.aliases, concepts.aliases)
                RETURNING id
                """,
                (name, concept_type, aliases or []),
            )
            return cur.fetchone()["id"]

    def upsert_dataset(
        self,
        *,
        name: str,
        domain: str | None = None,
        size: str | None = None,
        url: str | None = None,
    ) -> UUID:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets (name, domain, size, url)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    domain = COALESCE(EXCLUDED.domain, datasets.domain),
                    size = COALESCE(EXCLUDED.size, datasets.size),
                    url = COALESCE(EXCLUDED.url, datasets.url)
                RETURNING id
                """,
                (name, domain, size, url),
            )
            return cur.fetchone()["id"]

    def upsert_institution(self, *, name: str, country: str | None = None) -> UUID:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO institutions (name, country)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE
                  SET country = COALESCE(EXCLUDED.country, institutions.country)
                RETURNING id
                """,
                (name, country),
            )
            return cur.fetchone()["id"]


def _schema_without_vector(sql: str) -> str:
    """Strip pgvector usage so schema can init on plain Postgres."""
    lines = []
    for line in sql.splitlines():
        if 'CREATE EXTENSION IF NOT EXISTS "vector"' in line:
            continue
        line = line.replace("vector(768)", "TEXT")
        lines.append(line)
    return "\n".join(lines)
