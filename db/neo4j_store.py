"""Neo4j graph store for academic Graph RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from db.config import Neo4jConfig

SCHEMA_PATH = Path(__file__).with_name("neo4j_schema.cypher")


class Neo4jGraphStore:
    """Neo4j client implementing the academic paper graph schema.

    Node types: Paper, Author, Chunk, Concept, Venue, Dataset, Institution,
    Community
    Relationships:
      Paper-[:AUTHORED_BY]->Author
      Paper-[:PUBLISHED_IN]->Venue
      Paper-[:CITES]->Paper
      Paper-[:HAS_CHUNK]->Chunk
      Chunk-[:MENTIONS]->Concept
      Paper-[:USES_METHOD]->Concept
      Paper-[:EVALUATES_ON]->Dataset
      Paper-[:REPORTS_METRIC]->Concept  (relationship property: value)
      Paper-[:EXTENDS]->Paper
      Concept-[:RELATED_TO]->Concept
      Author-[:AFFILIATED_WITH]->Institution
      Concept-[:CO_OCCURS_WITH {weight}]->Concept
      Concept-[:BELONGS_TO]->Community
      Paper-[:BELONGS_TO]->Community
    """

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or Neo4jConfig.from_env()
        self._driver = None

    def connect(self):
        from neo4j import GraphDatabase

        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def init_schema(self) -> None:
        statements = [
            s.strip()
            for s in SCHEMA_PATH.read_text(encoding="utf-8").split(";")
            if s.strip() and not s.strip().startswith("//")
        ]
        driver = self.connect()
        with driver.session() as session:
            for stmt in statements:
                session.run(stmt)

    def upsert_paper(self, paper: dict[str, Any]) -> str:
        """Upsert a Paper node. ``paper`` must include ``id`` (str/UUID)."""
        paper_id = str(paper["id"])
        query = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.year = $year,
            p.doi = $doi,
            p.arxiv_id = $arxiv_id,
            p.url = $url,
            p.citation_count = $citation_count,
            p.source_filename = $source_filename
        RETURN p.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=paper_id,
                title=paper.get("title"),
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                doi=paper.get("doi"),
                arxiv_id=paper.get("arxiv_id"),
                url=paper.get("url"),
                citation_count=paper.get("citation_count", 0),
                source_filename=paper.get("source_filename"),
            )
            return result.single()["id"]

    def upsert_author(self, author: dict[str, Any]) -> str:
        author_id = str(author["id"])
        query = """
        MERGE (a:Author {id: $id})
        SET a.name = $name,
            a.affiliation = $affiliation,
            a.orcid = $orcid
        RETURN a.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=author_id,
                name=author.get("name"),
                affiliation=author.get("affiliation"),
                orcid=author.get("orcid"),
            )
            return result.single()["id"]

    def link_authored_by(self, paper_id: UUID | str, author_id: UUID | str) -> None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (a:Author {id: $author_id})
        MERGE (p)-[:AUTHORED_BY]->(a)
        """
        with self.connect().session() as session:
            session.run(query, paper_id=str(paper_id), author_id=str(author_id))

    def upsert_chunk(self, chunk: dict[str, Any], paper_id: UUID | str) -> str:
        chunk_id = str(chunk["id"])
        query = """
        MATCH (p:Paper {id: $paper_id})
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.section_type = $section_type,
            c.position = $position
        MERGE (p)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                paper_id=str(paper_id),
                id=chunk_id,
                text=chunk.get("text"),
                section_type=chunk.get("section_type", "body"),
                position=chunk.get("position", 0),
            )
            return result.single()["id"]

    def upsert_venue(self, venue: dict[str, Any]) -> str:
        venue_id = str(venue["id"])
        query = """
        MERGE (v:Venue {id: $id})
        SET v.name = $name, v.type = $type, v.year = $year
        RETURN v.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=venue_id,
                name=venue.get("name"),
                type=venue.get("type"),
                year=venue.get("year"),
            )
            return result.single()["id"]

    def link_published_in(self, paper_id: UUID | str, venue_id: UUID | str) -> None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (v:Venue {id: $venue_id})
        MERGE (p)-[:PUBLISHED_IN]->(v)
        """
        with self.connect().session() as session:
            session.run(query, paper_id=str(paper_id), venue_id=str(venue_id))

    def link_cites(self, citing_id: UUID | str, cited_id: UUID | str) -> None:
        query = """
        MATCH (a:Paper {id: $citing_id})
        MATCH (b:Paper {id: $cited_id})
        MERGE (a)-[:CITES]->(b)
        """
        with self.connect().session() as session:
            session.run(query, citing_id=str(citing_id), cited_id=str(cited_id))

    def link_extends(self, derived_id: UUID | str, base_id: UUID | str) -> None:
        query = """
        MATCH (a:Paper {id: $derived_id})
        MATCH (b:Paper {id: $base_id})
        MERGE (a)-[:EXTENDS]->(b)
        """
        with self.connect().session() as session:
            session.run(query, derived_id=str(derived_id), base_id=str(base_id))

    def upsert_concept(self, concept: dict[str, Any]) -> str:
        concept_id = str(concept["id"])
        query = """
        MERGE (c:Concept {id: $id})
        SET c.name = $name, c.type = $type, c.aliases = $aliases
        RETURN c.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=concept_id,
                name=concept.get("name"),
                type=concept.get("type"),
                aliases=concept.get("aliases") or [],
            )
            return result.single()["id"]

    def link_uses_method(self, paper_id: UUID | str, concept_id: UUID | str) -> None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (p)-[:USES_METHOD]->(c)
        """
        with self.connect().session() as session:
            session.run(query, paper_id=str(paper_id), concept_id=str(concept_id))

    def link_mentions(self, chunk_id: UUID | str, concept_id: UUID | str) -> None:
        query = """
        MATCH (ch:Chunk {id: $chunk_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (ch)-[:MENTIONS]->(c)
        """
        with self.connect().session() as session:
            session.run(query, chunk_id=str(chunk_id), concept_id=str(concept_id))

    def upsert_dataset(self, dataset: dict[str, Any]) -> str:
        dataset_id = str(dataset["id"])
        query = """
        MERGE (d:Dataset {id: $id})
        SET d.name = $name, d.domain = $domain, d.size = $size, d.url = $url
        RETURN d.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=dataset_id,
                name=dataset.get("name"),
                domain=dataset.get("domain"),
                size=dataset.get("size"),
                url=dataset.get("url"),
            )
            return result.single()["id"]

    def link_evaluates_on(self, paper_id: UUID | str, dataset_id: UUID | str) -> None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (d:Dataset {id: $dataset_id})
        MERGE (p)-[:EVALUATES_ON]->(d)
        """
        with self.connect().session() as session:
            session.run(query, paper_id=str(paper_id), dataset_id=str(dataset_id))

    def link_reports_metric(
        self, paper_id: UUID | str, concept_id: UUID | str, value: str | None = None
    ) -> None:
        query = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (c:Concept {id: $concept_id})
        MERGE (p)-[r:REPORTS_METRIC]->(c)
        SET r.value = $value
        """
        with self.connect().session() as session:
            session.run(
                query,
                paper_id=str(paper_id),
                concept_id=str(concept_id),
                value=value,
            )

    def link_related_concepts(
        self, concept_id: UUID | str, related_id: UUID | str
    ) -> None:
        query = """
        MATCH (a:Concept {id: $concept_id})
        MATCH (b:Concept {id: $related_id})
        MERGE (a)-[:RELATED_TO]->(b)
        """
        with self.connect().session() as session:
            session.run(query, concept_id=str(concept_id), related_id=str(related_id))

    def upsert_institution(self, institution: dict[str, Any]) -> str:
        institution_id = str(institution["id"])
        query = """
        MERGE (i:Institution {id: $id})
        SET i.name = $name, i.country = $country
        RETURN i.id AS id
        """
        with self.connect().session() as session:
            result = session.run(
                query,
                id=institution_id,
                name=institution.get("name"),
                country=institution.get("country"),
            )
            return result.single()["id"]

    def link_affiliated_with(
        self, author_id: UUID | str, institution_id: UUID | str
    ) -> None:
        query = """
        MATCH (a:Author {id: $author_id})
        MATCH (i:Institution {id: $institution_id})
        MERGE (a)-[:AFFILIATED_WITH]->(i)
        """
        with self.connect().session() as session:
            session.run(
                query, author_id=str(author_id), institution_id=str(institution_id)
            )
