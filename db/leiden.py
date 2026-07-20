"""Leiden community detection on Concept co-occurrence via Neo4j GDS."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from db.config import Neo4jConfig, leiden_gamma
from db.neo4j_store import Neo4jGraphStore

GRAPH_NAME = "leiden_concepts"

class LeidenError(RuntimeError):
    """Raised when the Concept graph is insufficient or GDS is unavailable."""


@dataclass(frozen=True)
class LeidenStats:
    community_count: int
    modularity: float
    size_min: int
    size_median: float
    size_max: int
    gamma: float
    concept_count: int
    co_occur_edges: int
    papers_assigned: int

    def format_summary(self) -> str:
        return (
            f"Leiden communities: {self.community_count}\n"
            f"Modularity: {self.modularity:.3f}\n"
            f"Size — min: {self.size_min}  median: {self.size_median:g}  "
            f"max: {self.size_max}\n"
            f"gamma: {self.gamma}\n"
            f"Concepts: {self.concept_count}  CO_OCCURS_WITH edges: "
            f"{self.co_occur_edges}  Papers assigned: {self.papers_assigned}"
        )


def rebuild_co_occurrence(session: Any) -> int:
    """Delete and rebuild weighted Concept-Concept CO_OCCURS_WITH edges.

    Weight = number of distinct Papers that share both concepts via
    ``USES_METHOD`` or ``HAS_CHUNK → MENTIONS``.

    Returns:
        Number of ``CO_OCCURS_WITH`` relationships created.
    """
    session.run("MATCH ()-[r:CO_OCCURS_WITH]->() DELETE r")

    # Path 1: Paper -[:USES_METHOD]-> Concept pairs
    session.run(
        """
        MATCH (p:Paper)-[:USES_METHOD]->(c1:Concept)
        MATCH (p)-[:USES_METHOD]->(c2:Concept)
        WHERE elementId(c1) < elementId(c2)
        WITH c1, c2, count(DISTINCT p) AS weight
        MERGE (c1)-[r:CO_OCCURS_WITH]->(c2)
        SET r.weight = coalesce(r.weight, 0) + weight
        """
    )

    # Path 2: Paper -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Concept pairs
    session.run(
        """
        MATCH (p:Paper)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c1:Concept)
        MATCH (p)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c2:Concept)
        WHERE elementId(c1) < elementId(c2)
        WITH c1, c2, count(DISTINCT p) AS weight
        MERGE (c1)-[r:CO_OCCURS_WITH]->(c2)
        SET r.weight = coalesce(r.weight, 0) + weight
        """
    )

    result = session.run("MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS n")
    return int(result.single()["n"])


def cleanup_previous_clustering(session: Any) -> None:
    """Remove Community nodes, BELONGS_TO edges, and leidenCommunity properties."""
    session.run("MATCH (comm:Community) DETACH DELETE comm")
    session.run(
        """
        MATCH ()-[r:BELONGS_TO]->()
        DELETE r
        """
    )
    session.run(
        """
        MATCH (n)
        WHERE n.leidenCommunity IS NOT NULL
        REMOVE n.leidenCommunity
        """
    )


def drop_graph_if_exists(session: Any, name: str = GRAPH_NAME) -> None:
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists",
        name=name,
    ).single()["exists"]
    if exists:
        session.run(
            "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
            name=name,
        )


def project_concept_graph(session: Any, name: str = GRAPH_NAME) -> dict[str, Any]:
    result = session.run(
        """
        CALL gds.graph.project(
          $name,
          'Concept',
          {CO_OCCURS_WITH: {orientation: 'UNDIRECTED', properties: 'weight'}}
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        name=name,
    )
    return dict(result.single())


def run_leiden_write(
    session: Any, gamma: float, name: str = GRAPH_NAME
) -> dict[str, Any]:
    result = session.run(
        """
        CALL gds.leiden.write($name, {
          writeProperty: 'leidenCommunity',
          relationshipWeightProperty: 'weight',
          gamma: $gamma
        })
        YIELD communityCount, modularity, modularities, nodePropertiesWritten
        RETURN communityCount, modularity, modularities, nodePropertiesWritten
        """,
        name=name,
        gamma=gamma,
    )
    return dict(result.single())


def materialize_communities(session: Any) -> int:
    """Create Community nodes and Concept-[:BELONGS_TO]->Community links."""
    result = session.run(
        """
        MATCH (c:Concept)
        WHERE c.leidenCommunity IS NOT NULL
        WITH c.leidenCommunity AS lid, collect(c) AS concepts, count(*) AS size
        MERGE (comm:Community {leidenId: lid})
        ON CREATE SET comm.id = randomUUID()
        SET comm.size = size
        WITH comm, concepts
        UNWIND concepts AS c
        MERGE (c)-[:BELONGS_TO]->(comm)
        RETURN count(DISTINCT comm) AS n
        """
    )
    return int(result.single()["n"])


def cascade_papers_to_communities(session: Any) -> int:
    """Assign each Paper to the majority Concept community (tie: smallest leidenId)."""
    result = session.run(
        """
        MATCH (p:Paper)
        CALL {
          WITH p
          MATCH (p)-[:USES_METHOD]->(c:Concept)
          WHERE c.leidenCommunity IS NOT NULL
          RETURN c.leidenCommunity AS lid
          UNION ALL
          WITH p
          MATCH (p)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(c:Concept)
          WHERE c.leidenCommunity IS NOT NULL
          RETURN c.leidenCommunity AS lid
        }
        WITH p, lid, count(*) AS votes
        ORDER BY p, votes DESC, lid ASC
        WITH p, collect(lid)[0] AS winner
        WHERE winner IS NOT NULL
        MATCH (comm:Community {leidenId: winner})
        MERGE (p)-[:BELONGS_TO]->(comm)
        RETURN count(DISTINCT p) AS n
        """
    )
    return int(result.single()["n"])


def fetch_size_stats(session: Any) -> tuple[int, float, int, int]:
    """Return (min, median, max, community_count) from Community.size."""
    rows = list(
        session.run(
            """
            MATCH (comm:Community)
            RETURN comm.size AS size
            ORDER BY size
            """
        )
    )
    sizes = [int(r["size"]) for r in rows]
    if not sizes:
        return 0, 0.0, 0, 0
    return (
        min(sizes),
        float(statistics.median(sizes)),
        max(sizes),
        len(sizes),
    )


def count_concepts(session: Any) -> int:
    return int(session.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"])


def run_leiden(
    gamma: float | None = None,
    *,
    store: Neo4jGraphStore | None = None,
    config: Neo4jConfig | None = None,
) -> LeidenStats:
    """Run the full idempotent Leiden pipeline on Concept co-occurrence.

    Args:
        gamma: Leiden resolution parameter (default from env / 1.0).
        store: Optional existing Neo4j store (not closed by this function).
        config: Optional Neo4j config when creating a store.

    Returns:
        Summary statistics for the run.

    Raises:
        LeidenError: If Concepts/co-occurrence are insufficient or GDS fails.
    """
    resolved_gamma = leiden_gamma() if gamma is None else float(gamma)
    owns_store = store is None
    graph = store or Neo4jGraphStore(config=config or Neo4jConfig.from_env())
    driver = graph.connect()

    try:
        with driver.session() as session:
            # Ensure Community constraints exist
            graph.init_schema()

            concept_n = count_concepts(session)
            if concept_n < 2:
                raise LeidenError(
                    f"Need at least 2 Concept nodes for Leiden (found {concept_n}). "
                    "Link Concepts via USES_METHOD / MENTIONS before clustering."
                )

            edge_n = rebuild_co_occurrence(session)
            if edge_n == 0:
                raise LeidenError(
                    "No CO_OCCURS_WITH edges after rebuild. Concepts must share "
                    "papers via USES_METHOD or HAS_CHUNK→MENTIONS."
                )

            cleanup_previous_clustering(session)

            try:
                drop_graph_if_exists(session)
                project_concept_graph(session)
                leiden_result = run_leiden_write(session, resolved_gamma)
            except Exception as exc:  # noqa: BLE001
                raise LeidenError(
                    f"GDS Leiden failed (is the graph-data-science plugin loaded?): {exc}"
                ) from exc
            finally:
                try:
                    drop_graph_if_exists(session)
                except Exception:  # noqa: BLE001
                    pass

            materialize_communities(session)
            papers_assigned = cascade_papers_to_communities(session)
            size_min, size_median, size_max, community_count = fetch_size_stats(session)

            stats = LeidenStats(
                community_count=int(
                    leiden_result.get("communityCount", community_count)
                ),
                modularity=float(leiden_result.get("modularity") or 0.0),
                size_min=size_min,
                size_median=size_median,
                size_max=size_max,
                gamma=resolved_gamma,
                concept_count=concept_n,
                co_occur_edges=edge_n,
                papers_assigned=papers_assigned,
            )
            print(stats.format_summary())
            return stats
    finally:
        if owns_store:
            graph.close()
