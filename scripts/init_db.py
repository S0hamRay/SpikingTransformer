#!/usr/bin/env python3
"""Initialize Postgres + Neo4j academic Graph RAG schemas."""

from __future__ import annotations

from dotenv import load_dotenv

from db.neo4j_store import Neo4jGraphStore
from db.postgres_store import PostgresStore


def main() -> None:
    load_dotenv()
    print("Initializing Postgres schema...")
    pg = PostgresStore()
    try:
        pg.init_schema()
        print("  Postgres OK")
    finally:
        pg.close()

    print("Initializing Neo4j constraints/indexes...")
    graph = Neo4jGraphStore()
    try:
        graph.init_schema()
        print("  Neo4j OK")
    finally:
        graph.close()

    print("Done.")


if __name__ == "__main__":
    main()
