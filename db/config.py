"""Shared database configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PostgresConfig:
    host: str = "localhost"
    port: int = 5433
    user: str = "rag"
    password: str = "rag"
    database: str = "academic_rag"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            user=os.getenv("POSTGRES_USER", "rag"),
            password=os.getenv("POSTGRES_PASSWORD", "rag"),
            database=os.getenv("POSTGRES_DB", "academic_rag"),
        )

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "ragpassword"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "ragpassword"),
        )


def sync_enabled() -> bool:
    """Whether ingested documents should be written to Postgres/Neo4j."""
    return os.getenv("RAG_SYNC_GRAPH_DB", "true").lower() in {"1", "true", "yes"}


def leiden_gamma() -> float:
    """Leiden resolution (gamma) parameter; default 1.0."""
    return float(os.getenv("LEIDEN_GAMMA", "1.0"))
