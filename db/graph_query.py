"""Read-only traversal of the Paper/Author/Concept Neo4j graph.

Accepts either Cypher or natural-language questions. Natural language is
translated to a read-only Cypher query via the local Ollama chat model.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from db.neo4j_store import Neo4jGraphStore

DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Leading keywords that indicate the input is already Cypher.
_CYPHER_START = re.compile(
    r"^\s*(MATCH|OPTIONAL\s+MATCH|CALL|WITH|UNWIND|RETURN|SHOW|PROFILE|EXPLAIN)\b",
    re.IGNORECASE,
)

# Mutating / admin clauses — rejected for both raw Cypher and LLM output.
_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|"
    r"FOREACH|APOC\.|gds\.\w+\.(write|mutate))\b",
    re.IGNORECASE,
)

_SCHEMA_HINT = """
Node labels: Paper, Author, Chunk, Concept, Venue, Dataset, Institution, Community
Relationships:
  Paper-[:AUTHORED_BY]->Author
  Paper-[:PUBLISHED_IN]->Venue
  Paper-[:CITES]->Paper
  Paper-[:HAS_CHUNK]->Chunk
  Chunk-[:MENTIONS]->Concept
  Paper-[:USES_METHOD]->Concept
  Paper-[:EVALUATES_ON]->Dataset
  Paper-[:REPORTS_METRIC]->Concept  (property: value)
  Paper-[:EXTENDS]->Paper
  Concept-[:RELATED_TO]->Concept
  Author-[:AFFILIATED_WITH]->Institution
  Concept-[:CO_OCCURS_WITH {weight}]->Concept
  Concept-[:BELONGS_TO]->Community
  Paper-[:BELONGS_TO]->Community
Paper properties: id, title, abstract, year, doi, arxiv_id, url, citation_count, source_filename
Author properties: id, name, orcid
Concept properties: id, name, type
Community properties: leidenId, size
""".strip()

_NL_TO_CYPHER_PROMPT = """You translate questions about an academic paper knowledge graph into Neo4j Cypher.

Rules:
- Output ONLY a single read-only Cypher query (MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN / CALL).
- Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or any write procedure.
- Limit results to 25 rows unless the user asks for a count.
- Use the schema below.

Schema:
{schema}

Question: {question}

Cypher:"""


class GraphQueryError(ValueError):
    """Raised when a graph query is invalid or unsafe."""


def looks_like_cypher(text: str) -> bool:
    """Return True if ``text`` appears to be a Cypher statement."""
    return bool(_CYPHER_START.search(text or ""))


def assert_read_only(cypher: str) -> str:
    """Validate and return a Cypher string that only reads data."""
    cleaned = (cypher or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:cypher)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    if not cleaned:
        raise GraphQueryError("Empty Cypher query.")
    if _WRITE_CLAUSE.search(cleaned):
        raise GraphQueryError(
            "Only read-only Cypher is allowed "
            "(no CREATE/MERGE/DELETE/SET/REMOVE/write procedures)."
        )
    return cleaned


def _chat_model():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        temperature=0.0,
    )


def natural_language_to_cypher(question: str) -> str:
    """Translate a natural-language graph question into read-only Cypher."""
    question = (question or "").strip()
    if not question:
        raise GraphQueryError("Empty question.")
    prompt = _NL_TO_CYPHER_PROMPT.format(schema=_SCHEMA_HINT, question=question)
    try:
        response = _chat_model().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GraphQueryError(
            f"Failed to translate natural language to Cypher: {exc}. "
            "Is Ollama running?"
        ) from exc
    return assert_read_only(str(text))


def _serialize_value(value: Any) -> Any:
    """Convert Neo4j driver values into JSON-friendly Python types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    # neo4j.graph.Node / Relationship / Path and temporal types
    if hasattr(value, "items") and callable(value.items):
        try:
            return {
                "_labels": list(getattr(value, "labels", [])),
                **{k: _serialize_value(v) for k, v in value.items()},
            }
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "type") and hasattr(value, "items"):
        try:
            return {
                "_type": value.type,
                **{k: _serialize_value(v) for k, v in value.items()},
            }
        except Exception:  # noqa: BLE001
            pass
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def execute_cypher(
    cypher: str,
    *,
    store: Neo4jGraphStore | None = None,
    parameters: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Run a read-only Cypher query and return row dicts."""
    query = assert_read_only(cypher)
    owned = store is None
    store = store or Neo4jGraphStore()
    rows: list[dict[str, Any]] = []
    try:
        with store.connect().session() as session:
            result = session.run(query, parameters or {})
            for i, record in enumerate(result):
                if i >= limit:
                    break
                rows.append({k: _serialize_value(record[k]) for k in record.keys()})
    finally:
        if owned:
            store.close()
    return rows


def run_graph_query(
    cypher_or_natural_language: str,
    *,
    store: Neo4jGraphStore | None = None,
) -> dict[str, Any]:
    """Traverse the academic graph with Cypher or natural language.

    Returns:
        Dict with ``query`` (executed Cypher), ``source`` (``"cypher"`` or
        ``"natural_language"``), and ``results`` (list of row dicts).
    """
    text = (cypher_or_natural_language or "").strip()
    if not text:
        raise GraphQueryError("Provide Cypher or a natural-language question.")

    if looks_like_cypher(text):
        cypher = assert_read_only(text)
        source = "cypher"
    else:
        cypher = natural_language_to_cypher(text)
        source = "natural_language"

    results = execute_cypher(cypher, store=store)
    return {"query": cypher, "source": source, "results": results}
