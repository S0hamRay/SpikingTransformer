# Spiking Transformer Language Model

A minimal chatbot with options to use addition only spiking self attention (faster training and fine tuning on neuromorphic hardware) or regular self attention. Persists conversation history via JSON files, supports web interface via Gradio, supports voice to text via SpeechRecognition.

## RAG mode (LangGraph + Ollama + Tavily CRAG)

The Gradio UI includes a third **RAG** chat mode. Upload `.txt` or `.pdf` files, index them into a local Chroma vector store, and ask questions answered by a local Ollama model via a LangGraph **corrective RAG** pipeline:

`retrieve → grade documents → generate` (or, if retrieval is weak, `rewrite query → Tavily web search → generate`).

When Postgres and Neo4j are running, ingested documents are also written into an academic paper graph (Paper / Author / Chunk / …) for future Graph RAG.

### Setup

1. Install [Ollama](https://ollama.com) and start it (`ollama serve` or open the app).
2. Pull the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`. For corrective web search, set:
   ```bash
   TAVILY_API_KEY=tvly-...
   ```
   Get a key at [tavily.com](https://tavily.com). Without it, RAG still works from uploaded docs; web-search correction is disabled.

### Run the API (engines + Gradio + MCP)

Chat, RAG, and graph engines are exposed through a FastAPI service. The Gradio
UI is mounted as a sub-app (not a standalone Gradio server). Graph RAG routes
are also published as MCP tools via [fastapi-mcp](https://github.com/tadata-org/fastapi_mcp):

```bash
python -m api.server
# or: python app.py
```

| Surface | URL |
|---------|-----|
| Gradio UI | http://127.0.0.1:8000/ |
| OpenAPI docs | http://127.0.0.1:8000/docs |
| MCP server | http://127.0.0.1:8000/mcp |

**MCP tools** (auto-generated from FastAPI `operation_id`s):

| Tool | Behavior |
|------|----------|
| `rag_query` | Corrective RAG answer for a question |
| `rag_ingest` | Index a `.txt` / `.pdf` paper by filesystem path |
| `graph_query` | Traverse Paper/Author/Concept (Cypher or natural language) |
| `run_community_detection` | Run Leiden; return modularity / community stats |

Example HTTP calls:

```bash
curl -X POST http://127.0.0.1:8000/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"file_path":"/path/to/paper.pdf"}'

curl -X POST http://127.0.0.1:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What attention variants are discussed?"}'

curl -X POST http://127.0.0.1:8000/graph/query \
  -H 'Content-Type: application/json' \
  -d '{"cypher_or_natural_language":"MATCH (p:Paper)-[:USES_METHOD]->(c:Concept) RETURN p.title, c.name LIMIT 10"}'

curl -X POST http://127.0.0.1:8000/graph/community-detection \
  -H 'Content-Type: application/json' \
  -d '{"gamma":1.5}'
```

In the Gradio UI (`/`):

1. Select **RAG** from the **Chat mode** dropdown.
2. Upload one or more `.txt` / `.pdf` files and click **Index documents**.
3. Ask questions about the uploaded content.

RAG session data (vectors and chat history) is stored under `rag_data/<session_id>/`.

### Academic Graph DB (Postgres + Neo4j)

Start the databases:

```bash
docker compose up -d
python scripts/init_db.py
```

- **Postgres** (`localhost:5433`): structured source of truth (`db/schema.sql`; host port 5433 avoids clashing with a local Postgres on 5432)
- **Neo4j** Browser: `http://localhost:7474` (bolt `7687`) — graph for Graph RAG traversal

Node types: `Paper`, `Author`, `Chunk`, `Concept`, `Venue`, `Dataset`, `Institution`

Key relationships: `AUTHORED_BY`, `PUBLISHED_IN`, `CITES`, `HAS_CHUNK`, `MENTIONS`, `USES_METHOD`, `EVALUATES_ON`, `REPORTS_METRIC`, `EXTENDS`, `RELATED_TO`, `AFFILIATED_WITH`, `CO_OCCURS_WITH`, `BELONGS_TO`

Set `RAG_SYNC_GRAPH_DB=false` in `.env` to keep RAG local-only (Chroma) without writing to the DBs.

### Community detection (Leiden / GDS)

Neo4j must load the Graph Data Science plugin (already configured in `docker-compose.yml`). After changing compose, recreate Neo4j:

```bash
docker compose up -d neo4j
python scripts/init_db.py
```

On ingest, basic entity extraction (lexicon + metric regex in `db/entity_extraction.py`) links Concepts via `USES_METHOD` and/or `HAS_CHUNK → MENTIONS`. Then:

```bash
python scripts/run_leiden.py
# or tune granularity:
python scripts/run_leiden.py --gamma 1.5
```

The script:

1. Rebuilds weighted `Concept-[:CO_OCCURS_WITH]->Concept` edges
2. Runs `gds.leiden.write` (idempotent: replaces previous `Community` nodes)
3. Creates `(:Concept)-[:BELONGS_TO]->(:Community)` and cascades Papers by majority vote
4. Prints community count, size min/median/max, and modularity

Override the default resolution with `LEIDEN_GAMMA` in `.env`.
