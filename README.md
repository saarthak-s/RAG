# DT Forecast Codebase Assistant

A retrieval-augmented generation (RAG) API that answers questions about its
own target codebase — grounded strictly in the actual source, not the LLM's
general knowledge.

It fetches a GitHub repository, chunks it intelligently (AST-aware for
Python, section-aware for Markdown), embeds the chunks into a local vector
store, and serves an `/ask` endpoint that retrieves relevant context and
asks an LLM to answer using only that context.

## How it works

```
build_db.py                          rag_engine.py                 main.py
┌─────────────────┐   embed    ┌──────────────────┐   retrieve   ┌──────────┐
│ Fetch repo files │ ─────────▶│    ChromaDB        │◀───────────│  /ask    │
│ Chunk by AST /   │            │ (persistent store) │             │ endpoint │
│ Markdown section │            └──────────────────┘             └──────────┘
└─────────────────┘                                                    │
                                                                        ▼
                                                              build prompt with
                                                              retrieved context
                                                                        │
                                                                        ▼
                                                                 Groq / Ollama
                                                                    LLM call
```

1. **`build_db.py`** pulls a fixed list of files from a GitHub repo, parses
   Python files with `ast` to extract functions/classes as individual
   chunks, splits Markdown by heading, and embeds everything into a
   persistent ChromaDB collection.
2. **`rag_engine.py`** retrieves the top-k relevant chunks for a query,
   builds a context-grounded prompt, and calls the configured LLM.
3. **`main.py`** exposes this as a FastAPI `POST /ask` endpoint.

## Features

- **AST-aware chunking** — Python source is split at the function/class
  level (via `ast.walk`) instead of naive line-based splitting, so retrieved
  chunks are always complete, syntactically valid units.
- **Swappable LLM backend** — `LLM_PROVIDER=groq` for the hosted API,
  `LLM_PROVIDER=ollama` for a local model with no rate limits, useful for
  iterating without burning API quota.
- **Disk-backed response cache** — identical prompts are served from
  `.llm_cache/` instead of re-calling the API, so repeating a question while
  developing costs nothing after the first call.
- **Persistent RPM/TPM rate limiter** — tracks request and token budgets in
  `.ratelimit.json`, calibrating itself from the API's own rate-limit
  headers so it stays accurate across restarts.
- **Retry with exponential backoff** — LLM calls are wrapped with `tenacity`
  to absorb transient API failures.

## Project structure

```
.
├── src/rag/
│   ├── build_db.py      # Fetch + chunk + embed the target repo
│   ├── rag_engine.py     # Retrieval, prompting, caching, rate limiting
│   ├── main.py            # FastAPI app exposing POST /ask
│   └── ask_api.py         # Minimal example client
├── chroma_db/             # Persistent vector store (generated)
├── .llm_cache/             # Cached LLM responses (generated)
├── pyproject.toml
└── uv.lock
```

## Getting started

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your-groq-api-key      # required if LLM_PROVIDER=groq
LLM_PROVIDER=groq                    # groq (default) or ollama
LLM_MODEL=openai/gpt-oss-20b         # optional override
RATE_LIMIT_RPM=30                    # optional, requests per minute
RATE_LIMIT_TPM=80000                 # optional, tokens per minute
```

To use a local model instead, set `LLM_PROVIDER=ollama` and make sure
[Ollama](https://ollama.com/) is running locally with your chosen model
pulled (default: `llama3.1:8b`).

### 3. Build the vector store

Edit `REPO_OWNER` / `REPO_NAME` / `REPO_FILES` in `build_db.py` to point at
the repository you want the assistant to answer questions about, then run:

```bash
uv run python src/rag/build_db.py
```

This fetches each file, chunks it, and upserts the chunks into a local
ChromaDB collection at `./chroma_db`. Safe to re-run — it overwrites
existing chunks by ID rather than duplicating them.

### 4. Run the API

```bash
uv run uvicorn rag.main:app --reload --port 8000
```

### 5. Ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "which decision tree model is used?"}'
```

Or use the included example client:

```bash
uv run python src/rag/ask_api.py
```

Response shape:

```json
{
  "answer": "The repository uses a DecisionTreeRegressor from scikit-learn...",
  "sources": ["train_dt_forecast_model.py (train_model)", "README.md (Section: Model)"]
}
```

## Design notes

This started as a learning project for RAG fundamentals (chunking
strategy, embeddings, retrieval, grounded prompting) and grew to include a
few things that come up once you actually run an LLM-backed service against
a rate-limited API: caching to avoid redundant calls during development, a
rate limiter that syncs against real API headers instead of guessing, and
retries for transient failures. Swapping `LLM_PROVIDER` to `ollama` removes
the need for any of that during local iteration.

## Possible extensions

- Streaming responses from the `/ask` endpoint
- Re-ranking retrieved chunks before prompting
- Support for indexing multiple repos into separate collections
- Basic auth / API key on the FastAPI endpoint