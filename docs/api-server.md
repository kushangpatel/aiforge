# API Server

AIForge ships a production-ready **REST API** that wraps every SDK module as an
HTTP endpoint. Deploy it as a standalone service, a Docker container, or a
sidecar — any language that can make HTTP requests can use AIForge.

---

## Table of Contents

- [API Server](#api-server)
  - [Table of Contents](#table-of-contents)
  - [Why Use the API Server?](#why-use-the-api-server)
  - [SDK vs HTTP API — When to Use Which](#sdk-vs-http-api--when-to-use-which)
  - [Getting Started](#getting-started)
    - [Install](#install)
    - [Start the Server](#start-the-server)
    - [CLI Options](#cli-options)
    - [Quick Smoke Test](#quick-smoke-test)
  - [Authentication](#authentication)
  - [API Reference](#api-reference)
    - [Endpoint Summary](#endpoint-summary)
    - [Health \& Metrics](#health--metrics)
      - [`GET /health`](#get-health)
      - [`GET /health/ready`](#get-healthready)
      - [`GET /stats`](#get-stats)
    - [`POST /summarize`](#post-summarize)
    - [Document Q\&A](#document-qa)
      - [`POST /qa/ingest`](#post-qaingest)
      - [`POST /qa/ask`](#post-qaask)
      - [`GET /qa/collections`](#get-qacollections)
      - [`GET /qa/collections/{name}/stats`](#get-qacollectionsnamestats)
      - [`DELETE /qa/collections/{name}`](#delete-qacollectionsname)
    - [`POST /extract`](#post-extract)
    - [`POST /classify`](#post-classify)
    - [Semantic Search](#semantic-search)
      - [`POST /search/index`](#post-searchindex)
      - [`POST /search/query`](#post-searchquery)
    - [`POST /generate`](#post-generate)
    - [Chat](#chat)
      - [`POST /chat`](#post-chat)
      - [`POST /chat/reset`](#post-chatreset)
      - [`GET /chat/conversations`](#get-chatconversations)
    - [`POST /generate/rewrite`](#post-generaterewrite)
    - [`POST /translate`](#post-translate)
    - [`POST /vision`](#post-vision)
    - [`POST /classify/sentiment`](#post-classifysentiment)
    - [`POST /moderate`](#post-moderate)
    - [`POST /pipeline`](#post-pipeline)
    - [`GET /jobs/{job_id}`](#get-jobsjob_id)
  - [Async Jobs \& Webhooks](#async-jobs--webhooks)
    - [Flow](#flow)
    - [Webhook Payload](#webhook-payload)
    - [Webhook Delivery](#webhook-delivery)
    - [SSRF Protection](#ssrf-protection)
    - [Webhook Configuration](#webhook-configuration)
  - [Integration Examples](#integration-examples)
    - [Node.js / TypeScript](#nodejs--typescript)
    - [Go](#go)
    - [Java](#java)
  - [CORS](#cors)
  - [Docker Deployment](#docker-deployment)
    - [Dockerfile](#dockerfile)
    - [Docker Compose](#docker-compose)
    - [Run with Docker](#run-with-docker)
  - [Environment Variables](#environment-variables)
  - [Graceful Shutdown](#graceful-shutdown)
  - [See Also](#see-also)

---

## Why Use the API Server?

The AIForge API server lets you:

- **Use AI from any language** — Node.js, Go, Java, Ruby, C#, shell scripts, or anything that speaks HTTP.
- **Centralise configuration** — One deployment, one `aiforge.yaml`, shared provider keys and cost tracking across all consumers.
- **Scale independently** — Deploy AIForge as its own container / sidecar and scale it horizontally behind a load balancer.
- **Enforce access control** — Bearer-token authentication protects all endpoints; public health probes remain open.
- **Observe everything** — Every response includes `cost_usd`, `input_tokens`, `output_tokens`, `latency_ms`, and a `trace_id` / `X-Request-Id` header for distributed tracing.
- **Go async when needed** — Long-running ingestion / indexing operations can return immediately with a `job_id` and notify your service via webhook on completion.

---

## SDK vs HTTP API — When to Use Which

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Python service / script | **SDK** (`from aiforge import …`) | Zero network overhead, full type safety, direct access to response objects |
| Non-Python service (Node, Go, Java, etc.) | **HTTP API** | Language-agnostic JSON over HTTP |
| Microservice / sidecar architecture | **HTTP API** | Deploy AIForge independently, scale separately |
| Jupyter notebooks / quick experiments | **SDK** | Fastest feedback loop, inline results |
| Streamlit / Playground prototyping | **SDK** (built-in) | `aiforge playground` already uses the SDK internally |
| CI/CD pipeline step | Either | SDK if the pipeline is Python; HTTP if it's a shell/curl step |
| Multiple consumers (frontend + backend) | **HTTP API** | Single deployment, shared config and cost tracking |
| Fine-grained control (streaming callbacks, custom retry) | **SDK** | Direct access to LangChain internals and module options |

**Rule of thumb:** if your code is Python and runs in the same process, use the SDK. If anything else needs to call AIForge — or you want a single shared deployment — use the HTTP API.

---

## Getting Started

### Install

```bash
pip install aiforge[server]       # includes uvicorn
```

### Start the Server

```bash
# Local development (no auth)
aiforge serve

# Bind to specific host/port with hot-reload
aiforge serve --host 0.0.0.0 --port 8000 --reload

# Production (with auth)
AIFORGE_API_KEY=your-secret-key aiforge serve --host 0.0.0.0 --port 8000
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--reload` | `false` | Enable hot-reload (development only) |

Once started, interactive Swagger docs are available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

### Quick Smoke Test

```bash
# Health check
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}

# Summarise some text
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "AIForge is a plug-and-play AI toolkit that wraps LangChain into task-specific modules.", "style": "concise"}'
```

---

## Authentication

When the `AIFORGE_API_KEY` environment variable is set (non-empty), **all** endpoints except the public paths require a `Bearer` token.

**Public paths (never require auth):**

| Path | Purpose |
|------|---------|
| `GET /health` | Liveness probe |
| `GET /health/ready` | Readiness probe |
| `GET /docs` | Swagger UI |
| `GET /openapi.json` | OpenAPI spec |
| `GET /redoc` | ReDoc viewer |

**Authenticated request:**

```bash
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "labels": ["bug", "feature"]}'
```

**Error responses:**

| Status | Body | When |
|--------|------|------|
| `401` | `{"detail": "Missing or invalid Authorization header"}` | Header absent or malformed |
| `401` | `{"detail": "Invalid API key"}` | Token does not match `AIFORGE_API_KEY` |

Token comparison uses `secrets.compare_digest()` to prevent timing attacks. When `AIFORGE_API_KEY` is **not set** (default for local dev), auth is disabled and all endpoints are open.

---

## API Reference

Every module endpoint follows the same conventions:

- **Content-Type:** `application/json`
- **Cost fields:** All responses include `cost_usd`, `input_tokens`, `output_tokens`, `latency_ms`, and `model` (where applicable).
- **Request IDs:** Every response includes an `X-Request-Id` header (UUID) for tracing.
- **Error codes:** `422` for validation errors, `429` for rate limits, `503` for provider errors, `500` for unexpected failures.

### Endpoint Summary

| Method | Endpoint | Module | Description |
|--------|----------|--------|-------------|
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/health/ready` | — | Readiness probe (validates provider connectivity) |
| `GET` | `/stats` | — | Aggregated cost / token / call-count metrics |
| `POST` | `/summarize` | Summarizer | Summarise text |
| `POST` | `/qa/ingest` | DocQA | Ingest documents into a collection |
| `POST` | `/qa/ask` | DocQA | Ask a question against a collection |
| `GET` | `/qa/collections` | DocQA | List all collections |
| `GET` | `/qa/collections/{name}/stats` | DocQA | Collection statistics |
| `DELETE` | `/qa/collections/{name}` | DocQA | Delete a collection |
| `POST` | `/extract` | Extractor | Extract structured data from text |
| `POST` | `/classify` | Classifier | Classify text against a taxonomy |
| `POST` | `/search/index` | SemanticSearch | Index documents for search |
| `POST` | `/search/query` | SemanticSearch | Semantic similarity search |
| `POST` | `/generate` | Generator | Generate content (template or freeform) |
| `POST` | `/generate/rewrite` | Generator | Rewrite text with instructions/tone/reading-level |
| `POST` | `/translate` | Translator | Translate text between languages |
| `POST` | `/vision` | Vision | Analyze images (describe, Q&A, extract, classify) |
| `POST` | `/chat` | ChatAgent | Send a chat message (multi-turn) |
| `POST` | `/chat/reset` | ChatAgent | Reset a conversation |
| `GET` | `/chat/conversations` | ChatAgent | List active conversation IDs |
| `POST` | `/moderate` | Moderator | Content moderation (PII, toxicity, prompt injection) |
| `POST` | `/classify/sentiment` | Classifier | Sentiment analysis with optional aspect breakdown |
| `POST` | `/pipeline` | Pipeline | Execute a multi-step pipeline |
| `GET` | `/jobs/{job_id}` | — | Poll async job status |
| `GET` | `/docs` | — | Swagger UI |
| `GET` | `/redoc` | — | ReDoc UI |

---

### Health & Metrics

#### `GET /health`

Liveness probe — returns immediately with no dependency checks.

**Response `200`:**

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### `GET /health/ready`

Readiness probe — validates that the configured LLM provider is reachable.

**Response `200`:**

```json
{
  "status": "ready",
  "provider": "openai",
  "version": "0.1.0"
}
```

**Response `503` (provider unreachable):**

```json
{
  "status": "not_ready",
  "provider": "openai",
  "version": "0.1.0",
  "detail": "Connection refused"
}
```

#### `GET /stats`

Aggregated cost, token, and call-count metrics since the server started.

**Response `200`:**

```json
{
  "total_cost_usd": 1.2345,
  "total_tokens": 84200,
  "call_count": 312,
  "by_module": {
    "Summarizer": 120,
    "Classifier": 95,
    "ChatAgent": 54,
    "DocQA": 43
  },
  "by_model": {
    "gpt-4o": 200,
    "gpt-4o-mini": 112
  }
}
```

---

### `POST /summarize`

Summarise text using one of four styles. Strategy (`stuff` or `map_reduce`) is auto-selected based on input length, or can be forced.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to summarise |
| `style` | `"concise" \| "detailed" \| "bullet_points" \| "executive"` | `"concise"` | Summarisation style |
| `max_length` | `integer` (≥ 1) | `500` | Target summary length (tokens) |
| `strategy` | `"stuff" \| "map_reduce" \| null` | `null` | Force a strategy; `null` = auto-detect |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `string` | Generated summary |
| `strategy_used` | `"stuff" \| "map_reduce"` | Strategy that was applied |
| `chunks_processed` | `integer` | Number of chunks processed (1 for stuff) |
| `cost_usd` | `float` | Total cost of the operation |
| `input_tokens` | `integer` | Input tokens consumed |
| `output_tokens` | `integer` | Output tokens generated |
| `model` | `string` | Model used (e.g. `"gpt-4o"`) |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed how businesses operate...",
    "style": "bullet_points",
    "max_length": 300
  }'
```

```json
{
  "summary": "- AI has transformed business operations...\n- Key areas include...",
  "strategy_used": "stuff",
  "chunks_processed": 1,
  "cost_usd": 0.0012,
  "input_tokens": 245,
  "output_tokens": 89,
  "model": "gpt-4o",
  "latency_ms": 1230.5
}
```

**Errors:** `422` (validation), `429` (rate limit), `503` (provider error)

---

### Document Q&A

#### `POST /qa/ingest`

Ingest files or directories into a vector collection. Supports 30+ text file formats (.py, .js, .md, .txt, .html, .json, .yaml, etc.). Automatically skips binary files, `node_modules`, `.git`, and other non-text content.

Supports **async execution** — pass `webhook_url` to receive a `202 Accepted` response and get notified on completion.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `string` | *(required)* | File path, directory path, or git URL |
| `collection` | `string` | `"default"` | Target collection name |
| `file_extensions` | `string[] \| null` | `null` | Filter to specific extensions (e.g. `[".py", ".md"]`) |
| `batch_size` | `integer` (≥ 1) | `100` | Embedding batch size |
| `mode` | `"full" \| "incremental"` | `"full"` | Ingest strategy (`incremental` purges stale chunks for updated/deleted files) |
| `webhook_url` | `string \| null` | `null` | Webhook URL for async mode (HTTPS required in production) |

**Response `200` (synchronous):**

| Field | Type | Description |
|-------|------|-------------|
| `chunks_indexed` | `integer` | Total document chunks indexed |
| `files_processed` | `integer` | Number of files processed |
| `languages` | `object` | Language → count breakdown (e.g. `{"python": 12, "markdown": 5}`) |
| `skipped_files` | `string[]` | Files that failed to process |
| `collection` | `string` | Collection name |
| `cost_usd` | `float` | Embedding cost |
| `input_tokens` | `integer` | Embedding tokens consumed |
| `model` | `string` | Embedding model used |
| `latency_ms` | `float` | End-to-end latency |
| `files_added` | `integer` | Incremental mode: new files |
| `files_updated` | `integer` | Incremental mode: updated files re-indexed |
| `files_unchanged` | `integer` | Incremental mode: unchanged files skipped |
| `files_deleted` | `integer` | Incremental mode: files removed from source scope |
| `chunks_deleted` | `integer` | Incremental mode: stale/orphan chunks deleted (best-effort on Pinecone) |

**Response `202` (async with webhook):**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "accepted"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/qa/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "./src/",
    "collection": "codebase",
    "file_extensions": [".py", ".md"],
    "batch_size": 50,
    "mode": "incremental"
  }'
```

```json
{
  "chunks_indexed": 342,
  "files_processed": 28,
  "languages": {"python": 22, "markdown": 6},
  "skipped_files": [],
  "collection": "codebase",
  "cost_usd": 0.0089,
  "input_tokens": 44500,
  "model": "text-embedding-3-small",
  "latency_ms": 8420.1,
  "files_added": 0,
  "files_updated": 3,
  "files_unchanged": 25,
  "files_deleted": 1,
  "chunks_deleted": 42
}
```

**Errors:** `404` (source not found), `422` (validation or invalid webhook URL)

---

#### `POST /qa/ask`

Ask a natural-language question against an ingested collection. Uses a LangGraph agent loop that retrieves relevant chunks, reasons over them, and may iterate up to `max_iterations` times.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `string` | *(required)* | Question to ask |
| `collection` | `string` | `"default"` | Collection to query |
| `filter_repo` | `string \| null` | `null` | Filter results to a specific repository / source |
| `max_iterations` | `integer` (1–10) | `3` | Maximum agent reasoning iterations |
| `include_trace` | `boolean` | `false` | Include step-by-step agent trace in response |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Answer text |
| `sources` | `string[]` | Source document references |
| `confidence` | `float` | Answer confidence score (0.0–1.0) |
| `iterations` | `integer` | Iterations actually used |
| `trace` | `object[] \| null` | Agent step trace (only if `include_trace=true`) |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `cost_usd` | `float` | Total cost |
| `model` | `string` | Model used |
| `trace_id` | `string` | Distributed trace ID |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in the API server?",
    "collection": "codebase",
    "max_iterations": 5,
    "include_trace": true
  }'
```

```json
{
  "text": "Authentication is handled by the AuthMiddleware class...",
  "sources": ["aiforge/server/middleware.py", "docs/api-auth.md"],
  "confidence": 0.92,
  "iterations": 2,
  "trace": [
    {"step": "classify", "output": "needs_retrieval"},
    {"step": "retrieve", "output": "found 4 chunks"},
    {"step": "reason", "output": "sufficient context"},
    {"step": "respond", "output": "generated answer"}
  ],
  "input_tokens": 1820,
  "output_tokens": 310,
  "cost_usd": 0.0045,
  "model": "gpt-4o",
  "trace_id": "abc-123-def",
  "latency_ms": 3210.8
}
```

---

#### `GET /qa/collections`

List all existing vector collections.

**Response `200`:**

```json
{
  "collections": ["codebase", "docs", "support-tickets"]
}
```

---

#### `GET /qa/collections/{name}/stats`

Get statistics for a specific collection.

**Path Parameters:** `name` — collection name

**Response `200`:**

```json
{
  "name": "codebase",
  "chunks": 342,
  "persist_dir": "/app/aiforge_data/chroma"
}
```

---

#### `DELETE /qa/collections/{name}`

Delete a collection and all its data.

**Path Parameters:** `name` — collection name

**Response `200`:**

```json
{
  "status": "deleted",
  "collection": "codebase"
}
```

---

### `POST /extract`

Extract structured data from unstructured text using a JSON Schema definition. Supports optional few-shot examples and automatic retry on schema validation failure.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to extract from |
| `output_schema` | `object` | *(required)* | JSON Schema defining the desired output structure |
| `examples` | `Example[]` | `[]` | Few-shot examples |

Each **Example** object:

| Field | Type | Description |
|-------|------|-------------|
| `input` | `string` | Example input text |
| `output` | `object` | Expected extraction result |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `data` | `object` | Extracted structured data |
| `raw_json` | `string` | Raw JSON string from the LLM |
| `validation_passed` | `boolean` | Whether output conforms to the schema |
| `retry_count` | `integer` | Number of retries needed |
| `cost_usd` | `float` | Total cost |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `model` | `string` | Model used |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Smith is the CTO of Acme Corp. He can be reached at john@acme.com or 555-0123.",
    "output_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "company": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"}
      },
      "required": ["name", "company"]
    }
  }'
```

```json
{
  "data": {
    "name": "John Smith",
    "title": "CTO",
    "company": "Acme Corp",
    "email": "john@acme.com",
    "phone": "555-0123"
  },
  "raw_json": "{\"name\":\"John Smith\",...}",
  "validation_passed": true,
  "retry_count": 0,
  "cost_usd": 0.0018,
  "input_tokens": 320,
  "output_tokens": 85,
  "model": "gpt-4o",
  "latency_ms": 1450.2
}
```

**Errors:** `422` (validation / schema error), `429` (rate limit), `503` (provider error)

---

### `POST /classify`

Classify text against a provided taxonomy of labels. Supports single-label and multi-label modes, plus optional few-shot examples.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to classify |
| `labels` | `string[]` (1–100) | *(required)* | Taxonomy labels |
| `examples` | `Example[]` | `[]` | Few-shot examples |
| `multi_label` | `boolean` | `false` | Allow multiple labels |

Each **Example** object:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Example text |
| `label` | `string` | Example label (must exist in `labels`) |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | `string \| null` | Top label (`null` when `multi_label=true`) |
| `scores` | `LabelScore[]` | All label scores sorted by confidence |
| `rationale` | `string` | LLM reasoning for the classification |
| `cost_usd` | `float` | Total cost |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `latency_ms` | `float` | End-to-end latency |

Each **LabelScore** object:

| Field | Type | Description |
|-------|------|-------------|
| `label` | `string` | Label name |
| `confidence` | `float` | Confidence 0.0–1.0 |

**Example:**

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The login button is broken on mobile Safari",
    "labels": ["bug", "feature_request", "question", "documentation"],
    "examples": [
      {"text": "App crashes on startup", "label": "bug"},
      {"text": "Can you add dark mode?", "label": "feature_request"}
    ]
  }'
```

```json
{
  "label": "bug",
  "scores": [
    {"label": "bug", "confidence": 0.94},
    {"label": "feature_request", "confidence": 0.03},
    {"label": "question", "confidence": 0.02},
    {"label": "documentation", "confidence": 0.01}
  ],
  "rationale": "The text describes a broken UI element which is a software defect.",
  "cost_usd": 0.0008,
  "input_tokens": 180,
  "output_tokens": 65,
  "latency_ms": 890.3
}
```

**Errors:** `422` (validation — empty labels, example label not in taxonomy)

---

### Semantic Search

#### `POST /search/index`

Index documents for semantic similarity search. Uses embedding models only (no LLM calls). Documents are upserted — re-indexing the same `id` updates the existing entry.

Supports **async execution** via `webhook_url`.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `documents` | `Document[]` | *(required, non-empty)* | Documents to index |
| `collection` | `string` | `"default"` | Target collection |
| `webhook_url` | `string \| null` | `null` | Webhook URL for async mode |

Each **Document** object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `string` | *(required)* | Unique document identifier |
| `text` | `string` | *(required)* | Content to embed |
| `metadata` | `object` | `{}` | Arbitrary metadata for filtering |

**Response `200` (synchronous):**

| Field | Type | Description |
|-------|------|-------------|
| `indexed_count` | `integer` | Documents indexed in this request |
| `collection` | `string` | Collection name |
| `total_documents` | `integer` | Total documents now in the collection |
| `cost_usd` | `float` | Embedding cost |
| `input_tokens` | `integer` | Embedding tokens consumed |
| `latency_ms` | `float` | End-to-end latency |

**Response `202` (async with webhook):**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "accepted"
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/search/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc-1", "text": "AIForge supports 8 task-specific modules.", "metadata": {"source": "readme"}},
      {"id": "doc-2", "text": "The API server exposes all modules over HTTP.", "metadata": {"source": "docs"}}
    ],
    "collection": "knowledge-base"
  }'
```

```json
{
  "indexed_count": 2,
  "collection": "knowledge-base",
  "total_documents": 2,
  "cost_usd": 0.0001,
  "input_tokens": 42,
  "latency_ms": 320.5
}
```

---

#### `POST /search/query`

Search for semantically similar documents. Uses cosine similarity (`1.0 - distance`). Supports ChromaDB metadata filters (`$lt`, `$gte`, `$and`, `$or`, etc.).

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `string` | *(required)* | Search query |
| `collection` | `string` | `"default"` | Collection to search |
| `top_k` | `integer` (1–100) | `5` | Number of results to return |
| `filter` | `object \| null` | `null` | Metadata filter (ChromaDB `where` clause) |
| `include_embeddings` | `boolean` | `false` | Include raw embedding vectors |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `hits` | `Hit[]` | Ranked search results |
| `query` | `string` | The original query |
| `total_searched` | `integer` | Total documents in the collection |
| `cost_usd` | `float` | Embedding cost |
| `input_tokens` | `integer` | Embedding tokens |
| `latency_ms` | `float` | End-to-end latency |

Each **Hit** object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Document ID |
| `text` | `string` | Document text |
| `score` | `float` | Similarity score (0.0–1.0, higher = more similar) |
| `metadata` | `object` | Document metadata |
| `embeddings` | `float[] \| null` | Raw embedding vector (only if `include_embeddings=true`) |

**Example:**

```bash
curl -X POST http://localhost:8000/search/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does the API server work?",
    "collection": "knowledge-base",
    "top_k": 3,
    "filter": {"source": "docs"}
  }'
```

```json
{
  "hits": [
    {
      "id": "doc-2",
      "text": "The API server exposes all modules over HTTP.",
      "score": 0.91,
      "metadata": {"source": "docs"},
      "embeddings": null
    }
  ],
  "query": "how does the API server work?",
  "total_searched": 2,
  "cost_usd": 0.0001,
  "input_tokens": 8,
  "latency_ms": 85.2
}
```

**Errors:** `422` (validation), `429` (rate limit), `503` (provider error)

---

### `POST /generate`

Generate content using a named template or freeform prompt. Supports built-in guardrails (PII leak, hallucination, competitor mention, off-topic) and custom guardrails.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template` | `string \| null` | `null` | Template name (mutually exclusive with `prompt` in practice) |
| `prompt` | `string \| null` | `null` | Freeform prompt |
| `variables` | `object` | `{}` | Template variables |
| `guardrails` | `string[]` | `[]` | Guardrail names to apply (e.g. `["pii_leak", "competitor_mention"]`) |
| `max_tokens` | `integer` (≥ 1) | `1024` | Maximum output tokens |
| `temperature` | `float` (0.0–2.0) | `0.7` | Sampling temperature |
| `competitors` | `string[]` | `[]` | Competitor names (for `competitor_mention` guardrail) |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Generated content |
| `guardrail_flags` | `string[]` | Guardrails that were triggered (empty if clean) |
| `retry_count` | `integer` | Number of retries due to guardrail violations |
| `cost_usd` | `float` | Total cost (including retries) |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `model` | `string` | Model used |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a product description for a wireless noise-cancelling headphone",
    "guardrails": ["competitor_mention"],
    "competitors": ["Sony", "Bose"],
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

```json
{
  "text": "Experience premium sound quality with our wireless noise-cancelling headphones...",
  "guardrail_flags": [],
  "retry_count": 0,
  "cost_usd": 0.0025,
  "input_tokens": 42,
  "output_tokens": 180,
  "model": "gpt-4o",
  "latency_ms": 2100.4
}
```

**Errors:** `422` (validation), `429` (rate limit), `503` (provider error)

---

### Chat

#### `POST /chat`

Send a message in a multi-turn conversation. Creates a new conversation if `conversation_id` is not provided. Supports three memory modes and optional streaming.

The server maintains up to **200 concurrent conversations** in memory with LRU eviction.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `string` | *(required)* | User message |
| `conversation_id` | `string \| null` | `null` | Existing conversation ID or `null` to create new |
| `system_prompt` | `string \| null` | `null` | Custom system prompt |
| `memory` | `"buffer" \| "summary" \| "window"` | `"buffer"` | Memory mode |
| `memory_k` | `integer` | `20` | Message window size (for `window` mode) |
| `stream` | `boolean` | `false` | Stream the response as `text/plain` chunks |

**Response `200` (non-streaming):**

| Field | Type | Description |
|-------|------|-------------|
| `message` | `string` | Assistant response |
| `conversation_id` | `string` | Conversation ID (newly created or existing) |
| `tool_calls` | `object[]` | Tool calls made during this turn |
| `iterations` | `integer` | Agent loop iterations |
| `memory_mode` | `string` | Memory mode used |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `cost` | `float` | Total cost |
| `model` | `string` | Model used |
| `trace_id` | `string` | Distributed trace ID |
| `latency_ms` | `float` | End-to-end latency |

**Response `200` (streaming) — `Content-Type: text/plain`:**

Returns a streaming response. Text chunks are yielded as they become available from the LLM.

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "system_prompt": "You are a helpful geography tutor.",
    "memory": "buffer"
  }'
```

```json
{
  "message": "The capital of France is Paris.",
  "conversation_id": "conv-a1b2c3d4",
  "tool_calls": [],
  "iterations": 1,
  "memory_mode": "buffer",
  "input_tokens": 35,
  "output_tokens": 12,
  "cost": 0.0003,
  "model": "gpt-4o",
  "trace_id": "trace-xyz",
  "latency_ms": 650.1
}
```

---

#### `POST /chat/reset`

Reset (clear history of) an existing conversation.

**Request:**

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | `string` | Conversation ID to reset |

**Response `200`:**

```json
{
  "status": "reset",
  "conversation_id": "conv-a1b2c3d4"
}
```

**Errors:** `404` (conversation not found)

---

#### `GET /chat/conversations`

List all active conversation IDs.

**Response `200`:**

```json
{
  "conversations": ["conv-a1b2c3d4", "conv-e5f6g7h8"],
  "count": 2
}
```

---

### `POST /generate/rewrite`

Rewrite existing text according to instructions, tone, and/or reading level. Uses the same guardrail loop as `/generate`.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Original text to rewrite |
| `instructions` | `string \| null` | `null` | Free-form rewriting directives |
| `tone` | `string \| null` | `null` | Target tone (e.g. `"formal"`, `"casual"`) |
| `reading_level` | `string \| null` | `null` | Target reading level (e.g. `"5th grade"`, `"college"`) |
| `guardrails` | `string[]` | `[]` | Guardrails to enforce on rewritten output |

If none of `instructions`, `tone`, or `reading_level` are provided, the rewriter defaults to "Improve clarity and readability".

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Rewritten text |
| `guardrail_flags` | `string[]` | Guardrails still violated after retries |
| `retry_count` | `integer` | Number of retry attempts |
| `cost_usd` | `float` | Total cost |
| `input_tokens` | `integer` | Input tokens consumed |
| `output_tokens` | `integer` | Output tokens generated |
| `model` | `string` | Model used |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/generate/rewrite \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The product is good. You should buy it.",
    "instructions": "Make it more compelling",
    "tone": "professional"
  }'
```

**Errors:** `422` (empty text)

---

### `POST /translate`

Translate text between languages with optional formality and domain controls. Auto-detects source language by default.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to translate |
| `target_language` | `string` | `"English"` | Language to translate into |
| `source_language` | `string` | `"auto"` | Source language (`"auto"` = detect) |
| `formality` | `"formal" \| "informal" \| "neutral"` | `"neutral"` | Formality level |
| `domain` | `string \| null` | `null` | Domain hint (e.g. `"legal"`, `"medical"`) |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Translated text |
| `source_language` | `string` | Detected or specified source language |
| `target_language` | `string` | Target language |
| `formality` | `string` | Formality level used |
| `model` | `string` | Model used |
| `cost_usd` | `float` | Cost of the operation |
| `input_tokens` | `integer` | Input tokens consumed |
| `output_tokens` | `integer` | Output tokens generated |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "target_language": "Spanish",
    "formality": "formal"
  }'
```

```json
{
  "text": "Hola, \u00bfc\u00f3mo est\u00e1 usted?",
  "source_language": "English",
  "target_language": "Spanish",
  "formality": "formal",
  "model": "gpt-4o-mini",
  "cost_usd": 0.0002,
  "input_tokens": 85,
  "output_tokens": 12,
  "latency_ms": 420.3
}
```

**Errors:** `422` (empty text)

---

### `POST /vision`

Analyze an image using a multimodal LLM. Requires a vision-capable model (e.g. GPT-4o, Claude 3).

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `string` | *(required)* | Image URL (HTTPS) |
| `task` | `"describe" \| "qa" \| "extract" \| "classify"` | `"describe"` | Analysis task |
| `question` | `string \| null` | `null` | Question about the image (required for `qa` task) |
| `focus` | `string \| null` | `null` | Area to focus on (for `describe` task) |
| `labels` | `string[]` | `[]` | Classification labels (for `classify` task) |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Analysis result |
| `task` | `string` | Task that was performed |
| `image_source` | `string` | Image source analyzed |
| `model` | `string` | Model used |
| `cost_usd` | `float` | Cost of the operation |
| `input_tokens` | `integer` | Input tokens consumed |
| `output_tokens` | `integer` | Output tokens generated |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/vision \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://example.com/photo.jpg",
    "task": "describe"
  }'
```

**Errors:** `422` (empty source, missing question for `qa` task)

---

### `POST /classify/sentiment`

Analyze text sentiment with optional aspect-level breakdown. Returns overall polarity, emotion types, and per-aspect scores.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to analyze |
| `aspects` | `string[]` | `[]` | Aspects to evaluate individually |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `overall` | `string` | Overall sentiment (`"positive"`, `"negative"`, `"neutral"`, `"mixed"`) |
| `score` | `float` | Score from -1.0 to 1.0 |
| `aspects` | `AspectOut[]` | Per-aspect results |
| `emotions` | `EmotionOut[]` | Detected emotions with intensity |
| `rationale` | `string` | Brief explanation |
| `model` | `string` | Model used |
| `cost_usd` | `float` | Cost of the operation |
| `input_tokens` | `integer` | Input tokens consumed |
| `output_tokens` | `integer` | Output tokens generated |
| `latency_ms` | `float` | End-to-end latency |

**Example:**

```bash
curl -X POST http://localhost:8000/classify/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The food was great but service was slow.",
    "aspects": ["food", "service"]
  }'
```

```json
{
  "overall": "mixed",
  "score": 0.15,
  "aspects": [
    {"aspect": "food", "sentiment": "positive", "score": 0.85},
    {"aspect": "service", "sentiment": "negative", "score": -0.6}
  ],
  "emotions": [
    {"emotion": "satisfaction", "intensity": 0.7},
    {"emotion": "frustration", "intensity": 0.5}
  ],
  "rationale": "Positive about food quality but negative about service speed.",
  "model": "gpt-4o-mini",
  "cost_usd": 0.0003,
  "input_tokens": 120,
  "output_tokens": 85,
  "latency_ms": 580.2
}
```

**Errors:** `422` (empty text)

---

### `POST /moderate`

Run content moderation checks: PII detection (regex-based with Luhn validation for credit cards), toxicity scoring, and prompt injection detection. Optionally redact PII in-place.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Text to moderate |
| `checks` | `string[]` | `["pii", "toxicity", "prompt_injection"]` | Check types to run |
| `pii_types` | `string[]` | `["email", "phone", "ssn", "credit_card"]` | PII categories to detect |
| `action` | `"check" \| "redact"` | `"check"` | `"check"` = flag only; `"redact"` = replace PII with `[REDACTED]` |

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `flagged` | `boolean` | `true` if any issues were found |
| `flags` | `Flag[]` | Detailed flag objects |
| `redacted_text` | `string \| null` | Redacted version (only if `action="redact"`) |
| `cost_usd` | `float` | LLM cost (for toxicity / prompt injection checks) |
| `input_tokens` | `integer` | Tokens consumed |
| `output_tokens` | `integer` | Tokens generated |
| `latency_ms` | `float` | End-to-end latency |

Each **Flag** object:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Flag category (`"pii"`, `"toxicity"`, `"prompt_injection"`) |
| `subtype` | `string \| null` | Specific type (`"email"`, `"phone"`, `"ssn"`, `"credit_card"`) |
| `span` | `string \| null` | Matched text fragment |
| `confidence` | `float` | Confidence score (0.0–1.0) |

**Example:**

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My email is john@example.com and my SSN is 123-45-6789",
    "checks": ["pii"],
    "pii_types": ["email", "ssn"],
    "action": "redact"
  }'
```

```json
{
  "flagged": true,
  "flags": [
    {"type": "pii", "subtype": "email", "span": "john@example.com", "confidence": 1.0},
    {"type": "pii", "subtype": "ssn", "span": "123-45-6789", "confidence": 1.0}
  ],
  "redacted_text": "My email is [REDACTED] and my SSN is [REDACTED]",
  "cost_usd": 0.0,
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_ms": 2.1
}
```

**Errors:** `422` (invalid `action` or unknown check/pii types)

---

### `POST /pipeline`

Execute a custom multi-step pipeline inline. Each step references an AIForge module by class name and passes configuration.

**Request:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | `string` | *(required)* | Input text for the pipeline |
| `steps` | `Step[]` | *(required, non-empty)* | Pipeline step definitions |
| `context` | `object \| null` | `null` | Shared context passed between steps |

Each **Step** object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `string` | *(required)* | Step name (used as key in results) |
| `module` | `string` | *(required)* | Module class name — `"Summarizer"`, `"Classifier"`, `"Extractor"`, `"Generator"`, `"Moderator"`, `"ChatAgent"`, `"DocQA"`, `"SemanticSearch"` |
| `config` | `object` | `{}` | Constructor kwargs for the module |

**Response `200`:**

```json
{
  "step_results": {
    "moderate": { "flagged": false, "flags": [] },
    "summarize": { "summary": "...", "cost_usd": 0.001 }
  },
  "context": {}
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Customer feedback text here...",
    "steps": [
      {"name": "moderate", "module": "Moderator", "config": {"checks": ["pii", "toxicity"]}},
      {"name": "classify", "module": "Classifier", "config": {"labels": ["positive", "negative", "neutral"]}},
      {"name": "summarize", "module": "Summarizer", "config": {"style": "concise"}}
    ]
  }'
```

**Errors:** `422` (empty steps, invalid module name, ValueError), `500` (execution error)

---

### `GET /jobs/{job_id}`

Poll the status of an async job created by `/qa/ingest` or `/search/index` with a `webhook_url`.

**Path Parameters:** `job_id` — the UUID returned in the `202` response

**Response `200`:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | Job UUID |
| `status` | `"accepted" \| "running" \| "completed" \| "failed"` | Current status |
| `result` | `object \| null` | Result payload (when `completed`) |
| `error` | `string \| null` | Error message (when `failed`) |
| `created_at` | `string` | ISO 8601 timestamp |
| `completed_at` | `string \| null` | ISO 8601 timestamp (when done) |

**Example:**

```bash
curl http://localhost:8000/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "result": {
    "indexed_count": 2,
    "collection": "knowledge-base",
    "total_documents": 2
  },
  "error": null,
  "created_at": "2026-03-25T10:30:00Z",
  "completed_at": "2026-03-25T10:30:05Z"
}
```

**Errors:** `404` (job not found)

---

## Async Jobs & Webhooks

The `/qa/ingest` and `/search/index` endpoints support **async execution** with webhook callbacks. Include a `webhook_url` field in the request body to opt in.

### Flow

1. Client sends request with `webhook_url` field
2. Server validates the URL (HTTPS required in production, no private IPs)
3. Server returns `202 Accepted` with a `job_id`
4. Server processes the work in the background
5. On completion (or failure), server POSTs the result to the `webhook_url`
6. Client can also poll `GET /jobs/{job_id}` at any time

### Webhook Payload

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "completed",
  "result": { "indexed_count": 42, "collection": "my-docs", "..." : "..." },
  "error": null,
  "completed_at": "2026-03-25T10:30:05Z"
}
```

On failure:

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "failed",
  "result": null,
  "error": "Source directory not found: ./nonexistent/",
  "completed_at": "2026-03-25T10:30:02Z"
}
```

### Webhook Delivery

- **Retries:** 3 attempts with exponential backoff (1s → 2s → 4s)
- **Timeout:** Configurable (default 10 seconds)
- **Method:** `POST` with `Content-Type: application/json`

### SSRF Protection

Webhook URLs are validated before acceptance:

- **Blocked:** private/internal IPs (`10.x`, `172.16-31.x`, `192.168.x`, `127.x`, `::1`), resolved DNS pointing to private ranges
- **Blocked:** `http://` URLs (unless `webhook_allow_insecure: true` in `aiforge.yaml`)
- **Blocked:** `localhost` and loopback addresses

### Webhook Configuration

```yaml
# aiforge.yaml
webhook_enabled: true
webhook_timeout: 10.0          # seconds per delivery attempt
webhook_allow_insecure: false   # allow http:// URLs (dev only)
```

---

## Integration Examples

All examples below assume the server is running at `http://localhost:8000` with `AIFORGE_API_KEY` set. Replace the base URL and token for your environment.

### Node.js / TypeScript

```typescript
// aiforge-client.ts — minimal wrapper around the AIForge API

const BASE_URL = process.env.AIFORGE_URL ?? "http://localhost:8000";
const API_KEY  = process.env.AIFORGE_API_KEY ?? "";

interface RequestOptions {
  method?: string;
  body?: unknown;
}

async function aiforge<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method: opts.method ?? (opts.body ? "POST" : "GET"),
    headers,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`AIForge ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Summarise ────────────────────────────────────────
interface SummarizeResponse {
  summary: string;
  strategy_used: "stuff" | "map_reduce";
  chunks_processed: number;
  cost_usd: number;
  model: string;
  latency_ms: number;
}

const summary = await aiforge<SummarizeResponse>("/summarize", {
  body: {
    text: "Long article content here...",
    style: "bullet_points",
    max_length: 300,
  },
});
console.log(summary.summary);
console.log(`Cost: $${summary.cost_usd.toFixed(4)}`);

// ── Classify ─────────────────────────────────────────
interface ClassifyResponse {
  label: string;
  scores: { label: string; confidence: number }[];
  rationale: string;
  cost_usd: number;
}

const classification = await aiforge<ClassifyResponse>("/classify", {
  body: {
    text: "The app crashes when I click the save button",
    labels: ["bug", "feature_request", "question"],
  },
});
console.log(`Label: ${classification.label} (${classification.scores[0].confidence})`);

// ── Extract ──────────────────────────────────────────
interface ExtractResponse {
  data: Record<string, unknown>;
  validation_passed: boolean;
  cost_usd: number;
}

const extraction = await aiforge<ExtractResponse>("/extract", {
  body: {
    text: "Jane Doe, VP of Engineering at TechCorp, jane@techcorp.io",
    output_schema: {
      type: "object",
      properties: {
        name: { type: "string" },
        title: { type: "string" },
        company: { type: "string" },
        email: { type: "string" },
      },
      required: ["name"],
    },
  },
});
console.log(extraction.data);

// ── Chat (multi-turn) ────────────────────────────────
interface ChatResponse {
  message: string;
  conversation_id: string;
  cost: number;
}

let convId: string | null = null;

// Turn 1
const turn1 = await aiforge<ChatResponse>("/chat", {
  body: { message: "What is Kubernetes?", conversation_id: convId },
});
convId = turn1.conversation_id;
console.log(turn1.message);

// Turn 2 (same conversation)
const turn2 = await aiforge<ChatResponse>("/chat", {
  body: { message: "How does it handle scaling?", conversation_id: convId },
});
console.log(turn2.message);

// ── Moderate ─────────────────────────────────────────
interface ModerateResponse {
  flagged: boolean;
  flags: { type: string; subtype: string | null; span: string | null }[];
  redacted_text: string | null;
}

const modResult = await aiforge<ModerateResponse>("/moderate", {
  body: {
    text: "Contact me at user@example.com",
    checks: ["pii"],
    action: "redact",
  },
});
console.log(modResult.redacted_text);

// ── Semantic Search ──────────────────────────────────
// Index
await aiforge("/search/index", {
  body: {
    documents: [
      { id: "1", text: "AIForge simplifies AI integration.", metadata: { source: "docs" } },
      { id: "2", text: "LangChain is a framework for LLM apps.", metadata: { source: "notes" } },
    ],
    collection: "kb",
  },
});

// Query
interface SearchResponse {
  hits: { id: string; text: string; score: number; metadata: Record<string, unknown> }[];
}

const results = await aiforge<SearchResponse>("/search/query", {
  body: { query: "AI integration tool", collection: "kb", top_k: 3 },
});
results.hits.forEach((h) => console.log(`${h.score.toFixed(2)} — ${h.text}`));

// ── Document Q&A ─────────────────────────────────────
// Ingest
await aiforge("/qa/ingest", {
  body: { source: "./docs/", collection: "project-docs" },
});

// Ask
interface QAResponse {
  text: string;
  sources: string[];
  confidence: number;
  cost_usd: number;
}

const answer = await aiforge<QAResponse>("/qa/ask", {
  body: { query: "How does authentication work?", collection: "project-docs" },
});
console.log(answer.text);
console.log("Sources:", answer.sources);

// ── Generate ─────────────────────────────────────────
interface GenerateResponse {
  text: string;
  guardrail_flags: string[];
  cost_usd: number;
}

const generated = await aiforge<GenerateResponse>("/generate", {
  body: {
    prompt: "Write a professional email declining a meeting invitation",
    guardrails: ["pii_leak"],
    temperature: 0.7,
  },
});
console.log(generated.text);

// ── Pipeline ─────────────────────────────────────────
const pipelineResult = await aiforge<{ step_results: Record<string, unknown> }>("/pipeline", {
  body: {
    text: "Customer complaint text here...",
    steps: [
      { name: "moderate", module: "Moderator", config: { checks: ["pii"] } },
      { name: "classify", module: "Classifier", config: { labels: ["positive", "negative", "neutral"] } },
      { name: "summarize", module: "Summarizer", config: { style: "concise" } },
    ],
  },
});
console.log(pipelineResult.step_results);
```

---

### Go

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// ── Client ───────────────────────────────────────────

var (
	baseURL = envOr("AIFORGE_URL", "http://localhost:8000")
	apiKey  = os.Getenv("AIFORGE_API_KEY")
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func aiforgeRequest(method, path string, body any, dest any) error {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal: %w", err)
		}
		reqBody = bytes.NewReader(b)
	}

	req, err := http.NewRequest(method, baseURL+path, reqBody)
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		detail, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("aiforge %d: %s", resp.StatusCode, string(detail))
	}

	if dest != nil {
		return json.NewDecoder(resp.Body).Decode(dest)
	}
	return nil
}

// ── Response types ───────────────────────────────────

type SummarizeResponse struct {
	Summary         string  `json:"summary"`
	StrategyUsed    string  `json:"strategy_used"`
	ChunksProcessed int     `json:"chunks_processed"`
	CostUSD         float64 `json:"cost_usd"`
	Model           string  `json:"model"`
	LatencyMs       float64 `json:"latency_ms"`
}

type ClassifyResponse struct {
	Label     string       `json:"label"`
	Scores    []LabelScore `json:"scores"`
	Rationale string       `json:"rationale"`
	CostUSD   float64      `json:"cost_usd"`
}

type LabelScore struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

type ExtractResponse struct {
	Data             map[string]any `json:"data"`
	ValidationPassed bool           `json:"validation_passed"`
	CostUSD          float64        `json:"cost_usd"`
}

type ChatResponse struct {
	Message        string  `json:"message"`
	ConversationID string  `json:"conversation_id"`
	Cost           float64 `json:"cost"`
	Model          string  `json:"model"`
}

type ModerateResponse struct {
	Flagged      bool   `json:"flagged"`
	RedactedText string `json:"redacted_text"`
}

type SearchHit struct {
	ID       string         `json:"id"`
	Text     string         `json:"text"`
	Score    float64        `json:"score"`
	Metadata map[string]any `json:"metadata"`
}

type SearchResponse struct {
	Hits          []SearchHit `json:"hits"`
	TotalSearched int         `json:"total_searched"`
}

type QAResponse struct {
	Text       string   `json:"text"`
	Sources    []string `json:"sources"`
	Confidence float64  `json:"confidence"`
	CostUSD    float64  `json:"cost_usd"`
}

type GenerateResponse struct {
	Text           string   `json:"text"`
	GuardrailFlags []string `json:"guardrail_flags"`
	CostUSD        float64  `json:"cost_usd"`
}

// ── Usage ────────────────────────────────────────────

func main() {
	// Summarise
	var sumResp SummarizeResponse
	err := aiforgeRequest("POST", "/summarize", map[string]any{
		"text":       "Long article content...",
		"style":      "bullet_points",
		"max_length": 300,
	}, &sumResp)
	if err != nil {
		panic(err)
	}
	fmt.Println("Summary:", sumResp.Summary)
	fmt.Printf("Cost: $%.4f\n", sumResp.CostUSD)

	// Classify
	var clsResp ClassifyResponse
	_ = aiforgeRequest("POST", "/classify", map[string]any{
		"text":   "The save button does not work on mobile",
		"labels": []string{"bug", "feature_request", "question"},
	}, &clsResp)
	fmt.Printf("Label: %s (%.2f)\n", clsResp.Label, clsResp.Scores[0].Confidence)

	// Extract
	var extResp ExtractResponse
	_ = aiforgeRequest("POST", "/extract", map[string]any{
		"text": "Jane Doe, CTO at TechCorp — jane@techcorp.io",
		"output_schema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":    map[string]any{"type": "string"},
				"title":   map[string]any{"type": "string"},
				"company": map[string]any{"type": "string"},
				"email":   map[string]any{"type": "string"},
			},
			"required": []string{"name"},
		},
	}, &extResp)
	fmt.Println("Extracted:", extResp.Data)

	// Chat (multi-turn)
	var turn1 ChatResponse
	_ = aiforgeRequest("POST", "/chat", map[string]any{
		"message": "What is Docker?",
	}, &turn1)
	fmt.Println("Turn 1:", turn1.Message)

	var turn2 ChatResponse
	_ = aiforgeRequest("POST", "/chat", map[string]any{
		"message":         "How does it differ from VMs?",
		"conversation_id": turn1.ConversationID,
	}, &turn2)
	fmt.Println("Turn 2:", turn2.Message)

	// Moderate
	var modResp ModerateResponse
	_ = aiforgeRequest("POST", "/moderate", map[string]any{
		"text":   "Email me at user@example.com",
		"checks": []string{"pii"},
		"action": "redact",
	}, &modResp)
	fmt.Println("Redacted:", modResp.RedactedText)

	// Search — index
	_ = aiforgeRequest("POST", "/search/index", map[string]any{
		"documents": []map[string]any{
			{"id": "1", "text": "AIForge simplifies AI.", "metadata": map[string]any{"src": "readme"}},
			{"id": "2", "text": "Go is a compiled language.", "metadata": map[string]any{"src": "notes"}},
		},
		"collection": "kb",
	}, nil)

	// Search — query
	var searchResp SearchResponse
	_ = aiforgeRequest("POST", "/search/query", map[string]any{
		"query":      "AI tool",
		"collection": "kb",
		"top_k":      3,
	}, &searchResp)
	for _, h := range searchResp.Hits {
		fmt.Printf("  %.2f — %s\n", h.Score, h.Text)
	}

	// Q&A — ingest + ask
	_ = aiforgeRequest("POST", "/qa/ingest", map[string]any{
		"source":     "./docs/",
		"collection": "project",
	}, nil)

	var qaResp QAResponse
	_ = aiforgeRequest("POST", "/qa/ask", map[string]any{
		"query":      "How does auth work?",
		"collection": "project",
	}, &qaResp)
	fmt.Println("Answer:", qaResp.Text)

	// Generate
	var genResp GenerateResponse
	_ = aiforgeRequest("POST", "/generate", map[string]any{
		"prompt":      "Write a haiku about Go programming",
		"temperature": 0.9,
	}, &genResp)
	fmt.Println("Generated:", genResp.Text)
}
```

---

### Java

```java
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

/**
 * Minimal AIForge HTTP client for Java 11+.
 * Uses java.net.http (no external dependencies beyond Jackson for JSON).
 *
 * Requires:
 *   - Java 11+
 *   - com.fasterxml.jackson.core:jackson-databind:2.17+
 */
public class AIForgeClient {

    private static final String BASE_URL =
            System.getenv().getOrDefault("AIFORGE_URL", "http://localhost:8000");
    private static final String API_KEY =
            System.getenv().getOrDefault("AIFORGE_API_KEY", "");

    private static final HttpClient HTTP = HttpClient.newHttpClient();
    private static final ObjectMapper JSON = new ObjectMapper();

    // ── Generic request helper ───────────────────────

    static JsonNode request(String method, String path, Object body) throws Exception {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + path))
                .header("Content-Type", "application/json");

        if (!API_KEY.isEmpty()) {
            builder.header("Authorization", "Bearer " + API_KEY);
        }

        if (body != null) {
            String json = JSON.writeValueAsString(body);
            builder.method(method, HttpRequest.BodyPublishers.ofString(json));
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        }

        HttpResponse<String> resp = HTTP.send(builder.build(),
                HttpResponse.BodyHandlers.ofString());

        if (resp.statusCode() >= 400) {
            throw new RuntimeException(
                    "AIForge " + resp.statusCode() + ": " + resp.body());
        }

        return JSON.readTree(resp.body());
    }

    static JsonNode post(String path, Object body) throws Exception {
        return request("POST", path, body);
    }

    static JsonNode get(String path) throws Exception {
        return request("GET", path, null);
    }

    // ── Examples ─────────────────────────────────────

    public static void main(String[] args) throws Exception {
        // ── Health check ──
        JsonNode health = get("/health");
        System.out.println("Status: " + health.get("status").asText());

        // ── Summarise ──
        ObjectNode sumBody = JSON.createObjectNode()
                .put("text", "Long article about machine learning and its applications...")
                .put("style", "bullet_points")
                .put("max_length", 300);

        JsonNode sumResp = post("/summarize", sumBody);
        System.out.println("Summary: " + sumResp.get("summary").asText());
        System.out.printf("Cost: $%.4f%n", sumResp.get("cost_usd").asDouble());

        // ── Classify ──
        ObjectNode clsBody = JSON.createObjectNode()
                .put("text", "The login page throws a 500 error");

        ArrayNode labels = clsBody.putArray("labels");
        labels.add("bug").add("feature_request").add("question");

        JsonNode clsResp = post("/classify", clsBody);
        System.out.printf("Label: %s (%.2f)%n",
                clsResp.get("label").asText(),
                clsResp.get("scores").get(0).get("confidence").asDouble());

        // ── Extract ──
        ObjectNode schema = JSON.createObjectNode()
                .put("type", "object");
        ObjectNode props = schema.putObject("properties");
        props.putObject("name").put("type", "string");
        props.putObject("title").put("type", "string");
        props.putObject("company").put("type", "string");
        props.putObject("email").put("type", "string");
        schema.putArray("required").add("name");

        ObjectNode extBody = JSON.createObjectNode()
                .put("text", "Jane Doe, VP Engineering at TechCorp, jane@techcorp.io");
        extBody.set("output_schema", schema);

        JsonNode extResp = post("/extract", extBody);
        System.out.println("Extracted: " + extResp.get("data"));

        // ── Chat (multi-turn) ──
        ObjectNode chat1 = JSON.createObjectNode()
                .put("message", "What is Kubernetes?");
        JsonNode chatResp1 = post("/chat", chat1);
        String convId = chatResp1.get("conversation_id").asText();
        System.out.println("Turn 1: " + chatResp1.get("message").asText());

        ObjectNode chat2 = JSON.createObjectNode()
                .put("message", "How does it handle auto-scaling?")
                .put("conversation_id", convId);
        JsonNode chatResp2 = post("/chat", chat2);
        System.out.println("Turn 2: " + chatResp2.get("message").asText());

        // ── Moderate ──
        ObjectNode modBody = JSON.createObjectNode()
                .put("text", "Call me at 555-0123 or user@example.com")
                .put("action", "redact");
        ArrayNode checks = modBody.putArray("checks");
        checks.add("pii");

        JsonNode modResp = post("/moderate", modBody);
        System.out.println("Redacted: " + modResp.get("redacted_text").asText());

        // ── Semantic Search — index ──
        ObjectNode idxBody = JSON.createObjectNode()
                .put("collection", "kb");
        ArrayNode docs = idxBody.putArray("documents");
        docs.addObject().put("id", "1").put("text", "AIForge simplifies AI.");
        docs.addObject().put("id", "2").put("text", "Java is a compiled language.");

        post("/search/index", idxBody);

        // ── Semantic Search — query ──
        ObjectNode qBody = JSON.createObjectNode()
                .put("query", "AI tool")
                .put("collection", "kb")
                .put("top_k", 3);

        JsonNode searchResp = post("/search/query", qBody);
        for (JsonNode hit : searchResp.get("hits")) {
            System.out.printf("  %.2f — %s%n",
                    hit.get("score").asDouble(),
                    hit.get("text").asText());
        }

        // ── Document Q&A — ingest + ask ──
        ObjectNode ingestBody = JSON.createObjectNode()
                .put("source", "./docs/")
                .put("collection", "project");
        post("/qa/ingest", ingestBody);

        ObjectNode askBody = JSON.createObjectNode()
                .put("query", "How does authentication work?")
                .put("collection", "project");
        JsonNode qaResp = post("/qa/ask", askBody);
        System.out.println("Answer: " + qaResp.get("text").asText());
        System.out.println("Sources: " + qaResp.get("sources"));

        // ── Generate ──
        ObjectNode genBody = JSON.createObjectNode()
                .put("prompt", "Write a haiku about Java programming")
                .put("temperature", 0.9);
        JsonNode genResp = post("/generate", genBody);
        System.out.println("Generated: " + genResp.get("text").asText());

        // ── Pipeline ──
        ObjectNode pipeBody = JSON.createObjectNode()
                .put("text", "Customer feedback text...");
        ArrayNode steps = pipeBody.putArray("steps");
        steps.addObject()
                .put("name", "moderate").put("module", "Moderator")
                .set("config", JSON.createObjectNode().putArray("checks").add("pii"));
        steps.addObject()
                .put("name", "classify").put("module", "Classifier")
                .set("config", JSON.createObjectNode()
                        .set("labels", JSON.createArrayNode()
                                .add("positive").add("negative").add("neutral")));
        steps.addObject()
                .put("name", "summarize").put("module", "Summarizer")
                .set("config", JSON.createObjectNode().put("style", "concise"));

        JsonNode pipeResp = post("/pipeline", pipeBody);
        System.out.println("Pipeline: " + pipeResp.get("step_results"));
    }
}
```

---

## CORS

CORS middleware is enabled by default. The allowed origin defaults to `http://localhost:8501` (the Playground port).

```bash
# Allow a specific production origin
export AIFORGE_CORS_ORIGINS=https://your-frontend.example.com

# Allow multiple origins (comma-separated)
export AIFORGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Allow all origins (development only — credentials are disabled when using *)
export AIFORGE_CORS_ORIGINS=*
```

**Behaviour:**

| Setting | `allow_credentials` | Notes |
|---------|-------------------|-------|
| Specific origins | `true` | Cookies / auth headers are forwarded |
| `*` (wildcard) | `false` | Browsers block credentials with wildcard CORS |

---

## Docker Deployment

### Dockerfile

The project includes a multi-stage `Dockerfile`:

1. **Builder stage** — installs AIForge with `server`, `agentic`, and `docs`
   extras into a prefix directory.
2. **Runtime stage** — copies the installed packages into a clean
   `python:3.12-slim` image.

```dockerfile
# ── Build stage ────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY aiforge/ aiforge/

RUN pip install --no-cache-dir --prefix=/install -e ".[server,agentic,docs]"

# ── Runtime stage ─────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["aiforge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
```

The image exposes port **8000** and includes a `HEALTHCHECK` that probes
`/health` every 30 seconds.

### Docker Compose

The `docker-compose.yml` defines two services:

| Service | Port | Description |
|---------|------|-------------|
| `api` | `8000` (override with `AIFORGE_PORT`) | FastAPI server with health check and persistent ChromaDB volume |
| `playground` | `8501` (override with `AIFORGE_PLAYGROUND_PORT`) | Streamlit UI; waits for API to be healthy before starting |

Both services read secrets from `.env` via `env_file`. ChromaDB data is
stored in a named volume (`chroma-data`) that survives container restarts.

### Run with Docker

```bash
# Build and start both services
docker compose up --build -d

# Or use the Makefile shortcut
make docker-up

# Custom ports
AIFORGE_PORT=9000 AIFORGE_PLAYGROUND_PORT=9501 docker compose up -d

# Run just the API server standalone
docker run -p 8000:8000 --env-file .env \
  -v chroma-data:/data/chroma \
  -e AIFORGE_CHROMA_PERSIST_DIR=/data/chroma \
  aiforge

# View logs
docker compose logs -f        # or: make docker-logs

# Tear down (volume persists)
docker compose down            # or: make docker-down

# Tear down AND delete data
docker compose down -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIFORGE_API_KEY` | *(empty — no auth)* | Bearer token for API authentication |
| `AIFORGE_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed CORS origins |
| `AIFORGE_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `AIFORGE_LOG_FORMAT` | `text` | Log format (`text` or `json` for structured logging) |
| `AIFORGE_PROVIDER` | `openai` | LLM provider (`openai`, `anthropic`, `ollama`, `azure`) |
| `AIFORGE_OPENAI_API_KEY` | — | OpenAI API key |
| `AIFORGE_ANTHROPIC_API_KEY` | — | Anthropic API key |
| `AIFORGE_CHROMA_PERSIST_DIR` | `./aiforge_data/chroma` | ChromaDB persistence directory |
| `AIFORGE_WEBHOOK_ENABLED` | `true` | Enable webhook support |
| `AIFORGE_WEBHOOK_TIMEOUT` | `10.0` | Webhook delivery timeout (seconds) |
| `AIFORGE_WEBHOOK_ALLOW_INSECURE` | `false` | Allow `http://` webhook URLs (dev only) |

---

## Graceful Shutdown

The API server handles `SIGTERM` and `SIGINT` signals gracefully:

1. Stop accepting new requests
2. Wait for in-flight requests to complete (up to 30-second timeout)
3. Flush pending cost tracking and tracing data
4. Close vector store connections (ChromaDB, pgvector, etc.)
5. Exit with code 0

This is handled by uvicorn's built-in signal handling plus a custom `lifespan` context manager for AIForge-specific cleanup.

---

## See Also

- [API Authentication](api-auth.md) — Bearer token setup, error responses, security properties
- [Configuration](configuration.md) — Auth, CORS, and server settings
- [Playground](playground.md) — Interactive UI that connects to the API server
- [Quick Start](quickstart.md) — Getting started
- [Modules](modules/) — Detailed SDK documentation for each module
