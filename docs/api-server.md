# API Server

AIForge can run as a **standalone REST API** exposing all modules as HTTP endpoints.

---

## Launch

```bash
# Start API server
aiforge serve --host 0.0.0.0 --port 8000

# With API key authentication (recommended for non-local deployments)
AIFORGE_API_KEY=your-secret-key aiforge serve --host 0.0.0.0 --port 8000

# Or with Docker
docker run -p 8000:8000 \
  -e AIFORGE_OPENAI_API_KEY=sk-... \
  -e AIFORGE_API_KEY=your-secret-key \
  aiforge/server
```

---

## Endpoints

| Method | Endpoint | Module | Description |
|--------|----------|--------|-------------|
| POST | `/summarize` | Summarizer | Summarize text |
| POST | `/qa/ingest` | DocQA | Ingest documents |
| POST | `/qa/ask` | DocQA | Ask a question |
| POST | `/extract` | Extractor | Extract structured data |
| POST | `/classify` | Classifier | Classify text |
| POST | `/search/index` | SemanticSearch | Index documents |
| POST | `/search/query` | SemanticSearch | Search |
| POST | `/generate` | Generator | Generate content |
| POST | `/chat` | ChatAgent | Chat message |
| POST | `/moderate` | Moderator | Content check |
| POST | `/moderate/redact` | Moderator | PII redaction |
| POST | `/pipeline` | Pipeline | Run a named pipeline |
| GET | `/jobs/{job_id}` | Webhooks | Poll async job status |
| GET | `/health` | — | Health + status |
| GET | `/docs` | — | OpenAPI/Swagger UI (auto) |

---

## Authentication

When `AIFORGE_API_KEY` is set (non-empty), all endpoints except `/health` and `/docs` require a Bearer token:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "labels": ["a", "b"]}'
```

If the key is missing or wrong, the API returns `401 Unauthorized`. When `AIFORGE_API_KEY` is **not set** (default for local dev), auth is disabled and all endpoints are open.

---

## Async Jobs & Webhooks

The `/qa/ingest` and `/search/index` endpoints support **async execution** with webhook notifications. Pass a `webhook_url` field in the request body to opt in:

```bash
curl -X POST http://localhost:8000/search/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"id": "1", "text": "..."}],
    "collection": "products",
    "webhook_url": "https://example.com/hooks/aiforge"
  }'

# Returns immediately:
# HTTP 202 Accepted
# {"job_id": "abc-123-def", "status": "accepted"}
```

### Polling Job Status

```bash
curl http://localhost:8000/jobs/abc-123-def

# {"job_id": "abc-123-def", "status": "completed", "result": {...}}
# Status values: accepted | running | completed | failed
```

### Webhook Delivery

On completion (or failure), AIForge sends a POST to the `webhook_url`:

```json
{
  "job_id": "abc-123-def",
  "status": "completed",
  "result": { ... }
}
```

### SSRF Protection

- Private/internal IPs (`10.x`, `172.16-31.x`, `192.168.x`, `127.x`, `::1`) are **blocked by default**.
- `http://` URLs are blocked unless `webhook_allow_insecure` is `true` in `aiforge.yaml`.

### Webhook Configuration

```yaml
# aiforge.yaml
webhook_enabled: true
webhook_timeout: 10.0         # seconds for delivery timeout
webhook_allow_insecure: false  # allow http:// URLs (dev only)
```

---

## CORS

CORS middleware is enabled by default for the Playground UI. The allowed origin defaults to `http://localhost:8501` (the Playground port).

For production, restrict origins by overriding via environment variable:

```bash
export AIFORGE_CORS_ORIGINS=https://your-frontend.example.com
```

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

## Example API Calls

### Classify

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The login button is broken on mobile",
    "labels": ["bug", "feature_request", "question"]
  }'

# Response:
{
  "label": "bug",
  "confidence": 0.94,
  "all_scores": {"bug": 0.94, "feature_request": 0.03, "question": 0.03},
  "cost": 0.0002,
  "trace_id": "abc-123"
}
```

### Summarize

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long text to summarize...",
    "style": "bullet_points",
    "max_length": 500
  }'
```

### Document Q&A

```bash
# Ingest
curl -X POST http://localhost:8000/qa/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs/", "collection": "my_docs"}'

# Ask
curl -X POST http://localhost:8000/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does auth work?", "collection": "my_docs"}'
```

---

## Health Endpoint

```json
GET /health

{
  "status": "healthy",
  "version": "0.1.0",
  "provider": "openai",
  "model": "gpt-4o",
  "modules_loaded": ["summarize", "qa", "extract", "classify", "search", "generate", "chat", "moderate"],
  "vector_store": {
    "type": "chroma",
    "persist_dir": "./aiforge_data",
    "total_documents": 4521
  },
  "uptime_seconds": 3600,
  "total_requests": 142,
  "total_cost": 0.87
}
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY aiforge/ ./aiforge/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist

# ---- Runtime stage ----
FROM python:3.12-slim

LABEL maintainer="Kushang Patel"
LABEL description="AIForge — Plug-and-Play AI Toolkit"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIFORGE_API_HOST=0.0.0.0 \
    AIFORGE_API_PORT=8000

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[server,docs] && \
    rm -rf /tmp/*.whl

RUN useradd --create-home appuser
USER appuser

VOLUME /app/aiforge_data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["aiforge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - aiforge_data:/app/aiforge_data
    env_file:
      - .env
    command: aiforge serve --host 0.0.0.0 --port 8000

  playground:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - AIFORGE_API_BASE=http://api:8000
    command: aiforge playground --port 8501
    depends_on:
      - api

volumes:
  aiforge_data:
```

---

## Async Support

The API server calls `arun()` directly for seamless async performance:

```python
# aiforge/server/routes/summarize.py
@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    s = Summarizer(**req.model_dump())
    return await s.arun(req.text)
```

---

## See Also

- [Configuration](configuration.md) — Auth, CORS, and server settings
- [Playground](playground.md) — Interactive UI that connects to the API server
- [Quick Start](quickstart.md) — Getting started
