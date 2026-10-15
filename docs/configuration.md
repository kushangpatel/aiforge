# Configuration

AIForge uses a **layered configuration system** built on
[Pydantic Settings v2](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
Every setting can be controlled via YAML, `.env`, environment variables, or
direct constructor arguments — with a clear, deterministic priority order.

---

## Table of Contents

- [Configuration Hierarchy](#configuration-hierarchy)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
  - [aiforge.yaml](#aiforgeyaml)
  - [.env](#env)
- [Settings Reference](#settings-reference)
  - [Provider & Model](#provider--model)
  - [API Keys (Secrets)](#api-keys-secrets)
  - [Ollama](#ollama)
  - [Azure OpenAI](#azure-openai)
  - [Google Gemini](#google-gemini)
  - [Amazon Bedrock](#amazon-bedrock)
  - [Mistral AI](#mistral-ai)
  - [Vector Store](#vector-store)
  - [Retrieval & Chunking](#retrieval--chunking)
  - [Rate Limiting](#rate-limiting)
  - [Cost Tracking](#cost-tracking)
  - [Request Cache](#request-cache)
  - [Webhooks](#webhooks)
  - [Tracing / Observability](#tracing--observability)
  - [Logging](#logging)
  - [API Server](#api-server)
  - [CORS](#cors)
  - [Playground](#playground)
  - [Model Router](#model-router)
  - [Custom Prompts](#custom-prompts)
  - [Custom Pricing](#custom-pricing)
  - [Custom Context Windows](#custom-context-windows)
- [Complete Environment Variable Reference](#complete-environment-variable-reference)
- [Validators](#validators)
- [Settings Loading Internals](#settings-loading-internals)
- [Using Settings in Code](#using-settings-in-code)
- [Vector Store Backends](#vector-store-backends)
  - [ChromaDB (default)](#chromadb-default)
  - [pgvector (PostgreSQL)](#pgvector-postgresql)
  - [Pinecone](#pinecone)
  - [Qdrant](#qdrant)
  - [Custom Vector Stores](#custom-vector-stores)
- [Plugin System](#plugin-system)
  - [Registry API](#registry-api)
  - [Built-in Registrations](#built-in-registrations)
  - [Custom Module](#custom-module)
  - [Custom Provider](#custom-provider)
- [Scaffolding with CLI](#scaffolding-with-cli)
- [See Also](#see-also)

---

## Configuration Hierarchy

Settings are resolved in order from **lowest** to **highest** priority.
Higher layers silently override lower ones — no merge, pure replacement:

```
1. Hardcoded defaults  (AIForgeSettings field defaults)
      ↓  overridden by
2. aiforge.yaml        (project-level config file in CWD)
      ↓  overridden by
3. .env file           (secrets & API keys ONLY)
      ↓  overridden by
4. Environment vars    (AIFORGE_* shell variables)
      ↓  overridden by
5. Constructor args    (Python kwargs — highest priority)
```

**Convention:** `.env` holds **secrets and API keys only**. All other
configuration belongs in `aiforge.yaml`.

Constructor arguments **always win**:

```python
from aiforge import Summarizer

# Uses gpt-4o-mini regardless of what's in yaml, .env, or shell env
s = Summarizer(model="gpt-4o-mini", temperature=0.5)
```

---

## Quick Start

```bash
# 1. Generate default config files
aiforge init
# Creates:  aiforge.yaml, .env.example, .gitignore

# 2. Copy secrets template and fill in your key
cp .env.example .env
# Edit .env → set AIFORGE_OPENAI_API_KEY=sk-...

# 3. (Optional) Customise non-secret settings
# Edit aiforge.yaml → change provider, model, vector store, etc.

# 4. Verify
python -c "from aiforge.infra.config import get_settings; s = get_settings(); print(s.PROVIDER, s.MODEL)"
# openai gpt-4o
```

---

## Configuration Files

### aiforge.yaml

The primary configuration file. Place it in your project root (the directory
where you run `aiforge` commands). All keys are **flat** (no nesting, except
for structured fields like `custom_prompts` and `custom_pricing`).

The YAML loader reads each top-level key, uppercases it, prefixes it with
`AIFORGE_`, and temporarily sets it as an environment variable so Pydantic
Settings picks it up. This means `provider: openai` in YAML becomes
`AIFORGE_PROVIDER=openai`.

```yaml
# aiforge.yaml — all non-secret settings go here

# ── Provider & Model ─────────────────────────────────
provider: openai                    # openai | anthropic | ollama | azure | gemini | bedrock | mistral
model: gpt-4o                      # default LLM model
temperature: 0.0                    # 0.0–2.0
prompt_cache: system                # system | auto | off  (Anthropic / Bedrock prompt caching)
embedding_model: text-embedding-3-small

# ── Vector Store ──────────────────────────────────────
vector_store: chroma                # chroma | pgvector | pinecone | qdrant
chroma_persist_dir: ./aiforge_data
# pgvector_dsn: postgresql://user:pass@localhost:5432/aiforge
# pinecone_index: my-index
# qdrant_url: http://localhost:6333

# ── Rate Limiting ─────────────────────────────────────
rate_limit_rpm: 500                 # requests per minute
rate_limit_tpm: 150000              # tokens per minute

# ── Cost Tracking ─────────────────────────────────────
cost_tracking_enabled: true
cost_warn_threshold: 1.00           # warn if single call > $1

# ── Request Cache ─────────────────────────────────────
cache_enabled: true
cache_ttl: 3600                     # seconds
cache_backend: memory               # memory | redis
cache_max_entries: 1000
# cache_redis_url: redis://localhost:6379/0

# ── Webhooks ──────────────────────────────────────────
webhook_enabled: true
webhook_timeout: 10.0               # delivery timeout in seconds
webhook_allow_insecure: false       # allow http:// URLs (dev only)

# ── Tracing / Observability ──────────────────────────
tracing_enabled: false
tracing_backend: langsmith          # langsmith | otel | console
langsmith_project: aiforge
# otel_exporter_endpoint: http://localhost:4317

# ── Logging ───────────────────────────────────────────
log_level: INFO                     # DEBUG | INFO | WARNING | ERROR
log_format: json                    # json | console

# ── API Server ────────────────────────────────────────
api_host: "0.0.0.0"
api_port: 8000
# api_key: my-secret-token          # prefer .env for secrets

# ── Playground ────────────────────────────────────────
playground_port: 8501

# ── Model Router ──────────────────────────────────────
model_router_enabled: true
default_quality: medium             # high | medium | low
prefer_local: false                 # prefer Ollama models at same tier

# ── Custom Pricing (USD per 1M tokens) ───────────────
# custom_pricing:
#   my-custom-model:
#     input: 1.50
#     output: 5.00

# ── Custom Context Windows ───────────────────────────
# custom_context_windows:
#   my-custom-model: 32000

# ── Custom Prompt Overrides ──────────────────────────
# custom_prompts:
#   summarize:
#     system: "You are a concise summariser."
#   classify:
#     system: "You classify text into categories."
```

### .env

The `.env` file holds **secrets only** — API keys, tokens, and credentials.
Pydantic Settings reads it automatically if present in the working directory.

```bash
# .env  — NEVER commit this file to version control

# ── OpenAI ──
AIFORGE_OPENAI_API_KEY=sk-...

# ── Anthropic ──
# AIFORGE_ANTHROPIC_API_KEY=ant-...
# AIFORGE_VOYAGE_API_KEY=pa-...

# ── Azure OpenAI ──
# AIFORGE_AZURE_OPENAI_API_KEY=...
# AIFORGE_AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/

# ── Google Gemini / Mistral ──
# AIFORGE_GOOGLE_API_KEY=...
# AIFORGE_MISTRAL_API_KEY=...

# ── Amazon Bedrock ── credentials come from the AWS chain, not from AIForge

# ── Pinecone ──
# AIFORGE_PINECONE_API_KEY=pk-...

# ── Qdrant ──
# AIFORGE_QDRANT_API_KEY=...

# ── LangSmith (tracing) ──
# AIFORGE_LANGSMITH_API_KEY=ls-...

# ── API Server Auth ──
# AIFORGE_API_KEY=your-secret-bearer-token
```

---

## Settings Reference

All settings use the `AIFORGE_` prefix as an environment variable. In `aiforge.yaml`, use the **lowercase** name without the prefix (e.g., `provider` not `AIFORGE_PROVIDER`).

### Provider & Model

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `PROVIDER` | `str` | `"openai"` | `AIFORGE_PROVIDER` | LLM provider: `openai`, `anthropic`, `ollama`, `azure`, `gemini`, `bedrock`, `mistral`, or any registered custom provider |
| `MODEL` | `str` | `"gpt-4o"` | `AIFORGE_MODEL` | Default LLM model name for the chosen provider (`gemini`, `bedrock` and `mistral` have their own defaults) |
| `TEMPERATURE` | `float` | `0.0` | `AIFORGE_TEMPERATURE` | Sampling temperature (validated: 0.0–2.0) |
| `MAX_TOKENS` | `int \| None` | `None` | `AIFORGE_MAX_TOKENS` | Hard token cap per LLM response; `None` = use model default |
| `PROMPT_CACHE` | `"system" \| "auto" \| "off"` | `"system"` | `AIFORGE_PROMPT_CACHE` | Anthropic and Bedrock prompt caching: cache the system prompt, also the conversation, or nothing (see [Providers](providers.md#prompt-caching)) |
| `EMBEDDING_MODEL` | `str` | `"text-embedding-3-small"` | `AIFORGE_EMBEDDING_MODEL` | Embedding model (validated against PROVIDER compatibility) |

**Provider-compatible embedding models:**

| Provider | Supported Embeddings |
|----------|---------------------|
| `openai` | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| `anthropic` | `voyage-3`, `voyage-3-lite`, `voyage-large-2`, `voyage-code-2` |
| `ollama` | `nomic-embed-text`, `mxbai-embed-large`, `all-minilm` |
| `azure` | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| `gemini` | `gemini-embedding-2`, `gemini-embedding-001` |
| `bedrock` | `amazon.titan-embed-text-v2:0`, `amazon.titan-embed-text-v1` |
| `mistral` | `mistral-embed`, `mistral-embed-2312`, `codestral-embed`, `codestral-embed-2505` |

### API Keys (Secrets)

All keys should go in `.env`, **not** in `aiforge.yaml`.

| Field | Type | Default | Env Var | Required When |
|-------|------|---------|---------|---------------|
| `OPENAI_API_KEY` | `str \| None` | `None` | `AIFORGE_OPENAI_API_KEY` | `PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `str \| None` | `None` | `AIFORGE_ANTHROPIC_API_KEY` | `PROVIDER=anthropic` |
| `VOYAGE_API_KEY` | `str \| None` | `None` | `AIFORGE_VOYAGE_API_KEY` | `PROVIDER=anthropic` with Voyage embeddings |
| `GOOGLE_API_KEY` | `str \| None` | `None` | `AIFORGE_GOOGLE_API_KEY` | `PROVIDER=gemini` |
| `MISTRAL_API_KEY` | `str \| None` | `None` | `AIFORGE_MISTRAL_API_KEY` | `PROVIDER=mistral` |
| `PINECONE_API_KEY` | `str \| None` | `None` | `AIFORGE_PINECONE_API_KEY` | `VECTOR_STORE=pinecone` |
| `QDRANT_API_KEY` | `str \| None` | `None` | `AIFORGE_QDRANT_API_KEY` | `VECTOR_STORE=qdrant` (Qdrant Cloud) |
| `LANGSMITH_API_KEY` | `str \| None` | `None` | `AIFORGE_LANGSMITH_API_KEY` | `TRACING_BACKEND=langsmith` |
| `API_KEY` | `str \| None` | `None` | `AIFORGE_API_KEY` | REST API auth — empty = auth disabled |

If a required key is missing, AIForge logs a **warning** (not an error) at startup. LLM calls will fail at runtime until the key is provided.

### Ollama

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `str` | `"http://localhost:11434"` | `AIFORGE_OLLAMA_BASE_URL` | Ollama server URL |

```yaml
provider: ollama
model: llama3
ollama_base_url: http://localhost:11434
embedding_model: nomic-embed-text
```

Ollama models have **zero cost** in the pricing table — local inference only.

### Azure OpenAI

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | `str \| None` | `None` | `AIFORGE_AZURE_OPENAI_API_KEY` | Azure API key |
| `AZURE_OPENAI_ENDPOINT` | `str \| None` | `None` | `AIFORGE_AZURE_OPENAI_ENDPOINT` | Endpoint URL (e.g., `https://<resource>.openai.azure.com/`) |
| `AZURE_OPENAI_DEPLOYMENT` | `str \| None` | `None` | `AIFORGE_AZURE_OPENAI_DEPLOYMENT` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `str` | `"2024-02-01"` | `AIFORGE_AZURE_OPENAI_API_VERSION` | API version |

```yaml
provider: azure
model: gpt-4o
azure_openai_deployment: my-gpt4o-deployment
```

```bash
# .env
AIFORGE_AZURE_OPENAI_API_KEY=...
AIFORGE_AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
```

### Google Gemini

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `GOOGLE_API_KEY` | `str \| None` | `None` | `AIFORGE_GOOGLE_API_KEY` | Gemini Developer API key (falls back to `GOOGLE_API_KEY` / `GEMINI_API_KEY`) |

```yaml
provider: gemini                 # model: gemini-3.8-flash, embedding_model: gemini-embedding-2
```

### Amazon Bedrock

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `AWS_REGION` | `str \| None` | `None` | `AIFORGE_AWS_REGION` | Bedrock region; `None` uses the AWS default region |
| `AWS_PROFILE` | `str \| None` | `None` | `AIFORGE_AWS_PROFILE` | Named profile from `~/.aws/config` |

AWS credentials come from the standard AWS chain (environment variables,
`~/.aws/credentials`, SSO, instance or container roles). AIForge has no
setting for them and never stores them.

```yaml
provider: bedrock                # model: global.anthropic.claude-sonnet-5-5
aws_region: us-east-1            # embedding_model: amazon.titan-embed-text-v2:0
# aws_profile: my-profile
```

### Mistral AI

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `MISTRAL_API_KEY` | `str \| None` | `None` | `AIFORGE_MISTRAL_API_KEY` | Mistral API key (falls back to `MISTRAL_API_KEY`) |

```yaml
provider: mistral                # model: mistral-medium-3-5, embedding_model: mistral-embed
```

`gemini`, `bedrock` and `mistral` use the default models shown above when
`MODEL` / `EMBEDDING_MODEL` are not set in any source. See
[Providers](providers.md#google-gemini) for model lists and prices.

### Vector Store

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `VECTOR_STORE` | `str` | `"chroma"` | `AIFORGE_VECTOR_STORE` | Backend: `chroma`, `pgvector`, `pinecone`, `qdrant`, or registered name |
| `CHROMA_PERSIST_DIR` | `str` | `"./aiforge_data"` | `AIFORGE_CHROMA_PERSIST_DIR` | ChromaDB persistence directory |
| `PGVECTOR_DSN` | `str \| None` | `None` | `AIFORGE_PGVECTOR_DSN` | PostgreSQL connection string |
| `PGVECTOR_COLLECTION` | `str` | `"default"` | `AIFORGE_PGVECTOR_COLLECTION` | pgvector table name |
| `PINECONE_API_KEY` | `str \| None` | `None` | `AIFORGE_PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX` | `str \| None` | `None` | `AIFORGE_PINECONE_INDEX` | Pinecone index name |
| `PINECONE_ENVIRONMENT` | `str \| None` | `None` | `AIFORGE_PINECONE_ENVIRONMENT` | Pinecone environment (deprecated; v3 SDK resolves automatically) |
| `QDRANT_URL` | `str \| None` | `None` | `AIFORGE_QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | `str \| None` | `None` | `AIFORGE_QDRANT_API_KEY` | Qdrant API key (Cloud only) |
| `QDRANT_COLLECTION` | `str \| None` | `None` | `AIFORGE_QDRANT_COLLECTION` | Qdrant collection name |

See [Vector Store Backends](#vector-store-backends) below for setup details.

### Retrieval & Chunking

Defaults for [`DocQA`](modules/qa.md#retrieval-modes) and
[`SemanticSearch`](modules/search.md#retrieval-modes). The matching constructor
arguments (`retrieval_mode=`, `rerank=`, `rerank_top_n=`, `fetch_k=`,
`chunking=`) override them per instance.

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `RETRIEVAL_MODE` | `str` | `"vector"` | `AIFORGE_RETRIEVAL_MODE` | `vector` (dense only), `hybrid` (dense + BM25 fused with RRF) or `keyword` (BM25 only) |
| `RRF_K` | `int` | `60` | `AIFORGE_RRF_K` | Reciprocal Rank Fusion damping constant for `hybrid` mode |
| `RERANK` | `bool` | `False` | `AIFORGE_RERANK` | Rerank retrieved candidates (with the LLM reranker unless a custom one is passed) |
| `RERANK_TOP_N` | `int \| None` | `None` | `AIFORGE_RERANK_TOP_N` | Hits kept after reranking; `None` keeps `retrieval_k` (DocQA) / `top_k` (SemanticSearch) |
| `FETCH_K` | `int` | `20` | `AIFORGE_FETCH_K` | Candidates fetched from each retriever before fusion / reranking |
| `CHUNKING` | `str` | `"recursive"` | `AIFORGE_CHUNKING` | DocQA chunking strategy: `recursive`, `markdown` (heading-aware) or `token` (tiktoken-sized) |

```yaml
# aiforge.yaml
retrieval_mode: hybrid
rerank: true
rerank_top_n: 5
fetch_k: 30
chunking: markdown
```

### Rate Limiting

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `RATE_LIMIT_RPM` | `int` | `500` | `AIFORGE_RATE_LIMIT_RPM` | Requests per minute cap |
| `RATE_LIMIT_TPM` | `int` | `150_000` | `AIFORGE_RATE_LIMIT_TPM` | Tokens per minute cap |

The rate limiter uses **fixed-window buckets** (refill every 60 seconds). When
limits are exceeded, requests **block** (sleep until capacity is available)
rather than failing with an error.

```yaml
rate_limit_rpm: 500
rate_limit_tpm: 150000
```

Limits can be updated at runtime:

```python
from aiforge.infra.rate_limiter import RateLimiter
RateLimiter.get_instance().update_limits(rpm=1000, tpm=300000)
```

### Cost Tracking

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `COST_TRACKING_ENABLED` | `bool` | `True` | `AIFORGE_COST_TRACKING_ENABLED` | Enable cost tracking for all LLM calls |
| `COST_WARN_THRESHOLD` | `float` | `1.0` | `AIFORGE_COST_WARN_THRESHOLD` | Log a WARNING when a single call exceeds this USD amount |

Every module response includes `cost` (USD), `input_tokens`, and `output_tokens`. The `CostTracker` singleton aggregates costs across all calls in the process:

```python
from aiforge.infra.cost_tracker import CostTracker

tracker = CostTracker.get_session()
print(tracker.summary())
# {
#   "total_cost_usd": 1.23,
#   "total_tokens": 45000,
#   "call_count": 42,
#   "by_module": {"Summarizer": 20, "Classifier": 22},
#   "by_model": {"gpt-4o": 30, "gpt-4o-mini": 12}
# }
```

The API server's `GET /stats` endpoint exposes these same metrics over HTTP.

**Built-in pricing table** covers OpenAI (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4.5, gpt-3.5-turbo, o1, o1-mini, o3-mini, o3), Anthropic (claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5 and earlier 4.x/5.x models), and Ollama models (zero cost). See [Custom Pricing](#custom-pricing) to override or add models.

### Request Cache

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `CACHE_ENABLED` | `bool` | `True` | `AIFORGE_CACHE_ENABLED` | Enable LLM response caching |
| `CACHE_TTL` | `int` | `3600` | `AIFORGE_CACHE_TTL` | Time-to-live in seconds |
| `CACHE_BACKEND` | `str` | `"memory"` | `AIFORGE_CACHE_BACKEND` | Backend: `memory` or `redis` |
| `CACHE_MAX_ENTRIES` | `int` | `1000` | `AIFORGE_CACHE_MAX_ENTRIES` | Max entries for in-memory cache (LRU eviction) |
| `CACHE_REDIS_URL` | `str` | `"redis://localhost:6379/0"` | `AIFORGE_CACHE_REDIS_URL` | Redis connection URL |

Cache keys are SHA-256 hashes of the request fingerprint (model + messages + temperature + max_tokens + module + provider). Identical requests within the TTL window return cached results with **zero cost and zero latency**.

**Memory backend:** In-process dict with LRU eviction and TTL expiration. Thread-safe.

**Redis backend:** Requires `redis` package. Keys are prefixed with `aiforge:cache:`. Uses `SETEX` for TTL.

```yaml
cache_enabled: true
cache_backend: redis
cache_redis_url: redis://localhost:6379/0
cache_ttl: 7200       # 2 hours
```

Cache statistics:

```python
from aiforge.infra.cache import RequestCache
cache = RequestCache.get_instance()
print(cache.stats())
# {"enabled": true, "size": 42, "hits": 120, "misses": 38, "hit_rate": 0.76, "ttl_seconds": 3600}
```

### Webhooks

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `WEBHOOK_ENABLED` | `bool` | `True` | `AIFORGE_WEBHOOK_ENABLED` | Enable webhook delivery for async jobs |
| `WEBHOOK_TIMEOUT` | `float` | `10.0` | `AIFORGE_WEBHOOK_TIMEOUT` | HTTP delivery timeout per attempt (seconds) |
| `WEBHOOK_ALLOW_INSECURE` | `bool` | `False` | `AIFORGE_WEBHOOK_ALLOW_INSECURE` | Allow `http://` webhook URLs (dev only!) |

Webhooks are used by `/qa/ingest` and `/search/index` for async job notifications. See [API Server — Async Jobs & Webhooks](api-server.md#async-jobs--webhooks) for details.

### Tracing / Observability

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `TRACING_ENABLED` | `bool` | `False` | `AIFORGE_TRACING_ENABLED` | Enable trace export |
| `TRACING_BACKEND` | `str` | `"langsmith"` | `AIFORGE_TRACING_BACKEND` | Backend: `langsmith`, `otel`, or `console` |
| `LANGSMITH_API_KEY` | `str \| None` | `None` | `AIFORGE_LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_PROJECT` | `str` | `"aiforge"` | `AIFORGE_LANGSMITH_PROJECT` | LangSmith project name |
| `OTEL_EXPORTER_ENDPOINT` | `str \| None` | `None` | `AIFORGE_OTEL_EXPORTER_ENDPOINT` | OpenTelemetry OTLP endpoint |

**Tracing backends:**

| Backend | Description | Requirements |
|---------|-------------|-------------|
| `langsmith` | LangSmith trace viewer — sets `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | `LANGSMITH_API_KEY` |
| `otel` | OpenTelemetry — exports spans via OTLP | `pip install aiforge-kit[otel]`, `OTEL_EXPORTER_ENDPOINT` |
| `console` | Prints trace info to stdout (debug/dev only) | None |

```yaml
tracing_enabled: true
tracing_backend: langsmith
langsmith_project: my-project
```

```bash
# .env
AIFORGE_LANGSMITH_API_KEY=ls-...
```

### Logging

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `LOG_LEVEL` | `str` | `"INFO"` | `AIFORGE_LOG_LEVEL` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT` | `str` | `"json"` | `AIFORGE_LOG_FORMAT` | Output format: `json` (machine-parseable) or `console` (human-readable with colours) |

AIForge uses [structlog](https://www.structlog.org/) for all logging. The logging pipeline includes:

- **Timestamp:** ISO 8601, UTC
- **Secret redaction:** The `_strip_secrets` processor detects and masks API keys (OpenAI `sk-`, Anthropic `ant-`, Voyage `pa-`, long base64 tokens) and sensitive field names (`api_key`, `secret`, `token`, `password`, `authorization`)
- **Stack info:** Full stack traces for errors
- **Stdlib routing:** Third-party library logs are routed through the same structlog pipeline

```yaml
log_level: DEBUG
log_format: console    # colourised, human-readable for local dev
```

### API Server

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `API_KEY` | `str \| None` | `None` | `AIFORGE_API_KEY` | Bearer token for REST API auth; empty = auth disabled |
| `API_HOST` | `str` | `"0.0.0.0"` | `AIFORGE_API_HOST` | Server bind address |
| `API_PORT` | `int` | `8000` | `AIFORGE_API_PORT` | Server port |

See [API Server](api-server.md) for full endpoint documentation and [API Authentication](api-auth.md) for auth details.

### CORS

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `CORS_ORIGINS` | `str` | `"*"` | `AIFORGE_CORS_ORIGINS` | Comma-separated allowed CORS origins, or `"*"` for all |

```bash
# Production — specific origins
export AIFORGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Development — allow all
export AIFORGE_CORS_ORIGINS=*
```

When `*` is used, `allow_credentials` is set to `false` (browser security requirement). With specific origins, credentials are forwarded.

### Playground

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `PLAYGROUND_PORT` | `int` | `8501` | `AIFORGE_PLAYGROUND_PORT` | Streamlit playground port |

### Model Router

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `MODEL_ROUTER_ENABLED` | `bool` | `True` | `AIFORGE_MODEL_ROUTER_ENABLED` | Enable automatic model selection by quality tier |
| `DEFAULT_QUALITY` | `str` | `"medium"` | `AIFORGE_DEFAULT_QUALITY` | Default quality tier: `high`, `medium`, or `low` |
| `PREFER_LOCAL` | `bool` | `False` | `AIFORGE_PREFER_LOCAL` | Prefer local Ollama models over cloud at the same tier |
| `CUSTOM_MODELS` | `list[dict]` | `[]` | `AIFORGE_CUSTOM_MODELS` | Custom models added to the built-in catalog |

The model router maps quality tiers to specific models:

| Quality | Typical Model | Best For |
|---------|-------------|----------|
| `high` | `gpt-4o`, `claude-sonnet-5-5` | Complex reasoning, nuanced tasks |
| `medium` | `gpt-4o-mini`, `claude-haiku-5-5` | Balanced cost/quality |
| `low` | `gpt-3.5-turbo`, local Ollama models | High-volume, simple tasks |

```python
from aiforge import Classifier

# Use the "high" tier model for this module
c = Classifier(quality="high")
```

### Custom Prompts

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `CUSTOM_PROMPTS` | `dict[str, dict[str, str]]` | `{}` | `AIFORGE_CUSTOM_PROMPTS` | Per-module system prompt overrides |

Override the system prompt for any module without modifying code:

```yaml
# aiforge.yaml
custom_prompts:
  summarize:
    system: "You are a medical professional. Summarise clinical documents."
  classify:
    system: "You are a customer-support triage agent."
```

Modules check `CUSTOM_PROMPTS[module_name]["system"]` at runtime and use it instead of the built-in prompt.

### Custom Pricing

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `CUSTOM_PRICING` | `dict[str, dict[str, float]]` | `{}` | `AIFORGE_CUSTOM_PRICING` | USD per 1M tokens — override or add model prices |

The built-in pricing table is a best-effort snapshot. Override any model's price or add new models:

```yaml
# aiforge.yaml
custom_pricing:
  gpt-4o:                    # override built-in
    input: 2.50
    output: 10.00
  my-custom-model:           # add new model
    input: 1.50
    output: 5.00
```

If a model is not in the built-in table and no override exists, cost is reported as `0.0` with a WARNING log.

**Cost formula:** `(input_price_per_1M × input_tokens / 1,000,000) + (output_price_per_1M × output_tokens / 1,000,000)`

**Fuzzy matching:** Date-suffix variants like `gpt-4o-2024-11-20` automatically resolve to the base `gpt-4o` pricing entry.

### Custom Context Windows

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `CUSTOM_CONTEXT_WINDOWS` | `dict[str, int]` | `{}` | `AIFORGE_CUSTOM_CONTEXT_WINDOWS` | Max-tokens overrides per model |

```yaml
# aiforge.yaml
custom_context_windows:
  my-custom-model: 32000
  gpt-4o: 200000             # override built-in 128K
```

Context window sizes are used by the Summarizer to decide between `stuff` and `map_reduce` strategies. The fallback for unknown models is `8,192`.

---

## Complete Environment Variable Reference

Every field in `AIForgeSettings` maps to an `AIFORGE_`-prefixed environment variable:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AIFORGE_PROVIDER` | `str` | `openai` | LLM provider |
| `AIFORGE_MODEL` | `str` | `gpt-4o` | Default LLM model |
| `AIFORGE_TEMPERATURE` | `float` | `0.0` | Sampling temperature (0.0–2.0) |
| `AIFORGE_MAX_TOKENS` | `int` | — | Hard token cap per response |
| `AIFORGE_PROMPT_CACHE` | `str` | `system` | Anthropic / Bedrock prompt caching: `system`, `auto` or `off` |
| `AIFORGE_EMBEDDING_MODEL` | `str` | `text-embedding-3-small` | Embedding model |
| `AIFORGE_OPENAI_API_KEY` | `str` | — | OpenAI API key |
| `AIFORGE_ANTHROPIC_API_KEY` | `str` | — | Anthropic API key |
| `AIFORGE_VOYAGE_API_KEY` | `str` | — | Voyage AI key |
| `AIFORGE_AZURE_OPENAI_API_KEY` | `str` | — | Azure OpenAI API key |
| `AIFORGE_AZURE_OPENAI_ENDPOINT` | `str` | — | Azure endpoint URL |
| `AIFORGE_AZURE_OPENAI_DEPLOYMENT` | `str` | — | Azure deployment name |
| `AIFORGE_AZURE_OPENAI_API_VERSION` | `str` | `2024-02-01` | Azure API version |
| `AIFORGE_OLLAMA_BASE_URL` | `str` | `http://localhost:11434` | Ollama server URL |
| `AIFORGE_GOOGLE_API_KEY` | `str` | — | Google Gemini API key |
| `AIFORGE_MISTRAL_API_KEY` | `str` | — | Mistral API key |
| `AIFORGE_AWS_REGION` | `str` | AWS default | Amazon Bedrock region |
| `AIFORGE_AWS_PROFILE` | `str` | — | AWS named profile for Bedrock |
| `AIFORGE_VECTOR_STORE` | `str` | `chroma` | Vector store backend |
| `AIFORGE_CHROMA_PERSIST_DIR` | `str` | `./aiforge_data` | ChromaDB directory |
| `AIFORGE_PGVECTOR_DSN` | `str` | — | PostgreSQL connection string |
| `AIFORGE_PGVECTOR_COLLECTION` | `str` | `default` | pgvector table name |
| `AIFORGE_PINECONE_API_KEY` | `str` | — | Pinecone API key |
| `AIFORGE_PINECONE_INDEX` | `str` | — | Pinecone index name |
| `AIFORGE_PINECONE_ENVIRONMENT` | `str` | — | Pinecone environment (deprecated) |
| `AIFORGE_QDRANT_URL` | `str` | — | Qdrant server URL |
| `AIFORGE_QDRANT_API_KEY` | `str` | — | Qdrant API key |
| `AIFORGE_QDRANT_COLLECTION` | `str` | — | Qdrant collection name |
| `AIFORGE_RETRIEVAL_MODE` | `str` | `vector` | `vector`, `hybrid` or `keyword` |
| `AIFORGE_RRF_K` | `int` | `60` | RRF damping constant (hybrid mode) |
| `AIFORGE_RERANK` | `bool` | `false` | Rerank retrieved candidates |
| `AIFORGE_RERANK_TOP_N` | `int` | — | Hits kept after reranking |
| `AIFORGE_FETCH_K` | `int` | `20` | Candidates fetched before fusion / reranking |
| `AIFORGE_CHUNKING` | `str` | `recursive` | `recursive`, `markdown` or `token` |
| `AIFORGE_RATE_LIMIT_RPM` | `int` | `500` | Requests per minute |
| `AIFORGE_RATE_LIMIT_TPM` | `int` | `150000` | Tokens per minute |
| `AIFORGE_COST_TRACKING_ENABLED` | `bool` | `true` | Enable cost tracking |
| `AIFORGE_COST_WARN_THRESHOLD` | `float` | `1.0` | Per-call cost warning (USD) |
| `AIFORGE_CACHE_ENABLED` | `bool` | `true` | Enable request cache |
| `AIFORGE_CACHE_TTL` | `int` | `3600` | Cache time-to-live (seconds) |
| `AIFORGE_CACHE_BACKEND` | `str` | `memory` | Cache backend |
| `AIFORGE_CACHE_MAX_ENTRIES` | `int` | `1000` | Max in-memory cache entries |
| `AIFORGE_CACHE_REDIS_URL` | `str` | `redis://localhost:6379/0` | Redis URL |
| `AIFORGE_WEBHOOK_ENABLED` | `bool` | `true` | Enable webhooks |
| `AIFORGE_WEBHOOK_TIMEOUT` | `float` | `10.0` | Webhook timeout (seconds) |
| `AIFORGE_WEBHOOK_ALLOW_INSECURE` | `bool` | `false` | Allow http:// webhooks |
| `AIFORGE_CORS_ORIGINS` | `str` | `*` | CORS allowed origins |
| `AIFORGE_TRACING_ENABLED` | `bool` | `false` | Enable tracing |
| `AIFORGE_TRACING_BACKEND` | `str` | `langsmith` | Tracing backend |
| `AIFORGE_LANGSMITH_API_KEY` | `str` | — | LangSmith API key |
| `AIFORGE_LANGSMITH_PROJECT` | `str` | `aiforge` | LangSmith project |
| `AIFORGE_OTEL_EXPORTER_ENDPOINT` | `str` | — | OTLP endpoint |
| `AIFORGE_LOG_LEVEL` | `str` | `INFO` | Log level |
| `AIFORGE_LOG_FORMAT` | `str` | `json` | Log format |
| `AIFORGE_API_KEY` | `str` | — | REST API Bearer token |
| `AIFORGE_API_HOST` | `str` | `0.0.0.0` | Server bind host |
| `AIFORGE_API_PORT` | `int` | `8000` | Server port |
| `AIFORGE_PLAYGROUND_PORT` | `int` | `8501` | Playground port |
| `AIFORGE_MODEL_ROUTER_ENABLED` | `bool` | `true` | Model router enabled |
| `AIFORGE_DEFAULT_QUALITY` | `str` | `medium` | Default quality tier |
| `AIFORGE_PREFER_LOCAL` | `bool` | `false` | Prefer Ollama models |
| `AIFORGE_CUSTOM_MODELS` | `list` | `[]` | Custom model catalog |
| `AIFORGE_CUSTOM_PROMPTS` | `dict` | `{}` | Per-module prompt overrides |
| `AIFORGE_CUSTOM_PRICING` | `dict` | `{}` | Model pricing overrides |
| `AIFORGE_CUSTOM_CONTEXT_WINDOWS` | `dict` | `{}` | Model context-window overrides |

---

## Validators

`AIForgeSettings` runs several validators during construction. If validation fails, Pydantic raises `ValidationError` with a descriptive message.

| Validator | Field(s) | Rule | Error |
|-----------|----------|------|-------|
| `validate_provider` | `PROVIDER` | Must be a built-in provider (`openai`, `anthropic`, `ollama`, `azure`, `gemini`, `bedrock`, `mistral`) or registered via `register_provider()` | `ValueError` |
| `validate_vector_store` | `VECTOR_STORE` | Must not be empty; stripped and lowercased | `ValueError` |
| `validate_cache_backend` | `CACHE_BACKEND` | Must be `memory` or `redis` | `ValueError` |
| `validate_default_quality` | `DEFAULT_QUALITY` | Must be `high`, `medium`, or `low` | `ValueError` |
| `validate_temperature` | `TEMPERATURE` | Must be in `[0.0, 2.0]` | `ValueError` |
| `apply_provider_defaults` | `MODEL`, `EMBEDDING_MODEL` | For `gemini`, `bedrock` and `mistral`, fills in the provider's default models when they are not set in any source | — |
| `validate_provider_embedding_combo` | `PROVIDER`, `EMBEDDING_MODEL` | Embedding model must be compatible with the selected provider | `ValueError` |
| `validate_api_keys_present` | Provider-specific keys | **Warning only** — if `PROVIDER=openai` but `OPENAI_API_KEY` is empty, a `UserWarning` is emitted | Warning (not error) |

---

## Settings Loading Internals

### How `get_settings()` works

```python
@lru_cache(maxsize=1)
def get_settings() -> AIForgeSettings:
    """Return the global settings singleton."""
    yaml_path = Path("aiforge.yaml")
    if yaml_path.exists():
        settings = AIForgeSettings.from_yaml(yaml_path)
    else:
        settings = AIForgeSettings()

    # Apply runtime overrides from config
    apply_pricing_overrides(settings.CUSTOM_PRICING)
    apply_context_window_overrides(settings.CUSTOM_CONTEXT_WINDOWS)
    return settings
```

- **Cached:** `@lru_cache(maxsize=1)` — first call loads config, subsequent calls return the same instance
- **YAML loading:** `from_yaml()` reads the YAML file, flattens keys to `AIFORGE_*` env vars, temporarily sets them so Pydantic picks them up, then restores the original env state
- **Pricing/context-window overrides:** Applied once at first call
- **Cache clearing:** Tests use `get_settings.cache_clear()` to reset

### How modules use settings

```python
from aiforge import Summarizer

# Option A: Use global settings (from aiforge.yaml / .env / env vars)
s = Summarizer()

# Option B: Override specific settings via kwargs
s = Summarizer(model="gpt-4o-mini", temperature=0.5)
# This creates a PRIVATE AIForgeSettings(**kwargs) — does not affect global settings

# Option C: Pass a quality tier (uses model router)
s = Summarizer(quality="high")
```

When `**kwargs` are passed to a module constructor:
1. A **private** `AIForgeSettings(**kwargs)` is created (not shared with other modules)
2. The kwargs override all other config sources for that instance
3. The global `get_settings()` singleton is unaffected

When no kwargs are passed:
1. The module uses the global singleton from `get_settings()`

---

## Vector Store Backends

AIForge supports four vector store backends out of the box. The default is **ChromaDB** (zero infrastructure). Swap backends via `AIFORGE_VECTOR_STORE` — all modules (`DocQA`, `SemanticSearch`) auto-route through the vector store factory.

### ChromaDB (default)

No extra install needed — ChromaDB is a core dependency.

```yaml
# aiforge.yaml
vector_store: chroma
chroma_persist_dir: ./aiforge_data
```

Data is persisted to disk at `chroma_persist_dir`. Each AIForge collection maps to a ChromaDB collection.

### pgvector (PostgreSQL)

```bash
pip install aiforge-kit[pgvector]
```

```yaml
# aiforge.yaml
vector_store: pgvector
pgvector_dsn: postgresql://user:pass@localhost:5432/aiforge
pgvector_collection: default
```

The adapter auto-creates the `vector` extension, collection table, and IVFFlat index on first use. Each collection maps to a separate table prefixed with `aiforge_`.

### Pinecone

```bash
pip install aiforge-kit[pinecone]
```

```yaml
# aiforge.yaml
vector_store: pinecone
pinecone_index: my-index
```

```bash
# .env
AIFORGE_PINECONE_API_KEY=pk-...
```

Each AIForge "collection" maps to a Pinecone **namespace** within the configured index. Text is stored in `metadata["_text"]` since Pinecone is a pure-vector store.

### Qdrant

```bash
pip install aiforge-kit[qdrant]
```

```yaml
# aiforge.yaml
vector_store: qdrant
qdrant_url: http://localhost:6333
qdrant_collection: default
```

```bash
# .env (only needed for Qdrant Cloud)
AIFORGE_QDRANT_API_KEY=...
```

The adapter auto-creates the Qdrant collection (cosine similarity) on first use if it doesn't exist.

### Custom Vector Stores

Register any backend that implements the `VectorStoreAdapter` ABC:

```python
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.registry import register_vectorstore

class WeaviateStore(VectorStoreAdapter):
    def add_documents(self, texts, embeddings, metadatas, ids): ...
    def query(self, embedding, top_k=5, filter=None, include_embeddings=False): ...
    def delete_collection(self, name): ...
    def list_collections(self): ...
    def collection_stats(self, name): ...

register_vectorstore("weaviate", WeaviateStore)
```

Then set `AIFORGE_VECTOR_STORE=weaviate` in your config.

---

## Plugin System

AIForge includes a **registry-based plugin system** for extending the toolkit
with custom modules, providers, and vector stores — all using a consistent
`register_*()` / `get_*()` pattern.

All built-in modules and vector stores are auto-registered on import.

### Registry API

```python
from aiforge.registry import (
    # Modules
    register_module, get_module, list_modules,
    # Providers
    register_provider, get_provider, list_providers,
    # Vector stores
    register_vectorstore, get_vectorstore, list_vectorstores,
)
```

All registry names are **normalised** to lowercase and stripped of whitespace.

### Built-in Registrations

After `import aiforge`, the following are auto-registered:

| Registry | Names |
|----------|-------|
| **Modules** | `summarize`, `qa`, `extract`, `classify`, `search`, `generate`, `chat`, `moderate` |
| **Vector stores** | `chroma`, `pinecone`, `qdrant`, `pgvector` (if respective clients installed) |
| **Providers** | `openai`, `anthropic`, `ollama`, `azure`, `gemini`, `bedrock`, `mistral` (if respective extras installed) |

### Custom Module

Extend `BaseModule` and register it for use in pipelines and discovery:

```python
from aiforge.core.types import BaseModule, ModuleResponse
from aiforge.registry import register_module

class SentimentAnalyzer(BaseModule):
    def run(self, text: str, **kwargs) -> ModuleResponse:
        result = self._call_llm(f"Classify sentiment: {text}")
        return ModuleResponse(text=result.text.strip(), cost=result.cost)

    async def arun(self, text: str, **kwargs) -> ModuleResponse:
        result = await self._acall_llm(f"Classify sentiment: {text}")
        return ModuleResponse(text=result.text.strip(), cost=result.cost)

register_module("sentiment", SentimentAnalyzer)

# Look it up by name
cls = get_module("sentiment")          # returns SentimentAnalyzer class
print(list_modules())                   # [..., 'sentiment', 'summarize', ...]
```

Registered custom modules work in `Pipeline` alongside built-in modules:

```python
from aiforge import Pipeline, Summarizer

pipe = Pipeline([
    ("sentiment", SentimentAnalyzer()),
    ("summarize", Summarizer()),
])
result = pipe.run("I absolutely love this product!")
```

### Custom Provider

Register an LLM + embedding provider pair so any module can use it via config:

```python
from aiforge.core.provider import LLMProvider, EmbeddingProvider
from aiforge.registry import register_provider

class InternalGatewayLLM(LLMProvider):
    def complete(self, messages, **kwargs): ...
    async def complete_async(self, messages, **kwargs): ...
    async def stream(self, messages, **kwargs): ...

class InternalGatewayEmbedding(EmbeddingProvider):
    def embed(self, texts): ...
    def embed_query(self, text): ...

register_provider("internal_gateway", InternalGatewayLLM, InternalGatewayEmbedding)
```

Then set `AIFORGE_PROVIDER=internal_gateway` or pass at runtime:

```python
from aiforge import Summarizer
s = Summarizer(provider="internal_gateway")
```

### Type Safety

- `register_module()` requires a `BaseModule` subclass — raises `TypeError` otherwise
- `register_vectorstore()` requires a `VectorStoreAdapter` subclass — raises `TypeError` otherwise
- `get_module()` / `get_vectorstore()` / `get_provider()` raise `KeyError` with a helpful message listing available names when a name isn't found

---

## Scaffolding with CLI

```bash
# Generate default config files
aiforge init
# Creates:
#   aiforge.yaml       — project config with sensible defaults
#   .env.example        — template for secrets
#   .gitignore          — includes .env and aiforge_data/

# Overwrite existing files
aiforge init --force
```

---

## See Also

- [API Authentication](api-auth.md) — Bearer token setup, error responses, security properties
- [API Server](api-server.md) — Server-specific settings, endpoints, deployment
- [Providers](providers.md) — Provider-specific configuration details
- [Cost Tracking](cost-tracking.md) — Cost tracking and pricing deep-dive
- [Modules](modules/) — Detailed SDK documentation for each module
- [Quick Start](quickstart.md) — Getting started
