# Configuration

AIForge uses a layered configuration system. Later layers override earlier ones.

---

## Configuration Hierarchy

```
1. Defaults (hardcoded sensible defaults)
      ↓ overridden by
2. aiforge.yaml (project-level config file — main config)
      ↓ overridden by
3. .env file (secrets & API keys ONLY)
      ↓ overridden by
4. Environment variables (AIFORGE_*)
      ↓ overridden by
5. Constructor arguments (code-level)
```

> **Convention:** `.env` holds **secrets and API keys only**. All other
> configuration belongs in `aiforge.yaml`. See `aiforge.yaml.example`
> for the full list of supported settings.

Constructor arguments **always win**:

```python
s = Summarizer(model="gpt-4o-mini", temperature=0.5)
# Uses gpt-4o-mini regardless of what's in yaml/env
```

---

## Settings Model

All fields use the `AIFORGE_` env-var prefix (added automatically by Pydantic Settings).
Field names in `config.py` do **not** include the prefix.

```python
# aiforge/infra/config.py  (simplified — see source for full model)
from pydantic_settings import BaseSettings, SettingsConfigDict

class AIForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIFORGE_",
        env_file=".env",
        extra="ignore",
    )

    # Provider & model
    PROVIDER: str = "openai"                    # openai | anthropic | ollama | azure
    MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # API keys (secrets — keep in .env)
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    VOYAGE_API_KEY: str | None = None
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    PINECONE_API_KEY: str | None = None
    QDRANT_API_KEY: str | None = None
    LANGSMITH_API_KEY: str | None = None
    API_KEY: str | None = None                  # Bearer token for REST API auth

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Vector store
    VECTOR_STORE: str = "chroma"                # chroma | pgvector | pinecone | qdrant
    CHROMA_PERSIST_DIR: str = "./aiforge_data"
    PGVECTOR_DSN: str | None = None
    PGVECTOR_COLLECTION: str = "default"
    PINECONE_INDEX: str | None = None
    QDRANT_URL: str | None = None
    QDRANT_COLLECTION: str | None = None

    # Rate limiting
    RATE_LIMIT_RPM: int = 500
    RATE_LIMIT_TPM: int = 150_000

    # Cost tracking
    COST_TRACKING_ENABLED: bool = True
    COST_WARN_THRESHOLD: float = 1.0            # USD per single call

    # Request cache
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600                       # seconds
    CACHE_BACKEND: str = "memory"               # memory | redis
    CACHE_MAX_ENTRIES: int = 1000
    CACHE_REDIS_URL: str = "redis://localhost:6379/0"

    # Webhooks
    WEBHOOK_ENABLED: bool = True
    WEBHOOK_TIMEOUT: float = 10.0               # seconds
    WEBHOOK_ALLOW_INSECURE: bool = False         # allow http:// URLs (dev only)

    # Tracing / observability
    TRACING_ENABLED: bool = False
    TRACING_BACKEND: str = "langsmith"          # langsmith | otel | console
    LANGSMITH_PROJECT: str = "aiforge"
    OTEL_EXPORTER_ENDPOINT: str | None = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"                    # json | console

    # API server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Playground
    PLAYGROUND_PORT: int = 8501

    # Model router
    MODEL_ROUTER_ENABLED: bool = True
    DEFAULT_QUALITY: str = "medium"             # high | medium | low
    PREFER_LOCAL: bool = False
    CUSTOM_MODELS: list[dict] = []

    # Custom prompt overrides
    CUSTOM_PROMPTS: dict[str, dict[str, str]] = {}
```

---

## Example `aiforge.yaml`

All keys are **flat** (no nesting). The YAML loader auto-adds the `AIFORGE_` prefix.
Secrets (API keys) belong in `.env`, not here.

```yaml
# aiforge.yaml

# Provider & model
provider: openai
model: gpt-4o
temperature: 0.0
embedding_model: text-embedding-3-small

# Vector store
vector_store: chroma                # chroma | pgvector | pinecone | qdrant
chroma_persist_dir: ./aiforge_data
# pgvector_dsn: postgresql://user:pass@localhost:5432/aiforge
# pgvector_collection: default
# pinecone_index: my-index
# qdrant_url: http://localhost:6333
# qdrant_collection: default

# Rate limiting
rate_limit_rpm: 500
rate_limit_tpm: 150000

# Cost tracking
cost_tracking_enabled: true
cost_warn_threshold: 1.00

# Request cache
cache_enabled: true
cache_ttl: 3600                     # seconds
cache_backend: memory               # memory | redis
cache_max_entries: 1000
# cache_redis_url: redis://localhost:6379/0

# Webhooks
webhook_enabled: true
webhook_timeout: 10.0               # delivery timeout in seconds
webhook_allow_insecure: false        # allow http:// URLs (dev only)

# Tracing / observability
tracing_enabled: false
tracing_backend: langsmith          # langsmith | otel | console
langsmith_project: aiforge

# Logging
log_level: INFO
log_format: json                    # json | console

# API server
api_host: "0.0.0.0"
api_port: 8000

# Playground
playground_port: 8501

# Model router
model_router_enabled: true
default_quality: medium             # high | medium | low
prefer_local: false
```

> See `aiforge.yaml.example` for the full annotated reference with all options including
> custom models, custom prompts, Ollama, and all vector store backends.

---

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `AIFORGE_PROVIDER` | No | LLM provider | `openai` |
| `AIFORGE_MODEL` | No | LLM model | `gpt-4o` |
| `AIFORGE_TEMPERATURE` | No | LLM temperature | `0` |
| `AIFORGE_EMBEDDING_MODEL` | No | Embedding model | `text-embedding-3-small` |
| `AIFORGE_OPENAI_API_KEY` | Yes (if OpenAI) | OpenAI API key | — |
| `AIFORGE_ANTHROPIC_API_KEY` | Yes (if Anthropic) | Anthropic API key | — |
| `AIFORGE_VOYAGE_API_KEY` | Yes (if Anthropic embeddings) | Voyage AI API key | — |
| `AIFORGE_AZURE_OPENAI_ENDPOINT` | Yes (if Azure) | Azure endpoint | — |
| `AIFORGE_AZURE_OPENAI_API_KEY` | Yes (if Azure) | Azure API key | — |
| `AIFORGE_OLLAMA_BASE_URL` | No | Ollama server URL | `http://localhost:11434` |
| `AIFORGE_VECTOR_STORE` | No | Vector store backend | `chroma` |
| `AIFORGE_CHROMA_PERSIST_DIR` | No | ChromaDB data directory | `./aiforge_data` |
| `AIFORGE_PGVECTOR_DSN` | Yes (if pgvector) | PostgreSQL connection string | — |
| `AIFORGE_PGVECTOR_COLLECTION` | No | pgvector table name | `default` |
| `AIFORGE_PINECONE_API_KEY` | Yes (if Pinecone) | Pinecone API key | — |
| `AIFORGE_PINECONE_INDEX` | Yes (if Pinecone) | Pinecone index name | — |
| `AIFORGE_QDRANT_URL` | Yes (if Qdrant) | Qdrant server URL | — |
| `AIFORGE_QDRANT_API_KEY` | No | Qdrant Cloud API key | — |
| `AIFORGE_QDRANT_COLLECTION` | No | Qdrant collection name | — |
| `AIFORGE_TRACING_ENABLED` | No | Enable tracing | `false` |
| `AIFORGE_LANGSMITH_API_KEY` | No | LangSmith key | — |
| `AIFORGE_COST_TRACKING_ENABLED` | No | Enable cost tracking | `true` |
| `AIFORGE_COST_WARN_THRESHOLD` | No | Warn if cost > N per call | `1.00` |
| `AIFORGE_RATE_LIMIT_RPM` | No | Requests per minute | `500` |
| `AIFORGE_RATE_LIMIT_TPM` | No | Tokens per minute | `150000` |
| `AIFORGE_API_HOST` | No | API server host | `0.0.0.0` |
| `AIFORGE_API_PORT` | No | API server port | `8000` |
| `AIFORGE_API_KEY` | No | Bearer token for REST API auth | — (disabled) |
| `AIFORGE_CORS_ORIGINS` | No | Allowed CORS origins | `http://localhost:8501` |

---

## Vector Store Backends

AIForge supports four vector store backends. The default is **ChromaDB** (zero infrastructure). Swap backends via `AIFORGE_VECTOR_STORE` — all modules (DocQA, SemanticSearch) auto-route through the factory.

### ChromaDB (default)

No extra install needed — ChromaDB is a core dependency.

```yaml
# aiforge.yaml
vector_store: chroma
chroma_persist_dir: ./aiforge_data
```

### pgvector (PostgreSQL)

```bash
pip install aiforge[pgvector]
```

```yaml
# aiforge.yaml
vector_store: pgvector
pgvector_dsn: postgresql://user:pass@localhost:5432/aiforge
pgvector_collection: default          # optional — maps to table name
```

The adapter auto-creates the `vector` extension, collection table, and IVFFlat index on first use. Each collection maps to a separate table prefixed with `aiforge_`.

### Pinecone

```bash
pip install aiforge[pinecone]
```

```yaml
# aiforge.yaml
vector_store: pinecone
pinecone_index: my-index
```

```bash
# .env (secret)
AIFORGE_PINECONE_API_KEY=pk-...
```

Each AIForge "collection" maps to a Pinecone **namespace** within the configured index. Text is stored in `metadata["_text"]` since Pinecone is a pure-vector store.

### Qdrant

```bash
pip install aiforge[qdrant]
```

```yaml
# aiforge.yaml
vector_store: qdrant
qdrant_url: http://localhost:6333
qdrant_collection: default            # optional
```

```bash
# .env (secret — only needed for Qdrant Cloud)
# AIFORGE_QDRANT_API_KEY=
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

AIForge includes a **registry-based plugin system** that lets you extend the toolkit with custom modules, providers, and vector stores — all using a consistent `register_*()` / `get_*()` pattern.

All built-in modules and vector stores are auto-registered on import, so they're always discoverable via the registry.

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

All registry names are **normalised** to lowercase and stripped of leading/trailing whitespace.

### Built-in Registrations

After `import aiforge`, the following are auto-registered:

| Registry | Names |
|----------|-------|
| **Modules** | `summarize`, `qa`, `extract`, `classify`, `search`, `generate`, `chat`, `moderate` |
| **Vector stores** | `chroma`, `pinecone`, `qdrant`, `pgvector` (if respective clients are installed) |
| **Providers** | `openai`, `anthropic`, `ollama`, `azure` (if respective extras are installed) |

### Custom Module

Extend `BaseModule` and register it for use in pipelines or for discovery via `get_module()`:

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

# Later, look it up by name
cls = get_module("sentiment")         # returns SentimentAnalyzer class
print(list_modules())                  # [..., 'sentiment', 'summarize', ...]
```

Registered custom modules can be used in `Pipeline` alongside built-in modules:

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

Then set `AIFORGE_PROVIDER=internal_gateway` in your config, or pass it at runtime:

```python
from aiforge import Summarizer
s = Summarizer(provider="internal_gateway")
```

### Custom Vector Store

See the [Custom Vector Stores](#custom-vector-stores) section above for an example. Any class that implements the `VectorStoreAdapter` ABC can be registered with `register_vectorstore()` and activated via `AIFORGE_VECTOR_STORE`.

### Type Safety

- `register_module()` requires the class to be a subclass of `BaseModule` — raises `TypeError` otherwise.
- `register_vectorstore()` requires the class to be a subclass of `VectorStoreAdapter` — raises `TypeError` otherwise.
- `get_module()` / `get_vectorstore()` / `get_provider()` raise `KeyError` with a helpful message listing available names if the name isn't found.

---

## Config Loading Order

```python
# 1. Hardcoded defaults (in AIForgeSettings)
# 2. aiforge.yaml (if exists in cwd or parent dirs)
# 3. .env file
# 4. Environment variables
# 5. Constructor arguments

# Constructor args always win:
s = Summarizer(model="gpt-4o-mini", temperature=0.5)
# Uses gpt-4o-mini regardless of what's in yaml/env
```

---

## Scaffolding with CLI

```bash
# Generate config files
aiforge init
# Creates:
#   aiforge.yaml       — project config with sensible defaults
#   .env.example        — template for environment variables
#   .gitignore          — includes .env and aiforge_data/
```

---

## Pricing Overrides

The built-in pricing table is a best-effort snapshot. Override any model price:

```yaml
# aiforge.yaml
pricing:
  gpt-4o: {input: 2.50, output: 10.00}  # per 1M tokens — override built-in
```

If a model isn't in the table and no override is set, cost is reported as `0.0` with a WARNING log.

---

## See Also

- [API Authentication](api-auth.md) — Bearer token setup, error responses, security properties
- [Providers](providers.md) — Provider-specific configuration
- [Cost Tracking](cost-tracking.md) — Cost tracking and pricing configuration
- [API Server](api-server.md) — Server-specific settings (auth, CORS, shutdown)
