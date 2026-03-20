# Configuration

AIForge uses a layered configuration system. Later layers override earlier ones.

---

## Configuration Hierarchy

```
1. Defaults (hardcoded sensible defaults)
      ↓ overridden by
2. aiforge.yaml (project-level config file)
      ↓ overridden by
3. .env file
      ↓ overridden by
4. Environment variables (AIFORGE_*)
      ↓ overridden by
5. Constructor arguments (code-level)
```

Constructor arguments **always win**:

```python
s = Summarizer(model="gpt-4o-mini", temperature=0.5)
# Uses gpt-4o-mini regardless of what's in yaml/env
```

---

## Settings Model

```python
# aiforge/infra/config.py
from pydantic_settings import BaseSettings

class AIForgeSettings(BaseSettings):
    # Provider
    AIFORGE_PROVIDER: str = "openai"                    # openai | anthropic | ollama | azure
    AIFORGE_MODEL: str = "gpt-4o"
    AIFORGE_TEMPERATURE: float = 0.0
    AIFORGE_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Provider credentials
    AIFORGE_OPENAI_API_KEY: str = ""
    AIFORGE_ANTHROPIC_API_KEY: str = ""
    AIFORGE_VOYAGE_API_KEY: str = ""                    # Required for Anthropic embeddings (Voyage AI)
    AIFORGE_AZURE_OPENAI_ENDPOINT: str = ""
    AIFORGE_AZURE_OPENAI_API_KEY: str = ""
    AIFORGE_OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Vector store
    AIFORGE_VECTOR_STORE: str = "chroma"                # chroma | pgvector | pinecone | qdrant
    AIFORGE_CHROMA_PERSIST_DIR: str = "./aiforge_data"

    # pgvector
    AIFORGE_PGVECTOR_DSN: str = ""                       # e.g. postgresql://user:pass@host:5432/db
    AIFORGE_PGVECTOR_COLLECTION: str = "default"

    # Pinecone
    AIFORGE_PINECONE_API_KEY: str = ""
    AIFORGE_PINECONE_INDEX: str = ""

    # Qdrant
    AIFORGE_QDRANT_URL: str = ""                         # e.g. http://localhost:6333
    AIFORGE_QDRANT_API_KEY: str = ""
    AIFORGE_QDRANT_COLLECTION: str = ""

    # Observability
    AIFORGE_TRACING_ENABLED: bool = False
    AIFORGE_TRACING_PROVIDER: str = "langsmith"         # langsmith | otel | console
    AIFORGE_LANGSMITH_API_KEY: str = ""
    AIFORGE_COST_TRACKING_ENABLED: bool = True
    AIFORGE_COST_WARN_THRESHOLD: float = 1.0

    # Rate limiting
    AIFORGE_RATE_LIMIT_RPM: int = 500                   # requests per minute
    AIFORGE_RATE_LIMIT_TPM: int = 150000                # tokens per minute

    # Server
    AIFORGE_API_HOST: str = "0.0.0.0"
    AIFORGE_API_PORT: int = 8000
    AIFORGE_API_KEY: str = ""                           # If set, enables Bearer token auth
    AIFORGE_CORS_ORIGINS: str = "http://localhost:8501"  # Comma-separated or JSON array

    class Config:
        env_file = ".env"
        yaml_file = "aiforge.yaml"
```

---

## Example `aiforge.yaml`

```yaml
# aiforge.yaml
provider: openai
model: gpt-4o
temperature: 0
embedding_model: text-embedding-3-small
vector_store: chroma                # chroma | pgvector | pinecone | qdrant
chroma_persist_dir: ./aiforge_data  # only used when vector_store=chroma

# pgvector_dsn: postgresql://user:pass@localhost:5432/aiforge
# pgvector_collection: default
# pinecone_index: my-index           # prefer API key in .env
# qdrant_url: http://localhost:6333
# qdrant_collection: default

# Provider credentials (prefer env vars for secrets)
# openai_api_key: sk-...      # DON'T put keys here — use env vars

# Module defaults
summarize:
  style: concise
  max_length: 500

qa:
  chunk_size: 1500
  chunk_overlap: 200
  retrieval_k: 6
  max_iterations: 3

classify:
  multi_label: false

chat:
  memory: buffer
  memory_k: 20

# Observability
tracing:
  enabled: false
  provider: langsmith           # "langsmith" | "otel" | "console"

cost_tracking:
  enabled: true
  warn_threshold: 1.00          # warn if single call > $1
```

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

```bash
# .env
AIFORGE_VECTOR_STORE=chroma
AIFORGE_CHROMA_PERSIST_DIR=./aiforge_data
```

### pgvector (PostgreSQL)

```bash
pip install aiforge[pgvector]
```

```bash
# .env
AIFORGE_VECTOR_STORE=pgvector
AIFORGE_PGVECTOR_DSN=postgresql://user:pass@localhost:5432/aiforge
AIFORGE_PGVECTOR_COLLECTION=default   # optional — maps to table name
```

The adapter auto-creates the `vector` extension, collection table, and IVFFlat index on first use. Each collection maps to a separate table prefixed with `aiforge_`.

### Pinecone

```bash
pip install aiforge[pinecone]
```

```bash
# .env
AIFORGE_VECTOR_STORE=pinecone
AIFORGE_PINECONE_API_KEY=pk-...
AIFORGE_PINECONE_INDEX=my-index
```

Each AIForge "collection" maps to a Pinecone **namespace** within the configured index. Text is stored in `metadata["_text"]` since Pinecone is a pure-vector store.

### Qdrant

```bash
pip install aiforge[qdrant]
```

```bash
# .env
AIFORGE_VECTOR_STORE=qdrant
AIFORGE_QDRANT_URL=http://localhost:6333
AIFORGE_QDRANT_API_KEY=               # optional — only needed for Qdrant Cloud
AIFORGE_QDRANT_COLLECTION=default     # optional
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

- [Providers](providers.md) — Provider-specific configuration
- [Cost Tracking](cost-tracking.md) — Cost tracking and pricing configuration
- [API Server](api-server.md) — Server-specific settings (auth, CORS, shutdown)
