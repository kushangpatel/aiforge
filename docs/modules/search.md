# Module: Search

`aiforge.modules.search` — Semantic search powered by embeddings and ChromaDB. Index any dataset (structured dicts, files, or directories), then query it with natural language. No LLM calls — only embedding calls — making it fast and cheap. Similarity scores are cosine-based, ranked highest first.

---

## Installation

The search module is included with the base `aiforge` install. ChromaDB is a core dependency.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import SemanticSearch
from aiforge import (                   # types
    SearchResponse,
    SearchHit,
    IndexResponse,
    SearchDocument,
    IndexRequest,
    SearchRequest,
)
```

---

## Quick Start

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="products")

# Index some documents
ss.index([
    {"id": "1", "text": "Wireless noise-cancelling headphones", "metadata": {"price": 299}},
    {"id": "2", "text": "Portable Bluetooth speaker with bass boost", "metadata": {"price": 79}},
    {"id": "3", "text": "USB-C fast charging cable 6ft", "metadata": {"price": 12}},
])

# Semantic search
results = ss.query("something to listen to music on the go", top_k=3)
for hit in results.hits:
    print(f"{hit.score:.3f} | {hit.text}")
# 0.923 | Portable Bluetooth speaker with bass boost
# 0.891 | Wireless noise-cancelling headphones
# 0.342 | USB-C fast charging cable 6ft
```

---

## Constructor

```python
SemanticSearch(
    collection: str = "default",
    persist_dir: str = "./aiforge_data",
    top_k: int = 5,
    store: VectorStoreAdapter | None = None,
    # Inherited from BaseModule:
    quality: str | None = None,
    provider: str | None = None,
    **kwargs,  # AIForgeSettings overrides
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str` | `"default"` | Vector store collection name. Leading/trailing whitespace is stripped. Must not be empty. |
| `persist_dir` | `str` | `"./aiforge_data"` | Data directory for ChromaDB persistence. Ignored by non-ChromaDB backends. |
| `top_k` | `int` | `5` | Default number of results per query. Can be overridden per-call. Must be 1–100. |
| `store` | `VectorStoreAdapter \| None` | `None` | Pre-built vector store adapter for dependency injection or testing. When `None`, a ChromaDB store is built automatically. |
| `quality` | `str \| None` | `None` | Quality tier for embedding model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | Embedding provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `EMBEDDING_MODEL="text-embedding-3-large"`. |

### Validation Rules

The constructor raises `ValueError` for:
- Empty or whitespace-only `collection`
- `top_k` outside the range `[1, 100]`

### Examples

```python
from aiforge import SemanticSearch

# Defaults — ChromaDB at ./aiforge_data, collection "default"
ss = SemanticSearch()

# Named collection
ss = SemanticSearch(collection="products")

# More results by default
ss = SemanticSearch(collection="faq", top_k=10)

# Custom persistence directory
ss = SemanticSearch(collection="prod", persist_dir="/data/vectordb")

# Use a specific embedding model
ss = SemanticSearch(collection="docs", EMBEDDING_MODEL="text-embedding-3-large")

# Quality-based routing
ss = SemanticSearch(collection="search", quality="high")
```

---

## Methods

### `index(documents, *, batch_size=100) -> IndexResponse`

Embed and upsert documents into the collection. Documents with the same `id` are updated (upsert semantics).

```python
def index(
    self,
    documents: list[dict | SearchDocument],
    *,
    batch_size: int = 100,
) -> IndexResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents` | `list[dict \| SearchDocument]` | *(required)* | Documents to index. Accepts `SearchDocument` instances or plain dicts with `id`, `text`, and optional `metadata` keys. |
| `batch_size` | `int` | `100` | Number of documents per embedding API call. Reduce if hitting provider rate limits. |

**Returns:** [`IndexResponse`](#indexresponse)

**Raises:** `ValueError` if `documents` is empty.

**Behavior:**
1. Each document is coerced to a `SearchDocument` (if a plain dict)
2. Documents are embedded in batches of `batch_size`
3. Embeddings + metadata are upserted into ChromaDB
4. Costs are calculated from estimated token counts (1 token ≈ 4 chars)

```python
ss = SemanticSearch(collection="products")

# Index from dicts
resp = ss.index([
    {"id": "p1", "text": "Wireless headphones", "metadata": {"price": 299, "category": "audio"}},
    {"id": "p2", "text": "Bluetooth speaker", "metadata": {"price": 79, "category": "audio"}},
    {"id": "p3", "text": "USB-C cable", "metadata": {"price": 12, "category": "accessories"}},
])
print(resp.indexed_count)     # 3
print(resp.total_documents)   # 3
print(resp.collection)        # "products"
print(f"Cost: ${resp.cost:.6f}")

# Upsert — re-indexing with same IDs updates existing documents
resp = ss.index([
    {"id": "p1", "text": "Wireless noise-cancelling headphones (updated)", "metadata": {"price": 249}},
])
print(resp.indexed_count)     # 1
print(resp.total_documents)   # 3 (not 4 — upsert, not append)
```

---

### `aindex(documents, *, batch_size=100) -> IndexResponse`

Async version of `index()`. Uses the async embedding provider internally.

```python
async def aindex(
    self,
    documents: list[dict | SearchDocument],
    *,
    batch_size: int = 100,
) -> IndexResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents` | `list[dict \| SearchDocument]` | *(required)* | Documents to index. |
| `batch_size` | `int` | `100` | Docs per embedding batch. |

**Returns:** [`IndexResponse`](#indexresponse)

```python
resp = await ss.aindex([
    {"id": "1", "text": "Document text", "metadata": {"source": "api"}},
])
```

---

### `index_files(source, file_extensions=None) -> IndexResponse`

Load files from a path or directory and index them. Each file becomes one `SearchDocument` with a deterministic UUID v5 ID (based on file path + content prefix). Files that cannot be loaded are skipped with a warning.

```python
def index_files(
    self,
    source: str | list[str],
    file_extensions: list[str] | None = None,
) -> IndexResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| list[str]` | *(required)* | File path, directory path, or list of file paths. |
| `file_extensions` | `list[str] \| None` | `None` | Extension whitelist with leading dot (e.g. `[".txt", ".md"]`). `None` accepts all supported text types. |

**Returns:** [`IndexResponse`](#indexresponse) — returns `indexed_count=0` if no documents were loaded.

```python
ss = SemanticSearch(collection="docs")

# Index all supported files in a directory
resp = ss.index_files("./docs/")
print(f"Indexed {resp.indexed_count} files")

# Index only markdown files
resp = ss.index_files("./docs/", file_extensions=[".md"])

# Index specific files
resp = ss.index_files(["README.md", "CHANGELOG.md"])
```

---

### `query(query, *, top_k=None, filter=None, include_embeddings=False) -> SearchResponse`

Embed the query and retrieve the most similar documents from the collection.

```python
def query(
    self,
    query: str,
    *,
    top_k: int | None = None,
    filter: dict | None = None,
    include_embeddings: bool = False,
) -> SearchResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query text. Must not be empty or whitespace-only. |
| `top_k` | `int \| None` | `None` | Number of results to return. `None` uses the instance default (`self.default_top_k`). Must be 1–100. |
| `filter` | `dict \| None` | `None` | ChromaDB `where` clause for metadata filtering, e.g. `{"price": {"$lt": 100}}`. |
| `include_embeddings` | `bool` | `False` | When `True`, each `SearchHit` includes its raw embedding vector. |

**Returns:** [`SearchResponse`](#searchresponse)

**Raises:** `ValueError` if `query` is empty/whitespace or `top_k` is out of range.

**Scoring:** Similarity scores are computed as `1.0 - cosine_distance`, clamped to `[0.0, 1.0]`. Higher is more similar.

```python
ss = SemanticSearch(collection="products")

# Basic search
results = ss.query("audio equipment")
for hit in results.hits:
    print(f"  {hit.score:.3f} | {hit.id} | {hit.text}")
# 0.923 | p2 | Bluetooth speaker
# 0.891 | p1 | Wireless headphones

# More results
results = ss.query("electronics", top_k=20)

# Metadata filtering
results = ss.query("audio", filter={"price": {"$lt": 100}})
print(results.hits[0].text)      # "Bluetooth speaker" (price: 79)
print(results.hits[0].metadata)  # {"price": 79, "category": "audio"}

# Include embeddings for downstream processing
results = ss.query("audio", include_embeddings=True)
print(len(results.hits[0].embeddings))  # 1536 (for text-embedding-3-small)

# Metrics
print(results.query)           # "audio"
print(results.total_searched)  # 3
print(results.cost)            # 0.000001
print(results.latency_ms)     # 45.2
```

---

### `aquery(query, *, top_k=None, filter=None, include_embeddings=False) -> SearchResponse`

Async version of `query()`. Embeds the query asynchronously and runs ChromaDB operations in a thread pool to avoid blocking the event loop.

```python
async def aquery(
    self,
    query: str,
    *,
    top_k: int | None = None,
    filter: dict | None = None,
    include_embeddings: bool = False,
) -> SearchResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query text. |
| `top_k` | `int \| None` | `None` | Number of results. |
| `filter` | `dict \| None` | `None` | Metadata filter. |
| `include_embeddings` | `bool` | `False` | Include embedding vectors. |

**Returns:** [`SearchResponse`](#searchresponse)

```python
results = await ss.aquery("audio equipment", top_k=5)
```

---

### `list_collections() -> list[str]`

Return names of all collections in the vector store.

```python
def list_collections(self) -> list[str]
```

**Returns:** `list[str]`

```python
collections = ss.list_collections()
print(collections)  # ["products", "docs", "faq"]
```

---

### `collection_stats(collection=None) -> dict`

Return statistics for a collection.

```python
def collection_stats(self, collection: str | None = None) -> dict
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str \| None` | `None` | Collection name. Defaults to this instance's `collection_name`. |

**Returns:** `dict` with keys like `"chunks"`, `"name"`.

```python
stats = ss.collection_stats()
print(stats)  # {"name": "products", "chunks": 3, ...}

# Check a different collection
stats = ss.collection_stats("docs")
```

---

### `delete_collection(collection=None) -> None`

Delete a collection and all its data.

```python
def delete_collection(self, collection: str | None = None) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str \| None` | `None` | Collection to delete. Defaults to this instance's `collection_name`. |

```python
ss.delete_collection()               # delete current collection
ss.delete_collection("old-index")    # delete a specific collection
```

---

### `run(text, **kwargs) -> SearchResponse`

BaseModule interface method. Delegates to `query()`. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> SearchResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The search query. |
| `**kwargs` | | Forwarded to `query()` — accepts `top_k`, `filter`, `include_embeddings`. |

**Returns:** [`SearchResponse`](#searchresponse)

```python
result = ss.run("audio equipment", top_k=3)
```

---

### `arun(text, **kwargs) -> SearchResponse`

Async version of `run()`. Delegates to `aquery()`.

```python
async def arun(self, text: str, **kwargs) -> SearchResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Run multiple searches concurrently. Inherited from `BaseModule`.

```python
def run_batch(
    self,
    items: list[str],
    concurrency: int = 10,
    max_batch: int = 100,
) -> BatchResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `list[str]` | *(required)* | List of search queries. |
| `concurrency` | `int` | `10` | Max concurrent queries. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

```python
queries = ["headphones", "speakers", "cables", "microphones"]
batch = ss.run_batch(queries, concurrency=4)
for r in batch.results:
    top = r.hits[0] if r.hits else None
    print(f"  Top: {top.text if top else 'no results'} ({top.score:.3f})" if top else "  No results")
```

---

## Response Types

### `IndexResponse`

Returned by `index()`, `aindex()`, and `index_files()`.

```python
from aiforge import IndexResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `indexed_count` | `int` | Number of documents upserted in this call. |
| `collection` | `str` | Name of the vector store collection. |
| `total_documents` | `int` | Total documents now in the collection (including previous contents). |
| `cost` | `float` | Embedding cost in USD. |
| `input_tokens` | `int` | Estimated tokens embedded (1 token ≈ 4 characters). |
| `output_tokens` | `int` | Always `0` (embeddings have no output tokens). |
| `model` | `str` | Embedding model name used. |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end time in milliseconds. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_text()` | `str` | `"Indexed 3 documents into 'products' (3 total)"` |

```python
resp = ss.index([...])

print(resp.to_text())          # "Indexed 3 documents into 'products' (3 total)"
print(resp.indexed_count)      # 3
print(resp.total_documents)    # 3
print(resp.cost)               # 0.000012
print(resp.model)              # "text-embedding-3-small"
```

---

### `SearchResponse`

Returned by `query()`, `aquery()`, `run()`, and `arun()`.

```python
from aiforge import SearchResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `hits` | `list[SearchHit]` | Ranked results, highest score first. Empty list if no matches. |
| `query` | `str` | The query string that produced these results. |
| `total_searched` | `int` | Total documents in the searched collection. |
| `cost` | `float` | Embedding cost for the query in USD. |
| `input_tokens` | `int` | Estimated tokens for the query embedding. |
| `output_tokens` | `int` | Always `0`. |
| `model` | `str` | Embedding model name used. |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end search time in milliseconds. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_text()` | `str` | Returns the top hit's text for pipeline chaining. Empty string if no hits. |

```python
results = ss.query("audio equipment")

print(results.query)           # "audio equipment"
print(results.total_searched)  # 100
print(len(results.hits))       # 5

# Pipeline chaining — returns top hit text
print(results.to_text())       # "Portable Bluetooth speaker with bass boost"

# No results → empty string
empty = ss.query("xyz nonexistent")
print(empty.to_text())         # ""
```

---

### `SearchHit`

Represents a single ranked search result.

```python
from aiforge import SearchHit
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Document identifier. |
| `text` | `str` | Document text content. |
| `score` | `float` | Cosine similarity score in `[0.0, 1.0]`. Computed as `1.0 - cosine_distance`. Higher is more similar. |
| `metadata` | `dict` | Document metadata (whatever was stored during indexing). |
| `embeddings` | `list[float] \| None` | Raw embedding vector. Only populated when `include_embeddings=True` is passed to the query. |

```python
hit = results.hits[0]
print(hit.id)          # "p2"
print(hit.text)        # "Portable Bluetooth speaker with bass boost"
print(hit.score)       # 0.923
print(hit.metadata)    # {"price": 79, "category": "audio"}
print(hit.embeddings)  # None (unless include_embeddings=True)
```

---

### `SearchDocument`

Input schema for documents to be indexed.

```python
from aiforge import SearchDocument
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | *(required)* | Unique document identifier. Minimum 1 character. |
| `text` | `str` | *(required)* | Document text content. Minimum 1 character. |
| `metadata` | `dict` | `{}` | Arbitrary metadata dictionary. |

```python
# Create explicitly
doc = SearchDocument(id="p1", text="Wireless headphones", metadata={"price": 299})

# Or pass dicts directly to index() — they're coerced automatically
ss.index([{"id": "p1", "text": "Wireless headphones", "metadata": {"price": 299}}])
```

---

### Request Schemas (API Server)

These schemas are used by the API server endpoints, not typically by SDK consumers:

**`IndexRequest`** — `POST /search/index`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `documents` | `list[SearchDocument]` | *(required)* | Documents to index. |
| `collection` | `str` | `"default"` | Collection name. |

**`SearchRequest`** — `POST /search/query`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | *(required)* | Search query text. |
| `collection` | `str` | `"default"` | Collection name. |
| `top_k` | `int` | `5` | Number of results (1–100). |
| `filter` | `dict \| None` | `None` | Metadata filter. |
| `include_embeddings` | `bool` | `False` | Include embedding vectors. |

---

## Similarity Scoring

Scores are computed as `1.0 - cosine_distance`, clamped to `[0.0, 1.0]`:

| Score | Interpretation |
|-------|---------------|
| `0.9+` | Very high similarity — near-exact semantic match |
| `0.7–0.9` | High similarity — related content |
| `0.5–0.7` | Moderate similarity — loosely related |
| `< 0.5` | Low similarity — likely not relevant |

> **Note:** This metric assumes cosine distance (ChromaDB default). Other distance metrics (L2, inner product) would need different conversion.

---

## Metadata Filtering

ChromaDB `where` clauses can filter results by metadata fields:

```python
# Exact match
results = ss.query("audio", filter={"category": "audio"})

# Comparison operators
results = ss.query("audio", filter={"price": {"$lt": 100}})       # less than
results = ss.query("audio", filter={"price": {"$gte": 50}})       # greater than or equal
results = ss.query("audio", filter={"price": {"$ne": 0}})         # not equal

# Logical operators
results = ss.query("audio", filter={
    "$and": [
        {"category": "audio"},
        {"price": {"$lt": 200}},
    ]
})
```

See the [ChromaDB filtering docs](https://docs.trychroma.com/guides#filtering-results) for the full `where` clause syntax.

---

## Upsert Semantics

When indexing documents with `index()`, documents are upserted by their `id`:
- **New ID** → inserted
- **Existing ID** → text and metadata are updated, embedding is re-computed

This makes it safe to re-index the same documents after updates without creating duplicates.

When using `index_files()`, IDs are generated deterministically from `file_path + content[:64]` via UUID v5, so re-indexing the same files is idempotent.

---

## No LLM Calls

Unlike other AIForge modules, `SemanticSearch` makes **no LLM calls** — only embedding calls. This means:
- Lower cost (embedding is ~10–100x cheaper than LLM completion)
- Lower latency (embedding is ~50ms vs ~500ms+ for LLM)
- No need for a completion model — only an embedding model is required

---

## Instance Properties

After constructing a `SemanticSearch`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `ss.collection_name` | `str` | Collection name (stripped). |
| `ss.persist_dir` | `str` | ChromaDB data directory. |
| `ss.default_top_k` | `int` | Default results per query. |
| `ss.settings` | `AIForgeSettings` | Resolved configuration. |
| `ss.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Product Catalog Search

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="products", top_k=5)

# Index product catalog
ss.index([
    {"id": "h1", "text": "Wireless noise-cancelling headphones with 30hr battery", "metadata": {"price": 299, "category": "audio"}},
    {"id": "s1", "text": "Portable Bluetooth speaker with deep bass", "metadata": {"price": 79, "category": "audio"}},
    {"id": "c1", "text": "USB-C fast charging cable 6ft braided", "metadata": {"price": 12, "category": "accessories"}},
    {"id": "m1", "text": "Condenser microphone for podcast recording", "metadata": {"price": 149, "category": "audio"}},
    {"id": "k1", "text": "Mechanical keyboard with RGB lighting", "metadata": {"price": 89, "category": "peripherals"}},
])

# Natural language search
results = ss.query("something to listen to music on the go")
for hit in results.hits:
    print(f"  {hit.score:.3f} | ${hit.metadata['price']} | {hit.text}")

# Filtered search — only affordable audio
results = ss.query("audio equipment", filter={
    "$and": [
        {"category": "audio"},
        {"price": {"$lt": 100}},
    ]
})
```

### Document Knowledge Base

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="knowledge-base")

# Index files from a directory
resp = ss.index_files("./docs/", file_extensions=[".md", ".txt"])
print(f"Indexed {resp.indexed_count} files, cost: ${resp.cost:.6f}")

# Search the knowledge base
results = ss.query("how to configure authentication")
print(results.hits[0].text)
print(results.hits[0].metadata.get("source"))  # file path
```

### Pipeline Integration

Use `SemanticSearch` as part of a pipeline via `run()` / `arun()`:

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="faq")

# Pipeline-compatible — delegates to query()
result = ss.run("How do I reset my password?", top_k=3)
print(result.to_text())  # top hit's text — can feed into next pipeline stage
```

### Batch Search

```python
ss = SemanticSearch(collection="products")

queries = [
    "wireless headphones",
    "portable speaker",
    "charging cable",
    "gaming keyboard",
]

batch = ss.run_batch(queries, concurrency=4)

for i, r in enumerate(batch.results):
    top = r.hits[0] if r.hits else None
    print(f"  '{queries[i]}' → {top.text if top else 'no results'}")

print(f"Total cost: ${batch.total_cost:.6f}")
```

### Collection Management

```python
ss = SemanticSearch(collection="temp-index")

# Index some data
ss.index([{"id": "1", "text": "Test document"}])

# List all collections
print(ss.list_collections())  # ["temp-index", "products", ...]

# Check stats
stats = ss.collection_stats()
print(stats)  # {"name": "temp-index", "chunks": 1, ...}

# Clean up
ss.delete_collection()
```

### Async Usage

```python
import asyncio
from aiforge import SemanticSearch

async def main():
    ss = SemanticSearch(collection="async-demo")

    # Async index
    resp = await ss.aindex([
        {"id": "1", "text": "First document"},
        {"id": "2", "text": "Second document"},
    ])
    print(f"Indexed {resp.indexed_count}")

    # Async search
    results = await ss.aquery("first", top_k=1)
    print(results.hits[0].text)

asyncio.run(main())
```

### Include Raw Embeddings

```python
results = ss.query("audio equipment", include_embeddings=True)

for hit in results.hits:
    print(f"  {hit.id}: score={hit.score:.3f}, embedding_dim={len(hit.embeddings)}")
    # p2: score=0.923, embedding_dim=1536
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
EMBEDDING_MODEL: "text-embedding-3-small"
```

```python
ss = SemanticSearch(collection="docs")  # picks up aiforge.yaml automatically
results = ss.query("test")
print(results.model)  # "text-embedding-3-small"
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Empty `collection`, empty `query`, `top_k` out of range, empty `documents` list | `SemanticSearch(collection="")` |
| `ProviderError` | Embedding provider returns an error | Invalid API key |
| `RateLimitError` | Provider rate limit exceeded | Large batch indexing |
| `APITimeoutError` | Provider request timed out | Slow network |

```python
from aiforge.core.types import ProviderError

try:
    results = ss.query("test")
except ProviderError as e:
    print(f"Embedding failed: {e}")

try:
    ss.index([])  # empty list
except ValueError as e:
    print(f"Validation error: {e}")
    # "documents must not be empty"
```

---

## File Structure

```
aiforge/modules/search/
├── __init__.py        # Exports: SemanticSearch, SearchDocument, SearchHit, SearchResponse, IndexResponse, IndexRequest, SearchRequest
├── module.py          # SemanticSearch class (index, query, collection management)
├── schemas.py         # Pydantic models (SearchDocument, SearchHit, IndexResponse, SearchResponse, IndexRequest, SearchRequest)
└── widget.py          # Streamlit playground widget (Search + Index modes)
```

---

## SemanticSearch vs DocQA

Both modules use vector stores and embeddings, but they serve different
purposes:

| | SemanticSearch | DocQA |
|---|---|---|
| **Purpose** | Find the most similar documents | Answer a question in natural language |
| **LLM calls** | None — embedding-only | Yes — agentic LangGraph pipeline with iterative refinement |
| **Ingestion** | Pre-structured documents with explicit `id`, `text`, `metadata` | Raw files, directories, or git URLs (auto-loads, language-aware chunking) |
| **Output** | Ranked `hits` with similarity `score`, `id`, `text` | Synthesized answer (`text`), `sources`, `confidence`, optional `trace` |
| **Cost** | Embedding tokens only | Embedding tokens (ingest) + LLM tokens (ask) |
| **Use when** | You need a *ranked list* of matching passages | You need a *written answer* derived from documents |

Use **SemanticSearch** when you want to find documents and handle the results
yourself; use **DocQA** when you want a fully synthesized answer.

---

## See Also

- [Quick Start](../quickstart.md) — Basic search example
- [DocQA](qa.md) — Full RAG pipeline with agentic refinement (uses search internally)
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /search/index` and `POST /search/query` endpoints
- [API Server > Webhooks](../api-server.md#async-jobs--webhooks) — Async indexing with webhook notifications
- [Cost Tracking](../cost-tracking.md) — Embedding cost tracking
- [Configuration](../configuration.md) — Full settings reference
