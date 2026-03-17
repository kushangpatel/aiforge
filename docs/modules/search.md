# Module: Search

`aiforge.search` — Semantic Search

---

## Purpose

Index any dataset and search it semantically. Auto-embeds, auto-indexes, returns ranked results with scores.

---

## Public API

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="products")

# Index documents
ss.index([
    {"id": "1", "text": "Wireless noise-cancelling headphones", "metadata": {"price": 299}},
    {"id": "2", "text": "Portable Bluetooth speaker with bass boost", "metadata": {"price": 79}},
    {"id": "3", "text": "USB-C fast charging cable 6ft", "metadata": {"price": 12}},
])

# Or index from files
ss.index_files("./product_descriptions/")

# Search
results = ss.query("something to listen to music on the go", top_k=5)
for r in results.hits:
    print(f"{r.score:.3f} | {r.text} | {r.metadata}")
# 0.923 | Portable Bluetooth speaker... | {"price": 79}
# 0.891 | Wireless noise-cancelling...  | {"price": 299}

# Search with metadata filter
results = ss.query("audio device", top_k=5, filter={"price": {"$lt": 100}})
```

### Async Support

```python
await ss.aindex(documents)
results = await ss.aquery("search query")
```

---

## Schemas

### Document

```python
class SearchDocument(BaseModel):
    id: str
    text: str
    metadata: dict = {}
```

### Index

```python
class IndexRequest(BaseModel):
    documents: list[SearchDocument]
    collection: str = "default"

class IndexResponse(BaseModel):
    indexed_count: int
    collection: str
    total_documents: int
```

### Search

```python
class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5
    filter: dict | None = None
    include_embeddings: bool = False

class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict

class SearchResponse(BaseModel):
    hits: list[SearchHit]
    query: str
    total_searched: int
    cost: float                         # embedding cost for query
    trace_id: str

    def to_text(self) -> str:
        return "\n\n".join(hit.text for hit in self.hits)
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-SRC-001 | Index documents | P0 | Accept text + metadata, embed, store in ChromaDB |
| FR-SRC-002 | Semantic query | P0 | Return ranked results by similarity |
| FR-SRC-003 | Metadata filtering | P1 | Filter results by metadata fields |
| FR-SRC-004 | Index from files | P1 | Auto-load and chunk files |
| FR-SRC-005 | Multiple collections | P1 | Support named collections |

---

## File Structure

```
aiforge/modules/search/
├── __init__.py
├── module.py          # SemanticSearch class
├── schemas.py         # Request/Response schemas
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic search example
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /search/index` and `POST /search/query` endpoints
- [Cost Tracking](../cost-tracking.md) — Embedding cost tracking
