# Module: DocQA

`aiforge.qa` — Document Q&A (RAG)

---

## Purpose

Upload documents → ask questions → get grounded answers with source citations. Full RAG pipeline with agentic refinement powered by LangGraph.

---

## Public API

```python
from aiforge import DocQA

qa = DocQA(
    collection="my_docs",      # ChromaDB collection name
    chunk_size=1500,           # characters per chunk
    chunk_overlap=200,         # overlap between chunks
    retrieval_k=6,             # chunks to retrieve per query
    max_iterations=3,          # agentic refinement loops
)

# Ingest documents
stats = qa.ingest("./docs/")                          # local directory
stats = qa.ingest("https://github.com/org/repo.git")  # git repo
stats = qa.ingest(["file1.py", "file2.md"])            # specific files
print(stats.chunks_indexed)    # 1,247
print(stats.files_processed)   # 89
print(stats.languages)         # {"python": 45, "markdown": 30, "go": 14}

# Collection management
collections = qa.list_collections()       # ["my_docs", "default"]
stats = qa.collection_stats("my_docs")   # {"chunks": 1247, "files": 89, "size_mb": 12.3}
qa.delete_collection("old_docs")          # remove collection and all its data

# Ask questions
answer = qa.ask("How does authentication work?")
print(answer.text)             # "Authentication uses JWT tokens..."
print(answer.sources)          # ["src/auth.py:45-67", "docs/api.md:12-30"]
print(answer.confidence)       # 0.92
print(answer.iterations)       # 2 (needed 1 refinement)
print(answer.trace)            # [{"step": "classify", ...}, {"step": "retrieve", ...}, ...]
print(answer.cost)             # 0.004

# Ask with repo filter (multi-repo support)
answer = qa.ask("How does pagination work?", filter_repo="dal-ingest-server")
```

### Async Support

```python
# Async ingestion
stats = await qa.aingest("./docs/")

# Async question
answer = await qa.aask("How does auth work?")
```

---

## RAG Pipeline (LangGraph Agent)

```
                    ┌──────────┐
                    │ classify │  ← Determine query type + complexity
                    └────┬─────┘
                         │
                    ┌────▼─────┐
             ┌──────│ retrieve │  ← Semantic search in ChromaDB
             │      └────┬─────┘
             │           │
             │      ┌────▼─────┐
             │      │  reason  │  ← LLM evaluates: "Do I have enough context?"
             │      └────┬─────┘
             │           │
             │     ┌─────┴──────┐
             │     │            │
             │  insufficient   sufficient
             │  (refine query) │
             │     │           │
             │     ▼      ┌────▼─────┐
             └─────┘      │ respond  │  ← Generate final answer with citations
                          └──────────┘
```

This is a **LangGraph StateGraph** with conditional edges. The agent autonomously decides whether to refine its search or respond.

---

## Agent State

```python
class QAAgentState(TypedDict):
    query: str                          # original question
    repo_filter: str | None             # optional repo filter
    intent: str                         # classified intent
    retrieved_context: list[str]        # accumulated chunks
    current_query: str                  # may differ from original after refinement
    iteration: int                      # current loop count
    max_iterations: int                 # cap
    trace: list[dict]                   # step-by-step trace
    answer: str                         # final answer
    sources: list[str]                  # file paths
    confidence: float                   # self-assessed confidence
```

---

## Supported Document Formats

| Category | Extensions | Loader | Required Extra |
|----------|-----------|--------|----------------|
| **Code** | `.py`, `.js`, `.ts`, `.go`, `.java`, `.rs`, `.c`, `.cpp`, `.rb` | `TextLoader` (language-aware splitter) | — (core) |
| **Markup** | `.md`, `.txt`, `.rst` | `TextLoader` | — (core) |
| **PDF** | `.pdf` | `PyPDFLoader` | `aiforge[docs]` |
| **Word** | `.docx` | `Docx2txtLoader` | `aiforge[docs]` |
| **HTML** | `.html`, `.htm` | `UnstructuredHTMLLoader` | `aiforge[docs]` |
| **Data** | `.csv`, `.json`, `.yaml`, `.yml` | `CSVLoader` / `JSONLoader` | — (core) |

Unsupported file types are silently skipped during ingestion, with a warning log entry. Use `stats.skipped_files` on the `IngestResponse` to see which were not processed.

---

## Schemas

### Ingest

```python
class IngestRequest(BaseModel):
    source: str                         # path, URL, or git URL
    collection: str = "default"
    file_extensions: list[str] | None = None

class IngestResponse(BaseModel):
    chunks_indexed: int
    files_processed: int
    languages: dict[str, int]
    duration_seconds: float
    collection: str
```

### Query

```python
class QARequest(BaseModel):
    query: str
    collection: str = "default"
    filter_repo: str | None = None
    max_iterations: int = 3
    include_trace: bool = False

class QAResponse(BaseModel):
    text: str
    sources: list[str]
    confidence: float
    iterations: int
    trace: list[dict] | None = None
    input_tokens: int
    output_tokens: int
    cost: float
    trace_id: str

    def to_text(self) -> str:
        return self.text
```

---

## Batch Embedding

Documents are batched in chunks of 100 with rate limiting to avoid exceeding provider embed limits (FR-QA-010).

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-QA-001 | Ingest local directory | P0 | Recursively process files, chunk, embed, store |
| FR-QA-002 | Ingest git repo | P0 | Clone repo, process, clean up temp dir |
| FR-QA-003 | Language-aware chunking | P0 | Use language-specific splitters for code files |
| FR-QA-004 | Ask with RAG | P0 | Retrieve relevant chunks, generate grounded answer |
| FR-QA-005 | Agentic refinement | P0 | Self-assess sufficiency, refine query, retrieve again (max 3 loops) |
| FR-QA-006 | Source citations | P0 | Return file paths of source chunks |
| FR-QA-007 | Confidence score | P1 | Self-assessed answer confidence |
| FR-QA-008 | Trace output | P1 | Step-by-step agent reasoning trace |
| FR-QA-009 | Multi-repo support | P1 | Filter queries by repo/collection |
| FR-QA-010 | Batch embedding | P0 | Batch add_documents in chunks of 100 with rate limiting |

---

## File Structure

```
aiforge/modules/qa/
├── __init__.py
├── module.py          # DocQA class
├── agent.py           # LangGraph RAG agent
├── prompts.py         # Prompt templates
├── schemas.py         # Request/Response schemas
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic Q&A example
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /qa/ingest` and `POST /qa/ask` endpoints
