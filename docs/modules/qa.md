# Module: DocQA

`aiforge.modules.qa` — Document question answering with RAG and agentic refinement. Ingest files into a ChromaDB-backed vector store, then ask natural-language questions and get grounded answers with source citations. A LangGraph agent classifies queries, retrieves context, self-assesses sufficiency, and optionally refines the search before generating an answer.

---

## Installation

The QA module is included with the base `aiforge` install. ChromaDB is pulled in as a core dependency.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import DocQA
from aiforge import QAResponse, IngestResponse  # types
```

---

## Quick Start

```python
from aiforge import DocQA

qa = DocQA(collection="my-project")

# Ingest a codebase
stats = qa.ingest("./src/")
print(f"Indexed {stats.chunks_indexed} chunks from {stats.files_processed} files")

# Ask a question
answer = qa.ask("How does the auth middleware work?")
print(answer.text)              # "Authentication uses JWT tokens issued by..."
print(answer.sources)           # ["src/auth.py", "src/middleware.py"]
print(f"Confidence: {answer.confidence:.0%}")  # "Confidence: 92%"
```

---

## Constructor

```python
DocQA(
    collection: str = "default",
    persist_dir: str = "./aiforge_data",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    retrieval_k: int = 6,
    max_iterations: int = 3,
    store: VectorStoreAdapter | None = None,
    system_prompt: str | None = None,
    *,
    chunking: str | None = None,         # "recursive" | "markdown" | "token"
    retrieval_mode: str | None = None,   # "vector" | "hybrid" | "keyword"
    rerank: bool | None = None,
    rerank_top_n: int | None = None,
    fetch_k: int | None = None,
    reranker: Reranker | None = None,
    # Inherited from BaseModule:
    quality: str | None = None,
    provider: str | None = None,
    **kwargs,  # AIForgeSettings overrides
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str` | `"default"` | Vector store collection name. Must not be empty or whitespace-only. |
| `persist_dir` | `str` | `"./aiforge_data"` | Data directory for ChromaDB persistence. Ignored by non-ChromaDB backends. |
| `chunk_size` | `int \| None` | `None` | Override default chunk size (characters) for the text splitter. `None` uses the splitter's built-in default. |
| `chunk_overlap` | `int \| None` | `None` | Override default chunk overlap (characters). `None` uses the splitter's built-in default. |
| `retrieval_k` | `int` | `6` | Number of chunks to retrieve from the vector store per agent iteration. |
| `max_iterations` | `int` | `3` | Maximum agentic refinement loops before the agent is forced to respond. |
| `store` | `VectorStoreAdapter \| None` | `None` | Pre-built vector store adapter for dependency injection or testing. When `None`, a ChromaDB store is built automatically. |
| `system_prompt` | `str \| None` | `None` | Override the default system prompt for LLM calls. See [Prompt Resolution](#prompt-resolution). |
| `chunking` | `str \| None` | `None` → `AIFORGE_CHUNKING` (`"recursive"`) | Chunking strategy: `"recursive"`, `"markdown"` or `"token"`. See [Chunking Strategies](#chunking-strategies). |
| `retrieval_mode` | `str \| None` | `None` → `AIFORGE_RETRIEVAL_MODE` (`"vector"`) | `"vector"`, `"hybrid"` or `"keyword"`. See [Retrieval Modes](#retrieval-modes). |
| `rerank` | `bool \| None` | `None` → `AIFORGE_RERANK` (`False`) | Rerank retrieved candidates. Defaults to `True` when `reranker` is given. |
| `rerank_top_n` | `int \| None` | `None` → `AIFORGE_RERANK_TOP_N` | Chunks kept per retrieval after reranking. `None` keeps `retrieval_k`. |
| `fetch_k` | `int \| None` | `None` → `AIFORGE_FETCH_K` (`20`) | Candidates fetched per retriever before fusion / reranking. |
| `reranker` | `Reranker \| None` | `None` | Custom reranker. When reranking without one, an `LLMReranker` on this module's provider is used. See [Reranking](#reranking). |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`. |

### Validation Rules

The constructor raises `ValueError` for:
- Empty or whitespace-only `collection`
- `retrieval_k < 1`
- `max_iterations < 1`

### Examples

```python
from aiforge import DocQA

# Defaults — ChromaDB at ./aiforge_data, collection "default"
qa = DocQA()

# Named collection with custom chunking
qa = DocQA(
    collection="backend-api",
    chunk_size=2000,
    chunk_overlap=300,
)

# More retrieval results per iteration
qa = DocQA(collection="large-repo", retrieval_k=12)

# Allow more agent refinement loops
qa = DocQA(collection="research", max_iterations=5)

# Custom persistence directory
qa = DocQA(collection="prod", persist_dir="/data/vectordb")

# Specific model for answer generation
qa = DocQA(collection="docs", MODEL="gpt-4o")

# Quality-based routing
qa = DocQA(collection="qa", quality="high")
```

---

## Methods

### `ingest(source, *, file_extensions=None, batch_size=100) -> IngestResponse`

Ingest documents from a file path, directory, or list of file paths. Documents are loaded, chunked with language-aware splitting, embedded in batches, and stored in ChromaDB.

```python
def ingest(
    self,
    source: str | list[str],
    *,
    file_extensions: list[str] | None = None,
    batch_size: int = 100,
) -> IngestResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| list[str]` | *(required)* | File path, directory path, or list of file paths to ingest. |
| `file_extensions` | `list[str] \| None` | `None` | Extension whitelist (with leading dot, e.g. `[".py", ".md"]`). `None` accepts all supported extensions. |
| `batch_size` | `int` | `100` | Number of chunks per embedding API call. |

**Returns:** [`IngestResponse`](#ingestresponse)

**Behavior:**
1. `DocLoader` recursively discovers files, skipping ignored directories (`.git`, `node_modules`, `__pycache__`, etc.) and binary extensions (`.png`, `.zip`, `.exe`, etc.)
2. If a root `.gitignore` exists in the ingested directory, matching files/directories are skipped automatically.
3. `TextSplitter` chunks documents with language-aware splitting
4. Chunks are embedded in batches of `batch_size` via the configured embedding provider
5. Embeddings are stored in ChromaDB with deterministic UUIDs (based on source path + content prefix) for idempotent re-ingestion
6. File metadata (`source`, `file_type`, `chunk_index`) is stored alongside each embedding

```python
qa = DocQA(collection="my-project")

# Ingest a directory
stats = qa.ingest("./src/")
print(stats.chunks_indexed)    # 1247
print(stats.files_processed)   # 89
print(stats.languages)         # {"python": 45, "markdown": 30, "go": 14}
print(stats.skipped_files)     # ["./src/logo.png", "./src/data.bin"]
print(stats.collection)        # "my-project"
print(f"Cost: ${stats.cost:.4f}")  # embedding cost

# Ingest specific files
stats = qa.ingest(["README.md", "src/main.py", "src/config.py"])

# Ingest with extension filter
stats = qa.ingest("./repo/", file_extensions=[".py", ".md"])

# Ingest a single file
stats = qa.ingest("./docs/architecture.md")
```

---

### `aingest(source, *, file_extensions=None, batch_size=100) -> IngestResponse`

Async version of `ingest()`. File loading runs in an executor; embedding calls use the async embedding provider.

```python
async def aingest(
    self,
    source: str | list[str],
    *,
    file_extensions: list[str] | None = None,
    batch_size: int = 100,
) -> IngestResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| list[str]` | *(required)* | File path, directory, or list of file paths. |
| `file_extensions` | `list[str] \| None` | `None` | Extension whitelist. |
| `batch_size` | `int` | `100` | Chunks per embedding batch. |

**Returns:** [`IngestResponse`](#ingestresponse)

```python
stats = await qa.aingest("./large-codebase/", batch_size=200)
```

---

### `ingest_incremental(source, *, file_extensions=None, batch_size=100) -> IngestResponse`

Incrementally re-index only changed/new files and purge stale chunks for modified/deleted files.

```python
def ingest_incremental(
    self,
    source: str | list[str],
    *,
    file_extensions: list[str] | None = None,
    batch_size: int = 100,
) -> IngestResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| list[str]` | *(required)* | Directory, file, file list, or git URL. |
| `file_extensions` | `list[str] \| None` | `None` | Extension whitelist. |
| `batch_size` | `int` | `100` | Chunks per embedding batch. |

**Returns:** [`IngestResponse`](#ingestresponse), including incremental counters:
`files_added`, `files_updated`, `files_unchanged`, `files_deleted`, `chunks_deleted`.

Notes:
- For local files, `mtime` metadata is used to skip unchanged files.
- For sources without stable local mtimes (for example git URLs), AIForge uses conservative behavior and may re-index files in scope.
- Pinecone does not return exact delete counts, so `chunks_deleted` is best-effort on that backend.

```python
qa = DocQA(collection="my-project")

# Recommended first run for long-lived repos
first = qa.ingest_incremental("./src/")

# Later: only changed files are re-embedded; stale chunks are purged
delta = qa.ingest_incremental("./src/")
print(delta.files_added, delta.files_updated, delta.files_deleted)
```

### `aingest_incremental(source, *, file_extensions=None, batch_size=100) -> IngestResponse`

Async version of `ingest_incremental()`.

---

### `ask(query, *, filter_repo=None, include_trace=False, system_prompt=None) -> QAResponse`

Ask a question against ingested documents using the LangGraph RAG agent.

The agent classifies the query, retrieves relevant chunks via semantic search, self-assesses whether the context is sufficient, and may refine the search query and retrieve again (up to `max_iterations` loops) before generating a final answer with source citations.

```python
def ask(
    self,
    query: str,
    *,
    filter_repo: str | None = None,
    include_trace: bool = False,
    system_prompt: str | None = None,
) -> QAResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *(required)* | The question to answer. Must not be empty or whitespace-only. |
| `filter_repo` | `str \| None` | `None` | Source-path substring filter. Only chunks whose `source` metadata contains this string are retrieved. Useful for multi-repo collections. |
| `include_trace` | `bool` | `False` | Include the step-by-step agent trace in the response. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override for the LLM. |

**Returns:** [`QAResponse`](#qaresponse)

**Raises:**
- `ValueError` — if `query` is empty or whitespace-only
- `AIForgeError` — if the collection contains no documents

```python
qa = DocQA(collection="backend-api")

# Basic question
answer = qa.ask("How does the auth middleware work?")
print(answer.text)           # grounded answer with inline citations
print(answer.sources)        # ["src/auth/middleware.py", "src/auth/jwt.py"]
print(answer.confidence)     # 0.92
print(answer.iterations)     # 1 (answered on first retrieval)
print(f"Cost: ${answer.cost:.4f}")

# With agent trace for debugging
answer = qa.ask("What caching strategy is used?", include_trace=True)
for step in answer.trace:
    print(step)
# {"step": "classify", "intent": "factual"}
# {"step": "retrieve", "query": "What caching strategy is used?", "chunks_found": 6, "iteration": 1}
# {"step": "reason", "result": "INSUFFICIENT: caching layer implementation details", "decision": "insufficient"}
# {"step": "retrieve", "query": "caching layer implementation details", "chunks_found": 6, "iteration": 2}
# {"step": "reason", "result": "SUFFICIENT: 0.88", "decision": "sufficient"}
# {"step": "respond"}

# Filter by repo (multi-repo collection)
answer = qa.ask("How does pagination work?", filter_repo="dal-ingest-server")

# Per-call system prompt
answer = qa.ask("Explain the error handling", system_prompt="Answer in bullet points only.")
```

---

### `aask(query, *, filter_repo=None, include_trace=False, system_prompt=None) -> QAResponse`

Async version of `ask()`. Uses `graph.ainvoke()` for the LangGraph agent.

```python
async def aask(
    self,
    query: str,
    *,
    filter_repo: str | None = None,
    include_trace: bool = False,
    system_prompt: str | None = None,
) -> QAResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *(required)* | The question to answer. |
| `filter_repo` | `str \| None` | `None` | Source-path substring filter. |
| `include_trace` | `bool` | `False` | Include agent trace. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`QAResponse`](#qaresponse)

```python
answer = await qa.aask("How does the retry logic work?")
```

---

### `list_collections() -> list[str]`

Return names of all collections in the vector store.

```python
def list_collections(self) -> list[str]
```

**Returns:** `list[str]`

```python
collections = qa.list_collections()
print(collections)  # ["backend-api", "frontend", "docs"]
```

---

### `collection_stats(collection=None) -> dict[str, Any]`

Return statistics for a collection.

```python
def collection_stats(self, collection: str | None = None) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str \| None` | `None` | Collection name. Defaults to this instance's `collection`. |

**Returns:** `dict` with keys like `"chunks"`, `"files"`, `"size_mb"`.

```python
stats = qa.collection_stats()
print(stats)  # {"chunks": 1247, "files": 89, ...}

# Check a different collection
stats = qa.collection_stats("other-project")
```

---

### `delete_collection(collection=None) -> None`

Delete a collection and all its data.

```python
def delete_collection(self, collection: str | None = None) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | `str \| None` | `None` | Collection to delete. Defaults to this instance's `collection`. |

```python
qa.delete_collection()                  # delete current collection
qa.delete_collection("old-project")     # delete a specific collection
```

---

### `run(text, **kwargs) -> QAResponse`

BaseModule interface method. Delegates to `ask()`. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> QAResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The question to answer. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`QAResponse`](#qaresponse)

---

### `arun(text, **kwargs) -> QAResponse`

Async version of `run()`. Delegates to `aask()`.

```python
async def arun(self, text: str, **kwargs) -> QAResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Ask multiple questions concurrently. Inherited from `BaseModule`.

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
| `items` | `list[str]` | *(required)* | List of questions to answer. |
| `concurrency` | `int` | `10` | Max concurrent questions. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

```python
questions = [
    "How does authentication work?",
    "What database is used?",
    "How are errors handled?",
]
batch = qa.run_batch(questions, concurrency=3)
for r in batch.results:
    print(f"  [{r.confidence:.0%}] {r.text[:80]}...")
print(f"Total cost: ${batch.total_cost:.4f}")
```

---

## Response Types

### `IngestResponse`

Returned by `ingest()`, `aingest()`, `ingest_incremental()`, and `aingest_incremental()`.

```python
from aiforge import IngestResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `chunks_indexed` | `int` | Number of text chunks embedded and stored. |
| `files_processed` | `int` | Number of files successfully loaded. |
| `languages` | `dict[str, int]` | File count per detected language, e.g. `{"python": 45, "markdown": 30}`. |
| `skipped_files` | `list[str]` | Files that were skipped (unsupported type, binary, etc.). |
| `collection` | `str` | The collection name documents were ingested into. |
| `cost` | `float` | Embedding cost in USD. |
| `input_tokens` | `int` | Estimated tokens embedded (1 token ≈ 4 characters). |
| `output_tokens` | `int` | Always `0` for ingestion. |
| `model` | `str` | Embedding model name used. |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end ingestion time in milliseconds. |
| `files_added` | `int` | Incremental mode: new files not previously indexed. |
| `files_updated` | `int` | Incremental mode: files whose previous chunks were replaced. |
| `files_unchanged` | `int` | Incremental mode: files skipped because unchanged. |
| `files_deleted` | `int` | Incremental mode: indexed files no longer present in source scope. |
| `chunks_deleted` | `int` | Incremental mode: stale/orphan chunks removed before re-indexing. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_text()` | `str` | `"Ingested 89 files into 'my-project' (1247 chunks)"` |

```python
stats = qa.ingest("./src/")

print(stats.to_text())
# "Ingested 89 files into 'my-project' (1247 chunks)"

print(stats.chunks_indexed)    # 1247
print(stats.files_processed)   # 89
print(stats.languages)         # {"python": 45, "markdown": 30, "go": 14}
print(stats.skipped_files)     # ["./src/logo.png"]
print(stats.cost)              # 0.0012
print(stats.model)             # "text-embedding-3-small"
```

---

### `QAResponse`

Returned by `ask()`, `aask()`, `run()`, and `arun()`.

```python
from aiforge import QAResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The generated answer with inline source citations. |
| `sources` | `list[str]` | File paths of retrieved source chunks. |
| `citations` | `list[Citation]` | One entry per distinct retrieved chunk: `source`, `page`, `heading_path`, `chunk_index`, `score`, `rerank_score`. `Citation.label()` gives e.g. `"guide.pdf, page 3, Setup > Install"`. |
| `confidence` | `float` | Self-assessed answer confidence `[0.0, 1.0]`. |
| `iterations` | `int` | Number of retrieve-reason loops the agent performed. |
| `trace` | `list[dict] \| None` | Step-by-step agent trace (only if `include_trace=True`). |
| `cost` | `float` | Total LLM cost in USD (all agent steps combined). |
| `input_tokens` | `int` | Total input tokens across all LLM calls. |
| `output_tokens` | `int` | Total output tokens across all LLM calls. |
| `model` | `str` | LLM model used for reasoning and generation. |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_text()` | `str` | Returns the `text` field (the answer). Useful for pipeline chaining. |

```python
answer = qa.ask("How does the auth middleware work?", include_trace=True)

print(answer.text)           # "Authentication uses JWT tokens issued by..."
print(answer.sources)        # ["src/auth/middleware.py", "src/auth/jwt.py"]
print(answer.confidence)     # 0.92
print(answer.iterations)     # 2
print(answer.cost)           # 0.004
print(answer.model)          # "gpt-4o-mini"
print(answer.latency_ms)     # 2340.5

# Trace shows every agent step
for step in answer.trace:
    print(step["step"], step.get("decision", ""))

# Pipeline chaining
print(answer.to_text())      # same as answer.text
```

---

## RAG Pipeline (LangGraph Agent)

The question-answering pipeline is a **LangGraph StateGraph** with four nodes and a conditional loop:

```
                    ┌──────────┐
                    │ classify │  ← Determine query intent
                    └────┬─────┘
                         │
                    ┌────▼─────┐
             ┌──────│ retrieve │  ← Semantic search in ChromaDB
             │      └────┬─────┘
             │           │
             │      ┌────▼─────┐
             │      │  reason  │  ← LLM self-assesses: "Do I have enough context?"
             │      └────┬─────┘
             │           │
             │     ┌─────┴──────┐
             │     │            │
             │  INSUFFICIENT   SUFFICIENT
             │  (refine query) (with confidence)
             │     │           │
             │     ▼      ┌────▼─────┐
             └─────┘      │ respond  │  ← Generate answer with citations
                          └──────────┘
```

### Agent Nodes

#### 1. Classify

Sends the query to the LLM with `CLASSIFY_PROMPT` to determine the intent type. Returns one of:

| Intent | Description |
|--------|-------------|
| `factual` | Asking for specific facts from the codebase |
| `conceptual` | Asking how something works at a high level |
| `debugging` | Asking about a bug or error |
| `comparison` | Comparing two approaches or implementations |
| `exploratory` | Open-ended exploration of the codebase |

The intent is logged in the trace and passed to the reason node for context.

#### 2. Retrieve

Runs the configured [retrieval mode](#retrieval-modes) for the current query (original or refined), then the [reranker](#reranking) if enabled. Returns `retrieval_k` chunks per iteration (`rerank_top_n` when reranking). Each chunk keeps its `source`, `page` and `heading_path`, which label it in the prompt context (`[source: guide.md, Setup > Install]`) and become `QAResponse.citations`. Chunks accumulate across iterations (LangGraph `operator.add` reducer).

If `filter_repo` is set, chunks are filtered by `source` metadata using `{"$contains": filter_repo}`.

#### 3. Reason

Sends the most recent `retrieval_k` chunks along with the query to the LLM with `REASON_PROMPT`. The LLM responds with either:

- **`SUFFICIENT: <confidence>`** — The context is adequate. Extracts a confidence score (0.0–1.0) and routes to the respond node.
- **`INSUFFICIENT: <refined query>`** — The context is inadequate. Extracts a refined search query and loops back to the retrieve node.

Robust parsing handles markdown bullets, brackets, quoted text, and edge cases. If no confidence score can be extracted, defaults to `0.7`.

#### 4. Respond

Sends all accumulated chunks and the original query to the LLM with `RESPOND_PROMPT`. The LLM generates a grounded answer with inline source citations in `[filename:line_range]` format.

If the agent exhausted `max_iterations` without reaching `SUFFICIENT`, a fallback confidence of `0.3` is assigned.

### Agent State

```python
class QAAgentState(TypedDict):
    query: str                          # original question
    repo_filter: str | None             # optional source-path filter
    intent: str                         # classified intent (factual, conceptual, ...)
    retrieved_chunks: list[dict]        # accumulated chunks (operator.add reducer)
    current_query: str                  # may differ from original after refinement
    iteration: int                      # current loop count
    max_iterations: int                 # cap on refinement loops
    trace: list[dict]                   # step-by-step trace (operator.add reducer)
    answer: str                         # final answer text
    sources: list[str]                  # source file paths
    confidence: float                   # self-assessed confidence [0.0, 1.0]
    decision: str                       # "sufficient" or "insufficient"
    total_input_tokens: int             # accumulated (operator.add reducer)
    total_output_tokens: int            # accumulated (operator.add reducer)
    model: str                          # LLM model name
```

Fields annotated with `operator.add` use LangGraph's reducer semantics — values are **accumulated** across node invocations rather than replaced. This means `retrieved_chunks`, `trace`, `total_input_tokens`, and `total_output_tokens` grow across iterations.

### Routing Logic

```python
def should_continue(state) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "respond"      # forced response after max iterations
    if state["decision"] == "sufficient":
        return "respond"      # agent is confident
    return "retrieve"         # refine query and try again
```

---

## Supported Document Formats

Supported text-based file extensions loaded during ingestion:

| Category | Extensions |
|----------|-----------|
| **Code** | `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.java`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`, `.rb`, `.php`, `.swift`, `.kt`, `.scala` |
| **Shell** | `.sh`, `.bash`, `.zsh` |
| **Markup** | `.md`, `.txt`, `.rst` |
| **Config** | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg` |
| **Data** | `.csv`, `.xml` |
| **Web** | `.html`, `.htm` |
| **Other** | `.sql`, `.dockerfile` |

### Auto-Ignored Extensions

Binary and generated files are automatically ignored:
`.pyc`, `.pyo`, `.class`, `.o`, `.so`, `.dll`, `.exe`, `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.ico`, `.mp3`, `.mp4`, `.avi`, `.mov`, `.whl`, `.egg-info`, `.lock`

### Auto-Ignored Directories

`.git`, `.svn`, `.hg`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `node_modules`, `.next`, `.nuxt`, `venv`, `.venv`, `env`, `.env`, `dist`, `build`, `.build`, `aiforge_data`

### Extension Whitelist

Use `file_extensions` to restrict ingestion:

```python
# Only Python and Markdown files
stats = qa.ingest("./repo/", file_extensions=[".py", ".md"])
```

When `file_extensions` is specified, only files matching those extensions are loaded. All others go in `stats.skipped_files`.

---

## Ingestion Pipeline Details

### Chunking

Documents are split using `TextSplitter` with configurable `chunk_size` and `chunk_overlap` and the selected [chunking strategy](#chunking-strategies). The splitter is language-aware for code files.

### Batch Embedding

Chunks are embedded in batches of `batch_size` (default 100). Each batch makes one call to the embedding provider. Token counts are estimated at 1 token ≈ 4 characters.

### Deterministic IDs

Each chunk receives a deterministic UUID v5 based on `source path + first 64 characters of content`. This makes re-ingestion idempotent — re-ingesting the same files updates existing chunks rather than creating duplicates.

### Metadata

Each chunk stores metadata:
- `source` — file path
- `file_type` — file extension
- `chunk_index` — position within the batch
- `page` — page number (PDFs, 0-based), when known
- `heading_path` — markdown heading path such as `"Install > From source"` (`chunking="markdown"`)
- `mtime` — file modification time (incremental ingestion)

---

## Retrieval Modes

`retrieval_mode` (or `AIFORGE_RETRIEVAL_MODE`) selects how chunks are found:

| Mode | How it works | Query embedding |
|------|--------------|-----------------|
| `"vector"` (default) | Dense cosine similarity in the vector store. Unchanged behaviour. | Yes |
| `"keyword"` | Okapi BM25 over chunk texts (`k1=1.5`, `b=0.75`). Good for exact identifiers, error codes and rare terms. | No |
| `"hybrid"` | Both, each fetching `fetch_k` candidates, fused with Reciprocal Rank Fusion: `score(d) = Σ 1 / (RRF_K + rank(d))`. | Yes |

```python
qa = DocQA(collection="docs", retrieval_mode="hybrid", fetch_k=30)
qa.ingest("./docs")
answer = qa.ask("What does error E4012 mean?")

# Inspect retrieval without running the agent
for hit in qa.retrieve("E4012", top_k=3):
    print(f"{hit['score']:.2f}", hit["metadata"]["source"])
```

**The keyword index.** The BM25 index is pure Python (no extra dependency)
and lives in memory. It is not persisted separately: it is built from the
vector store's own documents (`VectorStoreAdapter.list_all_documents()`) on
the first keyword or hybrid query, and rebuilt after this instance writes to
the store or whenever the store's chunk count changes (for example after
another process ingests). The vector store stays the single source of truth,
so the two indexes cannot drift apart across restarts. The tokenizer
lower-cases text and splits on non-alphanumeric characters (so
`auth_middleware` matches `auth` and `middleware`); there is no stemming.

**Backends.** Chroma (default), pgvector, Qdrant and Pinecone implement
`list_all_documents()`. A custom backend that does not falls back to vector
retrieval with a `retrieval.keyword_unsupported` warning. Metadata filters
(`filter_repo`, or `filter=` on `retrieve()`) are applied to keyword hits
locally; they support field equality, `$eq $ne $gt $gte $lt $lte $in $nin
$contains`, `$and` and `$or`. A filter using any other operator runs that
query in vector mode with a `retrieval.filter_unsupported` warning.

**Scores.** `hit["score"]` is always in `[0, 1]`: cosine similarity in vector
mode, BM25 relative to the best hit in keyword mode, and the RRF score
relative to the best possible (ranked first by both) in hybrid mode.

---

## Reranking

With `rerank=True` (or `AIFORGE_RERANK=true`) each retrieval fetches
`fetch_k` candidates, reranks them and keeps the best `rerank_top_n`
(default `retrieval_k`):

```python
qa = DocQA(collection="docs", retrieval_mode="hybrid", rerank=True, fetch_k=25, rerank_top_n=6)
```

The default `LLMReranker` scores every candidate in **one** LLM call using the
provider's native structured output (`response_format` with
`{"scores": [{"index": i, "score": 0.0-1.0}, ...]}`); providers without
native structured output are asked for the same JSON in text. If the call
fails or returns unusable output, the original retrieval order is kept and a
`rerank.llm_failed` warning is logged. Its tokens are included in
`QAResponse.input_tokens` / `output_tokens` and cost tracking.

**Custom rerankers.** Anything with a
`rerank(query, candidates, top_n) -> list[dict]` method satisfies the
`aiforge.core.rerank.Reranker` protocol. `ScoreFunctionReranker` wraps a
`(query, texts) -> scores` function, which makes a cross-encoder a one-liner:

```python
from sentence_transformers import CrossEncoder
from aiforge.core.rerank import ScoreFunctionReranker

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qa = DocQA(
    collection="docs",
    reranker=ScoreFunctionReranker(lambda q, docs: ce.predict([(q, d) for d in docs]).tolist()),
)
```

Passing `reranker=` enables reranking. A custom reranker that raises is
logged and the original order is kept. Reranked chunks carry
`rerank_score`, which is copied to `Citation.rerank_score`.

---

## Chunking Strategies

| `chunking` | Sizing | Behaviour |
|------------|--------|-----------|
| `"recursive"` (default) | characters: 1000/200 for code, 1200/200 for prose | Language-aware separators for code; paragraph → line → sentence for prose. Unchanged. |
| `"markdown"` | characters, as `recursive` | `.md` / `.markdown` files are split on headings first, so no chunk spans two sections, then oversized sections are split recursively. Each chunk gets `heading_path` metadata (`"Guide > Install > Linux"`). Headings inside code fences are ignored. Other files use `recursive`. |
| `"token"` | tiktoken tokens (`cl100k_base`): 400/60 | Same separators as `recursive`, but sized in tokens so chunks fit embedding-model limits regardless of language. |

Overlap defaults: when you set `chunk_size` but not `chunk_overlap`, the
overlap is the smaller of the strategy default and `chunk_size // 4`, so
small chunk sizes stay valid. An explicit `chunk_overlap=0` now means no
overlap.

`TextSplitter` can also be used directly, including with a custom token
counter (for example a Hugging Face tokenizer):

```python
from aiforge.core.text_splitter import TextSplitter

splitter = TextSplitter(chunk_size=256, strategy="token",
                        token_counter=lambda s: len(hf_tokenizer.encode(s)))
chunks = splitter.split(documents)
```

tiktoken downloads each encoding once and caches it; on offline machines,
pre-populate `TIKTOKEN_CACHE_DIR` or pass `token_counter=`.

---

## Prompt Resolution

System prompts for LLM calls (classify, reason, respond) are resolved in this precedence order:

1. **Per-call override:** `qa.ask(query, system_prompt="...")`
2. **Constructor override:** `DocQA(system_prompt="...")`
3. **Config file:** `CUSTOM_PROMPTS.qa.system` in `aiforge.yaml`
4. **Built-in default:** empty string (no system prompt — agent prompts supply all context)

The agent's node-specific prompts (`CLASSIFY_PROMPT`, `REASON_PROMPT`, `RESPOND_PROMPT`) are always applied and are not overridden by the system prompt — the system prompt is prepended as a system message.

---

## Instance Properties

After constructing a `DocQA`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `qa.collection` | `str` | Collection name. |
| `qa.persist_dir` | `str` | ChromaDB data directory. |
| `qa.chunk_size` | `int \| None` | Configured chunk size. |
| `qa.chunk_overlap` | `int \| None` | Configured chunk overlap. |
| `qa.retrieval_k` | `int` | Chunks retrieved per iteration. |
| `qa.max_iterations` | `int` | Max agentic refinement loops. |
| `qa.settings` | `AIForgeSettings` | Resolved configuration. |
| `qa.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Ingest and Query a Codebase

```python
from aiforge import DocQA

qa = DocQA(collection="backend-api", chunk_size=2000)

# Ingest entire codebase
stats = qa.ingest("./src/", file_extensions=[".py", ".go", ".md"])
print(f"Indexed {stats.chunks_indexed} chunks from {stats.files_processed} files")
print(f"Languages: {stats.languages}")
print(f"Embedding cost: ${stats.cost:.4f}")

# Ask questions
answer = qa.ask("How does the rate limiting work?")
print(answer.text)
print(f"Sources: {', '.join(answer.sources)}")
print(f"Confidence: {answer.confidence:.0%}")
print(f"Iterations: {answer.iterations}")
```

### Multi-Repo Collection

Ingest multiple repositories into the same collection and filter by repo at query time:

```python
qa = DocQA(collection="all-repos")

# Ingest multiple repos
qa.ingest("./backend/")
qa.ingest("./frontend/")
qa.ingest("./shared-lib/")

# Query scoped to a specific repo
answer = qa.ask(
    "How does the API authentication work?",
    filter_repo="backend",
)

# Query across all repos
answer = qa.ask("Where is the User model defined?")
```

### Agent Trace for Debugging

```python
qa = DocQA(collection="my-project", max_iterations=4)

answer = qa.ask(
    "What caching strategy does the application use?",
    include_trace=True,
)

print(f"Answer: {answer.text[:100]}...")
print(f"Confidence: {answer.confidence:.0%}")
print(f"Iterations: {answer.iterations}")
print(f"Cost: ${answer.cost:.4f}")
print()

for step in answer.trace:
    if step["step"] == "classify":
        print(f"  classify → intent: {step['intent']}")
    elif step["step"] == "retrieve":
        print(f"  retrieve → query: '{step['query']}', found: {step['chunks_found']} chunks (iter {step['iteration']})")
    elif step["step"] == "reason":
        print(f"  reason → {step['decision']}: {step['result'][:60]}...")
    elif step["step"] == "respond":
        print(f"  respond → answer generated")
```

### Incremental Ingestion

Use incremental mode to avoid stale chunks when files are modified or deleted:

```python
qa = DocQA(collection="my-project")

# First run (recommended): builds mtime metadata
stats = qa.ingest_incremental("./src/")
print(f"Indexed: {stats.chunks_indexed}")

# After code changes
stats = qa.ingest_incremental("./src/")
print(f"Added={stats.files_added}, Updated={stats.files_updated}, Deleted={stats.files_deleted}")
print(f"Purged stale chunks={stats.chunks_deleted}, New chunks={stats.chunks_indexed}")
```

### Collection Management

```python
qa = DocQA(collection="temp-analysis")

# List all collections
print(qa.list_collections())  # ["backend-api", "frontend", "temp-analysis"]

# Check stats
stats = qa.collection_stats()
print(stats)  # {"chunks": 500, ...}

# Clean up
qa.delete_collection()

# Delete a specific collection
qa.delete_collection("old-project")
```

### Async Usage

```python
import asyncio
from aiforge import DocQA

async def main():
    qa = DocQA(collection="async-demo")

    # Async ingest
    stats = await qa.aingest("./src/")
    print(f"Indexed {stats.chunks_indexed} chunks")

    # Async ask
    answer = await qa.aask("How does error handling work?")
    print(answer.text)
    print(f"Confidence: {answer.confidence:.0%}")

asyncio.run(main())
```

### Pipeline Integration

Use `DocQA` as part of a pipeline via the `run()` / `arun()` interface:

```python
from aiforge import DocQA

qa = DocQA(collection="my-project")

# Pipeline-compatible interface
result = qa.run("How does the auth middleware work?")
print(result.to_text())  # same as result.text
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
MODEL: "gpt-4o-mini"
EMBEDDING_MODEL: "text-embedding-3-small"
```

```python
qa = DocQA(collection="docs")  # picks up aiforge.yaml automatically
stats = qa.ingest("./docs/")
print(stats.model)  # "text-embedding-3-small"

answer = qa.ask("What does the config module do?")
print(answer.model)  # "gpt-4o-mini"
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Empty `collection`, `query`, `retrieval_k < 1`, or `max_iterations < 1` | `DocQA(collection="")` |
| `AIForgeError` | Collection has no documents when `ask()` is called | `qa.ask("?")` on empty collection |
| `ProviderError` | LLM or embedding provider returns an error | Invalid API key |
| `RateLimitError` | Provider rate limit exceeded | Batch ingestion too fast |
| `APITimeoutError` | Provider request timed out | Large embedding batch |
| Embedding failures | Logged and re-raised | Network issues during ingestion |

```python
from aiforge.core.types import AIForgeError, ProviderError

qa = DocQA(collection="my-project")

# Empty collection guard
try:
    answer = qa.ask("How does auth work?")
except AIForgeError as e:
    print(f"No documents: {e}")
    # "No documents ingested in collection 'my-project'"

# Empty query guard
try:
    answer = qa.ask("")
except ValueError as e:
    print(f"Invalid query: {e}")

# Provider errors
try:
    stats = qa.ingest("./src/")
except ProviderError as e:
    print(f"Embedding failed: {e}")
```

---

## File Structure

```
aiforge/modules/qa/
├── __init__.py        # Exports: DocQA, QAResponse, IngestResponse
├── module.py          # DocQA class (ingest, ask, collection management)
├── agent.py           # LangGraph StateGraph (classify → retrieve → reason → respond)
├── prompts.py         # Prompt templates (CLASSIFY_PROMPT, REASON_PROMPT, RESPOND_PROMPT)
├── schemas.py         # Pydantic models (QAResponse, IngestResponse)
└── widget.py          # Streamlit playground widget (Ask / Ingest / Collections modes)
```

---

## DocQA vs SemanticSearch

Both modules use vector stores and embeddings, but they serve different
purposes:

| | DocQA | SemanticSearch |
|---|---|---|
| **Purpose** | Answer a question in natural language | Find the most similar documents |
| **LLM calls** | Yes — agentic LangGraph pipeline with self-assessment and iterative refinement | None — embedding-only |
| **Ingestion** | Raw files, directories, or git URLs (auto-loads, language-aware chunking, embeds) | Pre-structured documents with explicit `id`, `text`, `metadata` |
| **Output** | Synthesized answer (`text`), `sources`, `confidence`, optional `trace` | Ranked `hits` with similarity `score`, `id`, `text` |
| **Cost** | Embedding tokens (ingest) + LLM tokens (ask) | Embedding tokens only |
| **Use when** | You need a *written answer* derived from documents | You need a *ranked list* of matching passages |

Use **DocQA** when you want an answer; use **SemanticSearch** when you want
to find documents and handle the results yourself.

---

## See Also

- [Quick Start](../quickstart.md) — Basic Q&A example
- [SemanticSearch](search.md) — Embedding-only retrieval (no LLM)
- [Pipelines](../pipelines.md) — Chain DocQA with other modules
- [API Server](../api-server.md) — `POST /qa/ingest` and `POST /qa/ask` endpoints
- [API Server > Webhooks](../api-server.md#async-jobs--webhooks) — Async ingestion with webhook notifications
- [Configuration](../configuration.md) — Full settings reference
- [Playground](../playground.md) — Interactive QA tab
