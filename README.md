# AIForge — Plug-and-Play AI Toolkit for Product Teams

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.2-orange.svg)](CHANGELOG.md)

**AIForge** is a modular, provider-agnostic Python SDK that gives product teams production-ready AI capabilities with a single `pip install` and a config file. Each module — summarization, document Q&A, extraction, classification, semantic search, content generation, conversational agents, content moderation, translation, and vision — is self-contained, composable, and ships with a built-in Streamlit UI widget for instant prototyping.

> **Enable any developer to add production-quality AI capabilities to their product in minutes, not weeks.**

---

## Key Features

| Feature | Description |
|---------|-------------|
| **10 AI Modules** | Summarize, DocQA (RAG), Extract, Classify, Search, Generate, Chat, Moderate, Translate, Vision |
| **Provider-Agnostic** | Swap OpenAI, Anthropic, Gemini, Bedrock, Mistral, Azure or Ollama (local) by changing 1 config value |
| **Production-Ready** | Retry, rate limiting, streaming, cost tracking out of the box |
| **Evals** | Score module outputs against test sets and gate CI on regressions (`aiforge eval run`) |
| **Guardrails** | PII and secret redaction, prompt-injection checks, output validation around any module |
| **Hybrid RAG** | Keyword + vector retrieval, reranking and heading-aware chunking for DocQA and Search |
| **Composable Pipelines** | Chain modules into workflows (e.g., extract → classify → summarize) |
| **Plugin System** | Extend with custom modules, providers, and vector stores via a unified registry |
| **Built-in Observability** | OpenTelemetry + LangSmith tracing, structured logging, cost tracking |
| **UI Playground** | Streamlit-powered interactive demo per module — launch with 1 command |
| **REST API** | Optional FastAPI server exposing all modules as HTTP endpoints |

---

## Technology Stack

| Technology | Role |
|-----------|------|
| **LangChain** | Core LLM abstraction for all 10 modules |
| **LangChain provider packages** | OpenAI, Anthropic, Google GenAI, AWS, Mistral and Ollama integrations (install as extras) |
| **LangGraph** | Stateful multi-turn chat agent & RAG agentic loops |
| **ChromaDB** | Default vector database for semantic search & RAG |
| **Pinecone / Qdrant / pgvector** | Production-scale vector store alternatives (install as extras) |
| **FastAPI** | REST API server |
| **Streamlit** | Interactive UI playground |
| **Pydantic v2** | Configuration, schemas, input/output validation |
| **OpenTelemetry** | Distributed tracing |

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- An API key for at least one LLM provider (OpenAI, Anthropic, Gemini, Mistral, AWS credentials for Bedrock, or Ollama for local)

### 1. Clone & Set Up Environment

```bash
git clone <repo-url>
cd aiforge

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example config files
cp .env.example .env
cp aiforge.yaml.example aiforge.yaml

# Edit .env and set your API key
# AIFORGE_OPENAI_API_KEY=sk-...
```

### 3. Use in Code

```python
from aiforge import Summarizer

s = Summarizer()
result = s.run("AIForge is a plug-and-play AI toolkit that gives teams...")
print(result.summary)
print(f"Cost: ${result.cost:.4f}")
```

### 4. Launch the Playground UI

```bash
aiforge playground
# Opens Streamlit app at http://localhost:8501
```

### 5. Start the API Server

```bash
aiforge serve
# REST API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## Installation Options

```bash
# From PyPI (the distribution is aiforge-kit; the import name is aiforge)
pip install aiforge-kit                          # Core + OpenAI provider + ChromaDB
pip install "aiforge-kit[anthropic,mcp]"         # Pick extras as needed

# From a clone
pip install -r requirements.txt
pip install -e ".[all]"                          # Editable install for development

# Extras (combine with commas, e.g. "aiforge-kit[gemini,server]")
pip install "aiforge-kit[anthropic]"      # Anthropic + Voyage AI embeddings
pip install "aiforge-kit[gemini]"         # Google Gemini
pip install "aiforge-kit[bedrock]"        # Amazon Bedrock
pip install "aiforge-kit[mistral]"        # Mistral AI
pip install "aiforge-kit[local]"          # Ollama (local models)
pip install "aiforge-kit[pinecone]"       # Pinecone vector store
pip install "aiforge-kit[qdrant]"         # Qdrant vector store
pip install "aiforge-kit[pgvector]"       # PostgreSQL + pgvector
pip install "aiforge-kit[server]"         # FastAPI REST server
pip install "aiforge-kit[playground]"     # Streamlit playground
pip install "aiforge-kit[agentic]"        # LangGraph agentic RAG / chat agent
pip install "aiforge-kit[observability]"  # OpenTelemetry + LangSmith
pip install "aiforge-kit[docs]"           # PDF and DOCX document loaders
pip install "aiforge-kit[mcp]"            # MCP client (stdio + Streamable HTTP)
pip install "aiforge-kit[all]"            # Everything
```

For development, `uv sync --all-extras` installs the exact versions CI tests
against from `uv.lock`. CI also runs the suite on the lowest versions
`pyproject.toml` allows, and Dependabot proposes lockfile updates weekly.

---

## Lint and Type checking

```bash
# Lint checker
ruff check .
# Output the lint findings to a file
ruff check . > ruff_output.txt

# Type checker
pyright .
```

---

## Modules

### Summarize — Text Summarization

```python
from aiforge import Summarizer

s = Summarizer(style="concise", max_length=200)
result = s.run("Long article text here...")
print(result.summary)

# Long documents auto-handled via map-reduce
result = s.run_file("report.pdf")
```

### DocQA — Document Q&A (RAG)

```python
from aiforge import DocQA

qa = DocQA(collection="my_docs")
qa.ingest("./docs/")                              # ingest a directory
answer = qa.ask("How does authentication work?")
print(answer.text)
print(answer.sources)                              # source citations
```

Hybrid retrieval, reranking and heading-aware chunking are opt-in:

```python
qa = DocQA(
    collection="my_docs",
    chunking="markdown",        # split on headings ("recursive" | "markdown" | "token")
    retrieval_mode="hybrid",    # BM25 keyword + vector, fused ("vector" | "hybrid" | "keyword")
    rerank=True,                # LLM reranker over the fused candidates
    rerank_top_n=4,
)
```

`SemanticSearch` takes the same `retrieval_mode`, `rerank` and `chunking` options.

### Extract — Structured Data Extraction

```python
from aiforge import Extractor
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

e = Extractor(output_schema=Invoice)
result = e.run("Invoice from Acme Corp. Total: $86.00 USD.")
print(result.data)  # Invoice(vendor="Acme Corp", total=86.0, currency="USD")
```

### Classify — Text Classification

```python
from aiforge import Classifier

c = Classifier(labels=["bug", "feature", "question"])
result = c.run("The login button is broken on mobile")
print(result.label)       # "bug"
print(result.confidence)  # 0.94
```

### Search — Semantic Search

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="products")
ss.index([
    {"id": "1", "text": "Wireless headphones", "metadata": {"price": 299}},
    {"id": "2", "text": "Bluetooth speaker", "metadata": {"price": 79}},
])
results = ss.query("something to listen to music", top_k=5)
```

### Generate — Template-Based Content Generation

```python
from aiforge import Generator

g = Generator(
    template="Write a {tone} product description for: {product_name}",
    guardrails=["no_pii", "professional_tone"],
)
result = g.run(product_name="AIForge", tone="professional")
```

### Chat — Conversational Agent

```python
from aiforge import ChatAgent

agent = ChatAgent(
    system_prompt="You are a helpful assistant.",
    memory="buffer",
)
response = agent.chat("What's the return policy?")
response = agent.chat("What about international orders?")  # remembers context
```

### Moderate — Content Safety & PII Detection

```python
from aiforge import Moderator

m = Moderator(checks=["pii", "toxicity", "prompt_injection"])
result = m.check("My email is john@example.com")
print(result.flagged)  # True

cleaned = m.redact("Call me at 555-0123")
print(cleaned.text)    # "Call me at [PHONE]"
```

### Guardrails — Input & Output Checks

```python
from aiforge import Summarizer
from aiforge.guardrails import Guard, PIIGuard, PromptInjectionGuard

guard = Guard(
    input=[PIIGuard(), PromptInjectionGuard()],
    output=[PIIGuard()],
    on_fail="redact",            # "block" (default) | "redact" | "warn"
)
safe_summarize = guard.wrap(Summarizer())
result = safe_summarize.run("Email jane@example.com about the outage...")
```

See [Guardrails](docs/guardrails.md) for all checks and redaction styles.

### Evals — Regression Testing for Prompts and Models

```python
from aiforge import Classifier
from aiforge.evals import evaluate, label_match

report = evaluate(
    Classifier(labels=["bug", "feature", "question"]),
    "evals/tickets.jsonl",       # or a list of {"input": ..., "expected": ...}
    scorers=[label_match()],
    min_pass_rate=0.9,
)
report.print()
assert report.passed
```

Or from CI: `aiforge eval run evals/tickets.jsonl -m classify -s label_match --min-pass-rate 0.9`.
See [Evaluations](docs/evals.md).

---

## Pipeline Composition

Chain modules into workflows:

```python
from aiforge import Pipeline, Extractor, Classifier, Summarizer

pipe = Pipeline([
    ("extract", Extractor(output_schema=TicketSchema)),
    ("classify", Classifier(labels=["critical", "high", "medium", "low"])),
    ("summarize", Summarizer(style="concise")),
])

result = pipe.run("Long customer complaint email...")
print(result["classify"].label)       # "critical"
print(result["summarize"].summary)    # One-line summary
```

Conditional routing:

```python
pipe = Pipeline([
    ("classify", Classifier(labels=["support", "sales", "spam"])),
    ("route", {
        "support": Summarizer(style="bullet_points"),
        "sales": Generator(template="..."),
        "spam": None,  # drop
    }),
])
```

---

## Configuration

AIForge uses a layered configuration system:

```
1. Defaults (sensible built-in values)
        ↓ overridden by
2. aiforge.yaml (project-level config)
        ↓ overridden by
3. .env file / environment variables (AIFORGE_*)
        ↓ overridden by
4. Constructor arguments (code-level)
```

### Key Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_PROVIDER` | No | `openai` | LLM provider (`openai`, `anthropic`, `gemini`, `bedrock`, `mistral`, `ollama`, `azure`) |
| `AIFORGE_MODEL` | No | `gpt-4o` | LLM model name |
| `AIFORGE_TEMPERATURE` | No | `0` | LLM temperature |
| `AIFORGE_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `AIFORGE_OPENAI_API_KEY` | Yes (if OpenAI) | — | OpenAI API key |
| `AIFORGE_ANTHROPIC_API_KEY` | Yes (if Anthropic) | — | Anthropic API key |
| `AIFORGE_VOYAGE_API_KEY` | Yes (if Anthropic) | — | Voyage AI key for Anthropic embeddings |
| `AIFORGE_GOOGLE_API_KEY` | Yes (if Gemini) | — | Gemini Developer API key |
| `AIFORGE_AWS_REGION` | No | AWS default | Bedrock region (credentials come from the AWS credential chain) |
| `AIFORGE_AWS_PROFILE` | No | — | Named AWS profile for Bedrock |
| `AIFORGE_MISTRAL_API_KEY` | Yes (if Mistral) | — | Mistral AI API key |
| `AIFORGE_OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `AIFORGE_VECTOR_STORE` | No | `chroma` | Vector backend (`chroma`, `pgvector`, `pinecone`, `qdrant`) |
| `AIFORGE_CHROMA_PERSIST_DIR` | No | `./aiforge_data` | ChromaDB data directory |
| `AIFORGE_PGVECTOR_DSN` | Yes (if pgvector) | — | PostgreSQL connection string |
| `AIFORGE_PINECONE_API_KEY` | Yes (if Pinecone) | — | Pinecone API key |
| `AIFORGE_PINECONE_INDEX` | Yes (if Pinecone) | — | Pinecone index name |
| `AIFORGE_QDRANT_URL` | Yes (if Qdrant) | — | Qdrant server URL |
| `AIFORGE_RETRIEVAL_MODE` | No | `vector` | RAG retrieval (`vector`, `hybrid`, `keyword`) |
| `AIFORGE_RERANK` | No | `false` | Rerank retrieved chunks before answering |
| `AIFORGE_CHUNKING` | No | `recursive` | Chunking strategy (`recursive`, `markdown`, `token`) |
| `AIFORGE_GUARDRAILS_ON_FAIL` | No | `block` | Default guardrail action (`block`, `redact`, `warn`) |
| `AIFORGE_GUARDRAILS_PII_REDACTION` | No | `placeholder` | PII redaction style (`placeholder`, `mask`, `hash`) |
| `AIFORGE_TRACING_ENABLED` | No | `false` | Enable LangSmith/OTel tracing |
| `AIFORGE_COST_TRACKING_ENABLED` | No | `true` | Enable cost tracking |
| `AIFORGE_API_KEY` | No | — | Bearer token auth for REST API |

See [.env.example](.env.example) for the full list.

### Swapping Providers

```yaml
# aiforge.yaml — change these 2 lines
provider: anthropic
model: claude-sonnet-5-5
```

Or at runtime:

```python
s = Summarizer(provider="anthropic", model="claude-sonnet-5-5")
```

---

## REST API

Start with `aiforge serve` — all modules are exposed as HTTP endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/summarize` | Summarize text |
| POST | `/qa/ingest` | Ingest documents |
| POST | `/qa/ask` | Ask a question |
| POST | `/extract` | Extract structured data |
| POST | `/classify` | Classify text |
| POST | `/search/index` | Index documents |
| POST | `/search/query` | Semantic search |
| POST | `/generate` | Generate content |
| POST | `/chat` | Chat message |
| POST | `/moderate` | Content safety check |
| POST | `/moderate/redact` | PII redaction |
| POST | `/pipeline` | Run a pipeline |
| GET | `/health` | Health & status |
| GET | `/docs` | Swagger UI |

Example:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "The login button is broken", "labels": ["bug", "feature", "question"]}'
```

---

## Playground UI

Interactive Streamlit app with tabs for each module:

```bash
aiforge playground
# http://localhost:8501
```

Tabs: Summarize | Q&A | Extract | Classify | Search | Generate | Chat | Moderate | Translate | Vision | Pipeline | Settings

---

## Docker

```bash
# Build and run
docker compose up --build

# Services:
#   api:        http://localhost:8000  (REST API)
#   playground:  http://localhost:8501  (Streamlit UI)
```

---

## CLI Reference

```bash
aiforge init          # Scaffold aiforge.yaml + .env.example
aiforge serve         # Start FastAPI server (default: 0.0.0.0:8000)
aiforge playground    # Start Streamlit playground (default: port 8501)
aiforge migrate       # Migrate vector store collections

aiforge summarize --text "Long text..."      # Every module has a command
aiforge qa ingest ./docs && aiforge qa ask "How does auth work?"
aiforge eval run evals/cases.jsonl -m classify -s label_match --min-pass-rate 0.9
aiforge eval compare report.json main-report.json  # exit 3 on regression
aiforge models list                          # Models, pricing, cost estimates
```

See the [CLI reference](docs/cli.md) for every command and flag.

---

## Project Structure

```
aiforge/
├── pyproject.toml                     # Package config, dependencies, extras
├── requirements.txt                   # Pinned dependencies for venv development
├── Makefile                           # Dev commands
├── Dockerfile                         # Multi-stage production build
├── docker-compose.yml                 # API + Playground services
├── aiforge.yaml.example              # Example configuration
├── .env.example                       # Example environment variables
│
├── aiforge/                           # SDK source code
│   ├── __init__.py                    # Public API exports
│   ├── __version__.py                 # Version string
│   ├── registry.py                    # Plugin registry (modules, providers, vector stores)
│   ├── core/                          # Provider-agnostic abstractions
│   │   ├── types.py                   # BaseModule, CompletionResult, error hierarchy
│   │   ├── provider.py               # LLMProvider, EmbeddingProvider ABCs
│   │   ├── vectorstore.py            # VectorStoreAdapter + ChromaDB default
│   │   ├── vectorstore_pinecone.py    # PineconeVectorStore adapter
│   │   ├── vectorstore_qdrant.py      # QdrantVectorStore adapter
│   │   ├── vectorstore_pgvector.py    # PgVectorStore adapter (PostgreSQL + pgvector)
│   │   ├── vectorstore_factory.py     # Config-driven vector store factory
│   │   ├── doc_loader.py             # Document loading (files, directories, git)
│   │   ├── text_splitter.py          # Recursive, markdown and token chunking
│   │   ├── retrieval.py              # BM25 index + hybrid retrieval (RRF)
│   │   ├── rerank.py                 # LLM and score-function rerankers
│   │   └── pricing.py                # Token pricing tables
│   ├── infra/                         # Cross-cutting infrastructure
│   │   ├── config.py                  # AIForgeSettings (Pydantic Settings)
│   │   ├── retry.py                   # Tenacity retry wrappers
│   │   ├── rate_limiter.py            # Token-bucket rate limiter
│   │   ├── cost_tracker.py            # Session-level cost aggregation
│   │   ├── tracing.py                 # LangSmith / OpenTelemetry setup
│   │   ├── logging.py                 # Structured logging (structlog)
│   │   └── streaming.py              # Async streaming helpers
│   ├── providers/                     # Concrete provider implementations
│   │   ├── _langchain.py              # Shared LangChain provider base classes
│   │   ├── openai.py                  # OpenAI (GPT-4o, text-embedding-3-small)
│   │   ├── anthropic.py              # Anthropic (Claude) + Voyage embeddings
│   │   ├── gemini.py                  # Google Gemini
│   │   ├── bedrock.py                 # Amazon Bedrock (Converse API)
│   │   ├── mistral.py                 # Mistral AI
│   │   ├── ollama.py                  # Ollama (Llama 3, Mistral — local)
│   │   └── azure.py                   # Azure OpenAI
│   ├── modules/                       # 10 AI modules
│   │   ├── summarize/                 # Text summarization
│   │   ├── qa/                        # Document Q&A (RAG)
│   │   ├── extract/                   # Structured data extraction
│   │   ├── classify/                  # Text classification
│   │   ├── search/                    # Semantic search
│   │   ├── generate/                  # Content generation
│   │   ├── chat/                      # Conversational agent (LangGraph)
│   │   ├── moderate/                  # Content moderation & PII detection
│   │   ├── translate/                 # Text translation
│   │   └── vision/                    # Vision / image understanding
│   ├── evals/                         # Datasets, scorers, eval runner, reports
│   ├── guardrails/                    # Guard, input/output checks, PII redaction
│   ├── mcp/                           # MCP client + tool bridge
│   ├── pipeline/                      # Pipeline composition engine
│   ├── server/                        # FastAPI REST API
│   ├── playground/                    # Streamlit UI
│   └── cli/                           # CLI (Typer)
│
├── tests/                             # Test suite
│   ├── conftest.py                    # Shared fixtures
│   ├── unit/                          # Unit tests per module
│   ├── integration/                   # Integration tests (real API calls)
│   └── fixtures/                      # Sample texts, documents, recorded responses
│
└── docs/                              # Documentation
    ├── quickstart.md
    ├── configuration.md
    ├── providers.md
    ├── pipelines.md
    ├── guardrails.md
    ├── evals.md
    ├── cost-tracking.md
    ├── cli.md
    └── modules/                       # Per-module docs
```

---

## Development

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd aiforge

# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies (including dev + all extras)
pip install -e ".[all]"

# Copy config templates
cp .env.example .env
cp aiforge.yaml.example aiforge.yaml
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aiforge --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_summarize.py

# Run specific test
pytest tests/unit/test_config.py::test_yaml_loading
```

### Code Quality

```bash
# Linting
ruff check aiforge/ tests/

# Type checking
mypy aiforge/

# Format
ruff format aiforge/ tests/
```

### Make Commands

```bash
make install         # Install dependencies
make test            # Run tests
make lint            # Run linter
make serve           # Start API server
make playground      # Start Streamlit UI
make docker          # Build Docker image
```

---

## Supported Providers

| Provider | LLM Models | Embedding Models | Local? |
|----------|-----------|-----------------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini | text-embedding-3-small, text-embedding-3-large | No |
| **Anthropic** | claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5 | voyage-3 (via Voyage AI) | No |
| **Ollama** | llama3, mistral, codellama, phi-3 | nomic-embed-text, mxbai-embed-large | Yes |
| **Azure OpenAI** | Same as OpenAI (deployed models) | Same as OpenAI | No |
| **Google Gemini** | gemini-3.8-flash, gemini-3.1-pro-preview, gemini-2.5-* | gemini-embedding-2 | No |
| **Amazon Bedrock** | Claude, Amazon Nova and other Converse models | amazon.titan-embed-text-v2:0 | No |
| **Mistral AI** | mistral-large-4, mistral-medium-3-5, codestral | mistral-embed | No |

See [Providers](docs/providers.md) for setup, default models and install extras.

---

## Extending AIForge

AIForge ships with a **plugin system** powered by a unified registry. Register custom modules, providers, and vector stores — then use them interchangeably with built-ins.

### Custom Module

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
```

### Custom Provider

```python
from aiforge.core.provider import LLMProvider, EmbeddingProvider
from aiforge.registry import register_provider

class MyProvider(LLMProvider):
    def complete(self, messages, **kwargs): ...
    async def complete_async(self, messages, **kwargs): ...
    async def stream(self, messages, **kwargs): ...

class MyEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts): ...
    def embed_query(self, text): ...

register_provider("my_provider", MyProvider, MyEmbeddingProvider)
```

### Custom Vector Store

```python
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.registry import register_vectorstore

class WeaviateStore(VectorStoreAdapter):
    def add_documents(self, texts, embeddings, metadatas, ids): ...
    def query(self, embedding, top_k, filter, include_embeddings): ...
    def delete_collection(self, name): ...
    def list_collections(self): ...
    def collection_stats(self, name): ...

register_vectorstore("weaviate", WeaviateStore)
```

### Registry Discovery

```python
from aiforge.registry import list_modules, list_providers, list_vectorstores

print(list_modules())       # ['chat', 'classify', ..., 'summarize']
print(list_providers())     # ['openai', 'anthropic', 'ollama', 'azure', 'gemini', 'bedrock', 'mistral']
print(list_vectorstores())  # ['chroma', 'pgvector', 'pinecone', 'qdrant']
```

See [Plugin System docs](docs/configuration.md#plugin-system) for full details.

---

## Cost Tracking

Every module call returns cost information:

```python
result = summarizer.run(text)
print(f"Cost: ${result.cost:.4f}")
print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")

# Session aggregates
from aiforge import CostTracker
tracker = CostTracker.get_session()
print(f"Total: ${tracker.total_cost:.4f}")
print(tracker.by_module)   # {"summarize": 0.012, "qa": 0.025, ...}
print(tracker.by_model)    # {"gpt-4o": 0.040, "text-embedding-3-small": 0.002}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Kushang Patel** — Lead Developer