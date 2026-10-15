# Quick Start Guide

Get AIForge running in under 5 minutes.

---

## 1. Installation

```bash
# Core + default provider (OpenAI)
pip install aiforge-kit

# Specific providers
pip install "aiforge-kit[anthropic]"        # Anthropic + Voyage AI embeddings
pip install "aiforge-kit[local]"            # Ollama

# Everything
pip install "aiforge-kit[all]"

# Vector store backends
pip install "aiforge-kit[pinecone]"         # Pinecone
pip install "aiforge-kit[qdrant]"           # Qdrant
pip install "aiforge-kit[pgvector]"         # PostgreSQL + pgvector

# API server + Playground UI
pip install "aiforge-kit[server]"           # FastAPI + Uvicorn
pip install "aiforge-kit[playground]"       # Streamlit UI
```

To install the latest unreleased code instead, install from GitHub:
`pip install "aiforge-kit[anthropic] @ git+https://github.com/kushangpatel/aiforge.git"`.

### Plugin System

AIForge auto-registers all 10 built-in modules and vector stores on import. You can extend it with custom plugins:

```python
from aiforge.registry import register_module, list_modules

# Register a custom module
register_module("sentiment", SentimentAnalyzer)

# See all available modules
print(list_modules())  # ['chat', 'classify', ..., 'sentiment', 'summarize']
```

See the [Plugin System](configuration.md#plugin-system) section for full details on custom modules, providers, and vector stores.

## 2. Scaffold Config

```bash
aiforge init
# Creates aiforge.yaml + .env
```

## 3. Set API Key

```bash
export AIFORGE_OPENAI_API_KEY=sk-...
```

## 4. Use in Code

```python
from aiforge import Summarizer

s = Summarizer()
result = s.run("AIForge is a plug-and-play AI toolkit for product teams...")
print(result.summary)
print(result.cost)  # Cost in USD
```

---

## End-to-End Example

```bash
# 1. Install
pip install aiforge-kit

# 2. Scaffold config
aiforge init

# 3. Set API key
export AIFORGE_OPENAI_API_KEY=sk-...

# 4. Use in code
python -c "
from aiforge import Summarizer
s = Summarizer()
print(s.run('AIForge is a plug-and-play AI toolkit...').summary)
"
```

---

## Module Independence

Each module is self-contained and can be used independently:

```python
# Install only what you need
pip install aiforge-kit                        # Core + OpenAI provider
pip install "aiforge-kit[anthropic]"         # Core + Anthropic + Voyage AI
pip install "aiforge-kit[local]"               # Core + Ollama provider
pip install "aiforge-kit[all]"                 # Everything

# Use only what you need
from aiforge import Summarizer     # Just summarization
from aiforge import DocQA          # Just document Q&A
from aiforge import Pipeline       # Chain multiple modules
```

---

## Try Each Module

### Summarize

```python
from aiforge import Summarizer

s = Summarizer(style="bullet_points")
result = s.run("Your long text here...")
print(result.summary)
```

### Document Q&A (RAG)

```python
from aiforge import DocQA

qa = DocQA(collection="my_docs")
qa.ingest("./docs/")
answer = qa.ask("How does authentication work?")
print(answer.text)
print(answer.sources)
```

### Extract Structured Data

```python
from aiforge import Extractor
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

e = Extractor(output_schema=Invoice)
result = e.run("Invoice from Acme Corp. Total: $86.00 USD.")
print(result.data.vendor)  # "Acme Corp"
```

### Classify Text

```python
from aiforge import Classifier

c = Classifier(labels=["bug", "feature_request", "question"])
result = c.run("The login button doesn't work on mobile")
print(result.label)       # "bug"
print(result.confidence)  # 0.94
```

### Semantic Search

```python
from aiforge import SemanticSearch

ss = SemanticSearch(collection="products")
ss.index([
    {"id": "1", "text": "Wireless headphones", "metadata": {"price": 299}},
    {"id": "2", "text": "Bluetooth speaker", "metadata": {"price": 79}},
])
results = ss.query("something to listen to music")
for r in results.hits:
    print(f"{r.score:.3f} | {r.text}")
```

### Generate Content

```python
from aiforge import Generator

g = Generator(
    template="Write a {tone} product description for {product_name}.",
    guardrails=["no_competitor_mentions"]
)
result = g.run(product_name="AIForge", tone="professional")
print(result.text)
```

### Chat Agent

```python
from aiforge import ChatAgent

agent = ChatAgent(system_prompt="You are a helpful assistant.")
response = agent.chat("What's the return policy?")
print(response.message)

# Multi-turn (memory maintained)
response = agent.chat("What about international orders?")
```

### Content Moderation

```python
from aiforge import Moderator

m = Moderator(checks=["pii", "toxicity", "prompt_injection"])
result = m.check("My email is john@example.com")
print(result.flagged)  # True

cleaned = m.redact("Call me at 555-867-5309")
print(cleaned.text)  # "Call me at [PHONE]"
```

---

## Chain Modules with Pipelines

```python
from aiforge import Pipeline, Extractor, Classifier, Summarizer

pipe = Pipeline([
    ("extract", Extractor(output_schema=TicketSchema)),
    ("classify", Classifier(labels=["critical", "high", "medium", "low"])),
    ("summarize", Summarizer(style="concise")),
])

result = pipe.run("Long customer complaint email text here...")
print(result["classify"].label)
print(result["summarize"].summary)
```

---

## Launch API Server

```bash
aiforge serve --host 0.0.0.0 --port 8000
```

> **SDK or API?** If your code is Python, import the SDK directly for zero
> overhead and full type safety.  Use the HTTP API when non-Python services need
> access or you want a shared deployment.  See the
> [SDK vs HTTP API](api-server.md#sdk-vs-http-api--when-to-use-which) guide for
> details.

## Launch Playground UI

```bash
aiforge playground
# Opens Streamlit app at http://localhost:8501
```

---

## Next Steps

- [CLI](cli.md) — Command-line interface reference
- [Configuration](configuration.md) — YAML, env vars, and code-level config
- [Providers](providers.md) — Swap between OpenAI, Anthropic, Ollama, Azure
- [Pipelines](pipelines.md) — Chain modules together
- [API Server](api-server.md) — REST API deployment
- [Playground](playground.md) — Interactive UI
- [Cost Tracking](cost-tracking.md) — Monitor token usage and costs
