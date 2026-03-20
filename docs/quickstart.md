# Quick Start Guide

Get AIForge running in under 5 minutes.

---

## 1. Installation

```bash
# Core + default provider (OpenAI)
pip install aiforge

# Specific providers
pip install aiforge[anthropic]
pip install aiforge[local]             # Ollama
pip install aiforge[azure]

# Everything
pip install aiforge[all]

# Vector store backends
pip install aiforge[pinecone]          # Pinecone
pip install aiforge[qdrant]            # Qdrant
pip install aiforge[pgvector]          # PostgreSQL + pgvector

# API server + UI
pip install aiforge[server]            # includes FastAPI + Streamlit
```

## 2. Scaffold Config

```bash
aiforge init
# Creates aiforge.yaml + .env.example + .gitignore
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
pip install aiforge

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
pip install aiforge[openai]        # Core + OpenAI provider
pip install aiforge[anthropic]     # Core + Anthropic provider
pip install aiforge[local]         # Core + Ollama provider
pip install aiforge[all]           # Everything

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

cleaned = m.redact("Call me at 555-0123")
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

## Launch Playground UI

```bash
aiforge playground
# Opens Streamlit app at http://localhost:8501
```

---

## Next Steps

- [Configuration](configuration.md) — YAML, env vars, and code-level config
- [Providers](providers.md) — Swap between OpenAI, Anthropic, Ollama, Azure
- [Pipelines](pipelines.md) — Chain modules together
- [API Server](api-server.md) — REST API deployment
- [Playground](playground.md) — Interactive UI
- [Cost Tracking](cost-tracking.md) — Monitor token usage and costs
