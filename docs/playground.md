# Playground

AIForge ships with a built-in **Streamlit playground** — a browser-based UI
for interactive testing, prototyping, and demos. Every module has its own
page with configurable controls, and a visual pipeline builder lets you chain
modules without writing any code.

---

## Table of Contents

- [Why Use the Playground?](#why-use-the-playground)
- [Launch](#launch)
  - [CLI](#cli)
  - [Custom Port](#custom-port)
  - [Docker Compose](#docker-compose)
- [Installation](#installation)
- [Pages Overview](#pages-overview)
- [Summarize](#summarize)
- [Document Q&A](#document-qa)
- [Semantic Search](#semantic-search)
- [Classify](#classify)
- [Extract](#extract)
- [Generate](#generate)
- [Chat](#chat)
  - [Built-in Tools](#built-in-tools)
  - [Memory Modes](#memory-modes)
- [Moderate](#moderate)
- [Pipeline Builder](#pipeline-builder)
  - [Available Modules](#available-modules)
  - [Step Configuration](#step-configuration)
  - [Results Display](#results-display)
- [Settings Dashboard](#settings-dashboard)
  - [Configuration Sections](#configuration-sections)
  - [Session Usage Metrics](#session-usage-metrics)
  - [API Key Status](#api-key-status)
- [Architecture](#architecture)
  - [File Structure](#file-structure)
  - [Widget Pattern](#widget-pattern)
  - [Page Router](#page-router)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Why Use the Playground?

| Scenario | Playground helps by... |
|----------|----------------------|
| Exploring a new module | Try different parameters interactively, see results instantly |
| Choosing between models | Switch provider/model in Settings, compare quality and cost side-by-side |
| Building a pipeline | Visual drag-and-drop step builder — no code needed |
| Stakeholder demos | Polished UI with live cost tracking, no terminal required |
| Debugging prompts | View full response metadata: tokens, cost, latency, model used |
| Testing document Q&A | Ingest files, browse collections, ask questions — all in one page |
| Evaluating content safety | Run moderator checks and redaction on sample text |

---

## Launch

### CLI

```bash
aiforge playground
# Opens at http://localhost:8501
```

### Custom Port

```bash
aiforge playground --port 9000
# Opens at http://localhost:9000
```

The port can also be set via configuration:

```yaml
# aiforge.yaml
playground_port: 8501
```

```bash
export AIFORGE_PLAYGROUND_PORT=9000
```

### Docker Compose

Run the playground alongside the API server:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    command: aiforge serve --host 0.0.0.0 --port 8000

  playground:
    build: .
    ports:
      - "8501:8501"
    env_file: .env
    environment:
      - AIFORGE_API_BASE=http://api:8000
    command: aiforge playground --port 8501
    depends_on:
      - api
```

The playground can operate in two modes:
- **Direct mode** (default): imports AIForge modules directly — same process,
  no network calls
- **API mode**: set `AIFORGE_API_BASE` to point at a running API server — the
  playground proxies calls over HTTP

---

## Installation

The playground requires the `streamlit` extra:

```bash
# Playground only
pip install -e "aiforge[playground]"

# Everything (playground + all vector stores + tracing + etc.)
pip install -e "aiforge[all]"
```

Requires `streamlit>=1.40.0`.

---

## Pages Overview

The playground has 9 module pages plus a Settings dashboard:

| Page | Module | Description |
|------|--------|-------------|
| **Summarize** | `Summarizer` | Text summarisation with style/strategy controls |
| **Q&A** | `DocQA` | Document ingestion, collection management, question answering |
| **Search** | `SemanticSearch` | Index documents and run semantic queries |
| **Classify** | `Classifier` | Multi-label classification with few-shot examples |
| **Extract** | `Extractor` | JSON schema-driven structured extraction |
| **Generate** | `Generator` | Template-based or freeform text generation |
| **Chat** | `ChatAgent` | Multi-turn conversation with tools and memory |
| **Moderate** | `Moderator` | Content safety checks and PII redaction |
| **Pipeline** | `Pipeline` | Visual multi-step pipeline builder |
| **Settings** | — | Read-only configuration dashboard and session metrics |

---

## Summarize

Paste or upload text and generate summaries with configurable parameters.

**Controls:**

| Control | Type | Options / Range | Default |
|---------|------|-----------------|---------|
| Style | Selectbox | `concise`, `detailed`, `bullet_points`, `executive` | `concise` |
| Max length | Slider | 50–2,000 words | 200 |
| Strategy | Selectbox | `auto-detect`, `stuff`, `map_reduce` | `auto-detect` |

**Input:** Text area (300px height)

**Output:**
- Generated summary text
- Expandable metrics panel:
  - Strategy used (`stuff` or `map_reduce`)
  - Chunks processed (for `map_reduce`)
  - Latency (ms), cost (USD), input/output tokens, model

---

## Document Q&A

Three-mode page for document ingestion, querying, and collection management.

### Ingest Mode

Upload documents to build a searchable knowledge base.

| Control | Type | Description |
|---------|------|-------------|
| Source | Text input | File path, directory path, or git URL |
| Extension filter | Text input | Comma-separated list (e.g., `.py,.md,.txt`) |

**Output:** Chunks indexed, files processed, skipped files, ingestion metrics.

### Ask Mode

Query ingested documents with natural language.

| Control | Type | Description |
|---------|------|-------------|
| Collection | Text input | Collection name to search |
| Query | Text input | Natural language question |
| Source filter | Text input | Optional — filter results by source file path |
| Show agent trace | Checkbox | Expand the multi-step reasoning trace |

**Output:** Answer with confidence score, source citations, agent iterations
(if trace enabled), cost and token metrics.

### Collections Mode

Browse and manage indexed collections.

- Read-only list of all collections
- Per-collection stats (chunk count)
- Delete button per collection

---

## Semantic Search

Two-mode page for indexing and searching documents.

### Index Mode

| Control | Type | Description |
|---------|------|-------------|
| Documents | Text area | One per line, format: `id \| text` |

**Output:** Documents indexed count, total documents in collection, metrics.

### Search Mode

| Control | Type | Description |
|---------|------|-------------|
| Query | Text input | Natural language search query |
| Top K | Slider (1–20) | Number of results to return (default: 5) |

**Output:** Ranked results with similarity score, document ID, text, and
metadata. Cost and token metrics.

---

## Classify

Text classification with optional few-shot examples.

| Control | Type | Description |
|---------|------|-------------|
| Labels | Text area | One label per line |
| Multi-label | Checkbox | Allow multiple labels per input |
| Few-shot examples | Toggle + text area | Format: `text \| label` (one per line) |

**Input:** Text area (140px height)

**Output:**
- Predicted label with confidence badge (colour-coded)
- Rationale (caption text)
- Confidence scores for all labels (table)
- Latency, cost, and token metrics

---

## Extract

Structured data extraction using JSON Schema.

| Control | Type | Description |
|---------|------|-------------|
| Output schema | Text area | JSON Schema defining the extraction target |
| Few-shot examples | Toggle + text area | JSON array of `{"input": ..., "output": ...}` objects |

**Input:** Text area (180px height)

**Output:**
- Validation status (success or warning)
- Extracted data as formatted JSON (or raw LLM output if validation fails)
- Metrics: validation passed/failed, retry count, latency, cost, tokens

---

## Generate

Template-based or freeform text generation with guardrails.

### Template Mode

| Control | Type | Description |
|---------|------|-------------|
| Template | Text area | Template with `{variable}` placeholders |
| Variables | Auto-generated inputs | One text field per detected `{variable}` |
| Max tokens | Number input | Hard cap on output length |
| Temperature | Slider | Sampling temperature (0.0–2.0) |
| Guardrails | Multiselect | Safety checks to apply to output |

### Freeform Mode

| Control | Type | Description |
|---------|------|-------------|
| Prompt | Text area | Free-text prompt |
| Max tokens | Number input | Hard cap on output length |
| Temperature | Slider | Sampling temperature (0.0–2.0) |
| Guardrails | Multiselect | Safety checks to apply to output |

**Output:** Generated text, guardrail flags (if any triggered), retry count,
latency, cost, and token metrics.

---

## Chat

Full multi-turn conversational interface with agent tools and memory.

### Sidebar Configuration

| Control | Type | Options / Range | Default |
|---------|------|-----------------|---------|
| System prompt | Text area (800px) | Free text | Default ChatAgent prompt |
| Memory mode | Selectbox | `buffer`, `summary`, `none` | `buffer` |
| Buffer size | Slider | 2–50 messages | 20 |
| Enable tools | Checkbox | — | `True` |
| Reset conversation | Button | — | — |

### Chat Interface

- Message history displayed with user/assistant roles
- Tool calls shown inline in expandable panels
- Per-message cost and token metrics
- Standard `st.chat_input` box at the bottom

### Built-in Tools

The Chat page provides four sandboxed tools that the agent can invoke:

| Tool | Description | Details |
|------|-------------|---------|
| `calculator` | Evaluate math expressions | Supports constants (`pi`, `e`), functions (`sqrt`, `sin`, `cos`, `log`, `abs`, etc.) |
| `current_time` | Get current UTC timestamp | Returns ISO 8601 format |
| `research` | Web + knowledge base search | DuckDuckGo Lite web search + vector store RAG search |
| `unit_converter` | Unit conversions | Temperature, length, weight, volume, angle |

Tools are safe and sandboxed — the calculator uses a restricted evaluator (no
`exec`/`eval`), and research is read-only.

### Memory Modes

| Mode | Behaviour |
|------|-----------|
| `buffer` | Keeps the last N messages in context (configurable via slider) |
| `summary` | Periodically summarises earlier messages to fit context window |
| `none` | No memory — each message is independent |

---

## Moderate

Content safety checking and PII redaction.

| Control | Type | Options | Default |
|---------|------|---------|---------|
| Checks | Multiselect | `pii`, `toxicity`, `prompt_injection` | All selected |
| PII types | Multiselect | `email`, `phone`, `ssn`, `credit_card` | All selected |
| Action | Radio | `check`, `redact` | `check` |

**Input:** Text area (160px height)

**Output (check mode):**
- Flagged items table (Type, Subtype, Span, Confidence)
- Or "Clean" status with checkmark

**Output (redact mode):**
- Redacted text in a code block
- Before/after comparison

Metrics: latency, cost, tokens.

---

## Pipeline Builder

Visual interface for constructing and executing multi-step pipelines.

### Available Modules

| Module | Config Fields |
|--------|--------------|
| `Summarizer` | Style (selectbox), max length (slider) |
| `Classifier` | Labels (text area), multi-label (checkbox) |
| `Generator` | Temperature (slider) |
| `Moderator` | Checks (multiselect) |

> `DocQA`, `SemanticSearch`, and `Extractor` are excluded from the pipeline
> builder because they require pre-ingested data or complex schema inputs that
> don't fit the step-chaining model.

### Step Configuration

1. Click **Add Step** (+) to add a new pipeline step
2. Enter a unique **step name**
3. Select a **module** from the dropdown
4. Expand the step to configure module-specific parameters
5. Click **Delete** (✕) to remove a step
6. Steps execute top-to-bottom — output of each feeds into the next

Steps are tracked with stable UUIDs, so adding/removing steps doesn't disrupt
other step configurations.

### Results Display

After execution:

- **Aggregate metrics:** Total cost (USD), total tokens
- **Per-step results:** Expandable panels showing:
  - Output text (via `response.to_text()`)
  - Full response details (JSON dump of all fields)
- **Halt status:** If a Router's `None` branch was triggered, shows which step
  halted the pipeline

**Input:** Large text area (200px height) for the initial pipeline input.

---

## Settings Dashboard

A read-only view of the current AIForge configuration and session metrics.
No settings are editable from this page — it reflects what's loaded from
`aiforge.yaml`, `.env`, and environment variables.

### Configuration Sections

| Section | Fields Displayed |
|---------|-----------------|
| **Provider & Model** | `PROVIDER`, `MODEL`, `EMBEDDING_MODEL`, `TEMPERATURE`, `MAX_TOKENS` |
| **Model Router** | `MODEL_ROUTER_ENABLED`, `DEFAULT_QUALITY`, `PREFER_LOCAL`, `CUSTOM_MODELS` |
| **Vector Store** | Backend type + backend-specific settings (Chroma dir, Pinecone index, pgvector DSN, Qdrant URL/collection) |
| **Rate Limiting & Cost** | `RATE_LIMIT_RPM`, `RATE_LIMIT_TPM`, `COST_TRACKING_ENABLED`, `COST_WARN_THRESHOLD` |
| **Request Cache** | `CACHE_ENABLED`, `CACHE_BACKEND`, `CACHE_TTL`, `CACHE_MAX_ENTRIES` |
| **Webhooks** | `WEBHOOK_ENABLED`, `WEBHOOK_TIMEOUT`, `WEBHOOK_ALLOW_INSECURE` |
| **Observability** | `TRACING_ENABLED`, `TRACING_BACKEND`, `LOG_LEVEL`, `LOG_FORMAT`, `LANGSMITH_PROJECT` |
| **API Server** | `API_HOST`, `API_PORT`, API key presence (lock icon) |

### Session Usage Metrics

When calls have been made during the session, the Settings page displays live
cost tracking data from the `CostTracker` singleton:

- **Total cost** (USD)
- **Total tokens** consumed
- **API call count**
- **By-module breakdown** — cost per module (Summarizer, Classifier, etc.)
- **By-model breakdown** — cost per model (gpt-4o, gpt-4o-mini, etc.)

### API Key Status

API keys are shown as **presence indicators only** — values are never displayed:

| Key | Status Icon |
|-----|-------------|
| OpenAI | 🔒 Configured / 🔓 Not set |
| Anthropic | 🔒 Configured / 🔓 Not set |
| Voyage AI | 🔒 Configured / 🔓 Not set |
| Azure OpenAI | 🔒 Configured / 🔓 Not set |
| Pinecone | 🔒 Configured / 🔓 Not set |
| Qdrant | 🔒 Configured / 🔓 Not set |
| LangSmith | 🔒 Configured / 🔓 Not set |

Custom overrides (`CUSTOM_PRICING`, `CUSTOM_CONTEXT_WINDOWS`, `CUSTOM_PROMPTS`)
are shown only when configured.

---

## Architecture

### File Structure

```
aiforge/
├── playground/
│   ├── app.py              ← Main Streamlit app + page config
│   └── pages/
│       ├── summarize.py    ← Loads summarize widget
│       ├── qa.py           ← Loads QA widget (3-mode)
│       ├── search.py       ← Loads search widget
│       ├── classify.py     ← Loads classify widget
│       ├── extract.py      ← Loads extract widget
│       ├── generate.py     ← Loads generate widget
│       ├── chat.py         ← Loads chat widget
│       ├── moderate.py     ← Loads moderate widget
│       ├── pipeline.py     ← Loads pipeline widget
│       └── settings.py     ← Config dashboard
├── modules/
│   ├── summarize/widget.py
│   ├── qa/widget.py
│   ├── extract/widget.py
│   ├── classify/widget.py
│   ├── search/widget.py
│   ├── generate/widget.py
│   ├── chat/widget.py      ← 450+ lines (tools, memory, chat UI)
│   └── moderate/widget.py
└── pipeline/
    └── widget.py            ← 400+ lines (step builder, execution)
```

### Widget Pattern

Each module's `widget.py` follows a consistent pattern:

```python
# aiforge/modules/<module>/widget.py

import streamlit as st
from aiforge import <Module>

def render():
    """Render the <Module> playground widget."""
    # 1. Sidebar / controls
    style = st.selectbox("Style", ["concise", "detailed", ...])

    # 2. Input
    text = st.text_area("Input text", height=200)

    # 3. Execute
    if st.button("Run", type="primary"):
        module = <Module>(style=style, ...)
        result = module.run(text)

        # 4. Display results
        st.write(result.to_text())

        # 5. Metrics expander
        with st.expander("Metrics"):
            st.metric("Cost", f"${result.cost:.4f}")
            st.metric("Tokens", result.input_tokens + result.output_tokens)
            st.metric("Latency", f"{result.latency_ms:.0f}ms")
```

### Page Router

The main app (`app.py`) uses `st.navigation()` to render pages:

```python
# aiforge/playground/app.py (simplified)
import streamlit as st

st.set_page_config(
    page_title="AIForge Playground",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pages auto-discovered from pages/ directory
# st.navigation() handles page routing
```

Pages are discovered from the `pages/` directory automatically by Streamlit.
Each page imports and calls its module's `widget.render()` function.

---

## Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AIFORGE_PLAYGROUND_PORT` | `int` | `8501` | Streamlit server port |

The playground reads all other settings from the standard AIForge configuration
(`aiforge.yaml`, `.env`, environment variables). Changes to provider, model,
vector store, etc. are reflected immediately on the Settings page and affect
all module executions.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: streamlit` | Playground extra not installed | `pip install -e "aiforge[playground]"` |
| Port already in use | Another process on port 8501 | `aiforge playground --port 9000` or stop the other process |
| API keys show 🔓 | Keys not configured | Set keys in `.env` or environment variables |
| Chat tools don't appear | "Enable tools" unchecked | Check the "Enable built-in tools" checkbox in the sidebar |
| Pipeline step config lost | Step was deleted and re-added | Steps use stable UUIDs — don't delete/re-add, edit in place |
| Settings page shows stale data | Config loaded at startup | Restart the playground to pick up config changes |
| No cost data on Settings page | No module calls made yet | Run at least one module, then check Settings |
| Ingest fails with "no files found" | Path or extension filter incorrect | Check the source path exists and extensions match (e.g., `.py,.md`) |
| Chat memory fills up | Buffer size too small for long conversations | Increase buffer size slider or switch to `summary` mode |
| Pipeline halts unexpectedly | Router matched a `None` branch | Check classification labels in the Classify step config |

---

## See Also

- [Configuration](configuration.md) — Full configuration reference
- [API Server](api-server.md) — REST API that the playground can proxy to
- [Pipelines](pipelines.md) — Pipeline SDK documentation
- [Cost Tracking](cost-tracking.md) — Cost tracking and pricing details
- [Modules](modules/) — Per-module SDK documentation
- [Quick Start](quickstart.md) — Getting started
