# UI Playground

AIForge includes a built-in Streamlit playground for interactive prototyping and demos.

---

## Launch

```bash
aiforge playground
# Opens Streamlit app at http://localhost:8501
```

---

## Tabs

| Tab | Description |
|-----|-------------|
| **Summarize** | Paste text or upload file → get summary. Style selector. |
| **Q&A** | Ingest panel (URL/file upload) + question input → answer with sources |
| **Extract** | Schema builder (visual JSON schema) + text input → structured output |
| **Classify** | Label editor + text input → classification with confidence bars |
| **Search** | Index panel + search input → ranked results |
| **Generate** | Template editor + variable inputs → generated content |
| **Chat** | Chat interface with tool toggle |
| **Moderate** | Text input → flagged items highlighted in-line |
| **Pipeline** | Visual pipeline builder — drag modules into a sequence |
| **Settings** | Provider selector, model config, cost dashboard |

---

## Shared Sidebar

Every tab shares a common sidebar for provider/model selection and session stats:

```
┌────────────────────────┐
│ 🔧 AIForge Playground  │
│                        │
│ Provider: [OpenAI ▾]   │
│ Model:    [gpt-4o  ▾]  │
│ Temp:     [0.0 ──●── ] │
│                        │
│ ── Session Stats ──    │
│ Requests:  12          │
│ Tokens:    8,421       │
│ Cost:      $0.042      │
│                        │
│ ── Collections ──      │
│ • default (1,247 docs) │
│ • products (89 docs)   │
└────────────────────────┘
```

---

## Tab Details

### Summarize Tab

- **Input**: Paste text or upload a file (txt, md, pdf, docx)
- **Controls**: Style selector (`concise`, `detailed`, `bullet_points`, `executive`), max length slider
- **Output**: Generated summary with token count and cost display

### Q&A Tab

- **Ingest Panel**: Upload files, paste URLs, or provide git repo URLs for ingestion
- **Collection Selector**: Choose which collection to query
- **Question Input**: Natural language query
- **Output**: Answer with source citations, confidence score, and agent trace (expandable)

### Extract Tab

- **Schema Builder**: Visual JSON schema editor for defining output structure
- **Input**: Paste unstructured text
- **Output**: Extracted structured data displayed as JSON, with validation status

### Classify Tab

- **Label Editor**: Add/remove classification labels
- **Examples**: Optional few-shot examples for improved accuracy
- **Input**: Text to classify
- **Output**: Label with confidence score and horizontal bar chart of all scores

### Search Tab

- **Index Panel**: Add documents to a collection (text + metadata)
- **Search Input**: Natural language query with optional metadata filters
- **Output**: Ranked results with similarity scores

### Generate Tab

- **Template Editor**: Write templates with `{variable}` placeholders
- **Variable Inputs**: Auto-generated input fields for each template variable
- **Guardrails Toggle**: Enable/disable guardrails
- **Output**: Generated content with guardrail status

### Chat Tab

- **Chat Interface**: Multi-turn conversation with message history
- **Tool Toggle**: Enable/disable agent tools (calculator, web search, etc.)
- **System Prompt**: Editable system prompt
- **Controls**: Memory window slider, reset conversation button

### Moderate Tab

- **Input**: Paste text for safety analysis
- **Check Selectors**: Toggle PII, toxicity, prompt injection checks
- **Output**: Flagged items highlighted in-line with badges
- **Redact Button**: One-click PII redaction with before/after view

### Pipeline Tab

- **Visual Builder**: Drag-and-drop module sequence
- **Configuration**: Per-step module settings
- **Input**: Text to process through the pipeline
- **Output**: Results from each pipeline step displayed in sequence

### Settings Tab

- **Provider Selector**: Switch between OpenAI, Anthropic, Ollama, Azure
- **Model Config**: Model name, temperature, and other parameters
- **Cost Dashboard**: Session-level cost tracking with per-module and per-model breakdowns
- **Collection Management**: View, create, and delete vector store collections

---

## Architecture

Each module includes a `widget.py` file that provides its Streamlit UI component:

```
aiforge/modules/
├── summarize/widget.py
├── qa/widget.py
├── extract/widget.py
├── classify/widget.py
├── search/widget.py
├── generate/widget.py
├── chat/widget.py
└── moderate/widget.py
```

The main app (`aiforge/playground/app.py`) acts as a tab router, loading each module's widget into its own Streamlit tab. Pages in `aiforge/playground/pages/` handle per-tab layout and logic.

---

## Docker Deployment

Run the playground alongside the API server with Docker Compose:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: aiforge serve --host 0.0.0.0 --port 8000

  playground:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AIFORGE_API_BASE=http://api:8000
    command: aiforge playground --port 8501
    depends_on:
      - api
```

---

## See Also

- [API Server](api-server.md) — REST API that the playground calls
- [Quick Start](quickstart.md) — Getting started
- Module docs for individual module APIs
