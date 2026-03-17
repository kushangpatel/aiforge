# Module: Summarize

`aiforge.summarize` — Text Summarization

---

## Purpose

Summarize any text or document. Automatically handles long documents by chunking + map-reduce.

---

## Public API

```python
from aiforge import Summarizer

s = Summarizer(
    style="concise",           # "concise" | "detailed" | "bullet_points" | "executive"
    max_length=500,            # target summary length (tokens)
    language="en",             # output language
)

# Simple text
result = s.run("Long article text here...")
print(result.summary)          # The summary text
print(result.input_tokens)     # Input tokens used
print(result.output_tokens)    # Output tokens used
print(result.cost)             # Cost in USD

# Document file
result = s.run_file("report.pdf")

# Long document (auto map-reduce)
result = s.run("50-page document...", strategy="map_reduce")
# strategy: "stuff" (fit in one call) | "map_reduce" (chunk + summarize + combine)
```

### Async Support

```python
# Async
result = await s.arun(text)

# Streaming
async for chunk in s.arun_stream(text):
    print(chunk, end="")

# Streaming with callback
def on_token(token: str):
    sys.stdout.write(token)

s.run(text, on_token=on_token)
```

---

## Schemas

### Request

```python
class SummarizeRequest(BaseModel):
    text: str
    style: Literal["concise", "detailed", "bullet_points", "executive"] = "concise"
    max_length: int = 500
    language: str = "en"
```

### Response

```python
class SummarizeResponse(BaseModel):
    summary: str
    input_tokens: int
    output_tokens: int
    cost: float
    strategy_used: Literal["stuff", "map_reduce"]
    chunks_processed: int       # 1 if stuff, N if map_reduce
    trace_id: str

    def to_text(self) -> str:
        return self.summary
```

---

## Long Document Strategy

The context window threshold is **model-aware** — AIForge reads the active model's max token limit from a built-in registry (e.g., `gpt-4o` = 128K, `claude-3.5-sonnet` = 200K, `llama3` = 8K). If no entry exists, the default threshold is 8,192 tokens.

```
Input tokens > (model_context_window × 0.8)?   # 80% to leave room for output
  ├── No  → "stuff" strategy: send entire text in one LLM call
  └── Yes → "map_reduce" strategy:
              1. Split into overlapping chunks (chunk_size = context_window × 0.6)
              2. Summarize each chunk independently (map)
              3. Combine chunk summaries into final summary (reduce)
              4. If combined still too long, reduce again recursively
```

---

## Prompt Templates

Customizable templates in `aiforge/modules/summarize/prompts.py`:

```python
SUMMARIZE_CONCISE = """Summarize the following text in {max_length} words or fewer.
Focus on the key points and main conclusions.

Text:
{text}

Summary:"""

SUMMARIZE_MAP = """Summarize this section concisely:
{chunk}"""

SUMMARIZE_REDUCE = """Combine these section summaries into one coherent summary
of {max_length} words or fewer:
{summaries}"""
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-SUM-001 | Summarize text | P0 | Accept text string, return summary |
| FR-SUM-002 | Summarize file | P1 | Accept file path (txt, md, pdf, docx) |
| FR-SUM-003 | Style selection | P0 | concise, detailed, bullet_points, executive |
| FR-SUM-004 | Long document handling | P0 | Auto map-reduce for docs exceeding context window |
| FR-SUM-005 | Streaming | P1 | Stream summary tokens as they generate |
| FR-SUM-006 | Cost tracking | P0 | Return token counts and cost |

---

## File Structure

```
aiforge/modules/summarize/
├── __init__.py
├── module.py          # Summarizer class
├── prompts.py         # Prompt templates
├── schemas.py         # SummarizeRequest, SummarizeResponse
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic summarization example
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /summarize` endpoint
