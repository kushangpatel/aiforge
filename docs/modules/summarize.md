# Module: Summarize

`aiforge.modules.summarize` — Text summarization with automatic strategy selection. Short texts are summarized in a single LLM call (`stuff`); long texts that exceed the model's context window are split into overlapping chunks, summarized independently, and merged (`map_reduce`). Four built-in styles, configurable max length, and full cost tracking across all LLM calls.

---

## Installation

The summarize module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import Summarizer
from aiforge import SummarizeResponse  # type
```

---

## Quick Start

```python
from aiforge import Summarizer

s = Summarizer(style="concise")

result = s.summarize("Long article text here...")
print(result.summary)           # The summary text
print(result.strategy_used)     # "stuff" or "map_reduce"
print(f"Cost: ${result.cost:.4f}")
```

---

## Constructor

```python
Summarizer(
    style: SummarizeStyle = "concise",
    max_length: int = 500,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    # Inherited from BaseModule:
    quality: str | None = None,
    provider: str | None = None,
    **kwargs,  # AIForgeSettings overrides
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | `SummarizeStyle` | `"concise"` | Summary style. One of: `"concise"`, `"detailed"`, `"bullet_points"`, `"executive"`. |
| `max_length` | `int` | `500` | Target summary length in words. Must be ≥ 1. Passed to the prompt template as `{max_length}`. |
| `temperature` | `float` | `0.0` | LLM sampling temperature. Must be in `[0.0, 2.0]`. Lower = more deterministic. |
| `system_prompt` | `str \| None` | `None` | Override the default system prompt. See [Prompt Resolution](#prompt-resolution). |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`. |

### Summary Styles

| Style | Description | Best For |
|-------|-------------|----------|
| `"concise"` | Key points and main conclusions, minimal length | Quick overviews, chat responses |
| `"detailed"` | All major points, supporting evidence, and conclusions | Research, documentation |
| `"bullet_points"` | One key point per bullet | Meeting notes, action items |
| `"executive"` | Focus on decisions, actions, and outcomes | Reports, stakeholder briefings |

### Validation Rules

The constructor raises `ValueError` for:
- `style` not in `{"concise", "detailed", "bullet_points", "executive"}`
- `max_length < 1`
- `temperature` outside `[0.0, 2.0]`

### Examples

```python
from aiforge import Summarizer

# Default — concise, 500 words, deterministic
s = Summarizer()

# Detailed summary, up to 1000 words
s = Summarizer(style="detailed", max_length=1000)

# Bullet points with shorter output
s = Summarizer(style="bullet_points", max_length=200)

# Executive summary with specific model
s = Summarizer(style="executive", MODEL="gpt-4o")

# More creative summaries
s = Summarizer(temperature=0.7)

# Quality-based routing
s = Summarizer(quality="low")  # cheapest model
```

---

## Methods

### `summarize(text, *, strategy=None, system_prompt=None) -> SummarizeResponse`

Summarize text synchronously. Automatically selects `stuff` or `map_reduce` strategy based on text length vs. model context window.

```python
def summarize(
    self,
    text: str,
    *,
    strategy: str | None = None,
    system_prompt: str | None = None,
) -> SummarizeResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Source text to summarize. Must not be empty or whitespace-only. |
| `strategy` | `str \| None` | `None` | Force `"stuff"` or `"map_reduce"`. `None` auto-selects based on token count. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`SummarizeResponse`](#summarizeresponse)

**Raises:** `ValueError` if `text` is empty/whitespace or `strategy` is invalid.

```python
s = Summarizer(style="concise", max_length=200)

# Auto strategy selection
result = s.summarize("Short article text...")
print(result.summary)
print(result.strategy_used)     # "stuff" (text fits in context)
print(result.chunks_processed)  # 1

# Force map_reduce on short text (for testing)
result = s.summarize("Short text", strategy="map_reduce")
print(result.strategy_used)     # "map_reduce"
print(result.chunks_processed)  # 1+ chunks

# Long text — auto map_reduce
long_text = open("large_report.txt").read()  # 100,000 words
result = s.summarize(long_text)
print(result.strategy_used)     # "map_reduce"
print(result.chunks_processed)  # 5 (depends on text length & model context)
print(f"Cost: ${result.cost:.4f}")  # aggregated across all map + reduce calls
```

---

### `asummarize(text, *, strategy=None, system_prompt=None) -> SummarizeResponse`

Async version of `summarize()`. When using `map_reduce`, chunk summaries are processed **concurrently** via `asyncio.gather()` for faster throughput.

```python
async def asummarize(
    self,
    text: str,
    *,
    strategy: str | None = None,
    system_prompt: str | None = None,
) -> SummarizeResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Source text to summarize. |
| `strategy` | `str \| None` | `None` | Force strategy or auto-select. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`SummarizeResponse`](#summarizeresponse)

```python
result = await s.asummarize(long_text)
print(result.summary)
# Faster than sync for map_reduce — chunks run concurrently
```

---

### `run(text, **kwargs) -> SummarizeResponse`

BaseModule interface method. Delegates to `summarize()`. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> SummarizeResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to summarize. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`SummarizeResponse`](#summarizeresponse)

```python
result = s.run("Some text to summarize")
print(result.to_text())  # same as result.summary
```

---

### `arun(text, **kwargs) -> SummarizeResponse`

Async version of `run()`. Delegates to `asummarize()`.

```python
async def arun(self, text: str, **kwargs) -> SummarizeResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Summarize multiple texts concurrently. Inherited from `BaseModule`.

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
| `items` | `list[str]` | *(required)* | List of texts to summarize. |
| `concurrency` | `int` | `10` | Max concurrent summarizations. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

```python
articles = [
    "First article text...",
    "Second article text...",
    "Third article text...",
]

batch = s.run_batch(articles, concurrency=3)
for r in batch.results:
    print(f"  [{r.strategy_used}] {r.summary[:80]}...")
print(f"Total cost: ${batch.total_cost:.4f}")
```

---

## Response Types

### `SummarizeResponse`

Returned by `summarize()`, `asummarize()`, `run()`, and `arun()`.

```python
from aiforge import SummarizeResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `str` | The generated summary text. |
| `strategy_used` | `Literal["stuff", "map_reduce"]` | Which strategy was used. `"stuff"` = single call. `"map_reduce"` = chunked. |
| `chunks_processed` | `int` | Number of chunks processed. `1` for stuff, `N` for map_reduce. |
| `cost` | `float` | Total cost in USD across all LLM calls (map + reduce). |
| `input_tokens` | `int` | Total input tokens across all LLM calls. |
| `output_tokens` | `int` | Total output tokens across all LLM calls. |
| `model` | `str` | LLM model used. |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_text()` | `str` | Returns `summary`. Useful for pipeline chaining. |

```python
result = s.summarize("Long article...")

print(result.summary)           # "The article discusses..."
print(result.strategy_used)     # "stuff"
print(result.chunks_processed)  # 1
print(result.cost)              # 0.0012
print(result.input_tokens)      # 1500
print(result.output_tokens)     # 200
print(result.model)             # "gpt-4o-mini"
print(result.latency_ms)        # 850.3

# Pipeline chaining
print(result.to_text())         # same as result.summary
```

### `SummarizeStyle`

Type alias for the valid style literals:

```python
from aiforge.modules.summarize.schemas import SummarizeStyle

# SummarizeStyle = Literal["concise", "detailed", "bullet_points", "executive"]
```

---

## Strategy Selection

The summarizer automatically selects between `stuff` and `map_reduce` based on token count vs. the model's context window:

```
Input tokens > (context_window × 0.8)?
  ├── No  → "stuff": send entire text in one LLM call
  └── Yes → "map_reduce":
              1. Split text into overlapping chunks
              2. Summarize each chunk independently (map phase)
              3. Combine chunk summaries into final summary (reduce phase)
```

### Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `_STUFF_THRESHOLD_RATIO` | `0.8` | Auto-select `map_reduce` when tokens exceed 80% of context window |
| `_CHUNK_SIZE_RATIO` | `0.6` | Each chunk targets 60% of the context window |
| `_CHUNK_OVERLAP` | `200` | Characters of overlap between adjacent chunks |

### Token Counting

Token counts are estimated using **tiktoken** when the model is recognized (OpenAI models). For unrecognized models (Anthropic, Ollama, etc.), a `len(text) // 4` heuristic is used.

### Context Window Awareness

The strategy selector reads the model's max token limit from AIForge's built-in pricing registry via `get_context_window()`. Examples:
- `gpt-4o` → 128K tokens → stuff threshold at ~102K tokens
- `gpt-4o-mini` → 128K tokens → stuff threshold at ~102K tokens
- `claude-sonnet-5-5` → 1M tokens → stuff threshold at ~800K tokens

### Force a Strategy

Override auto-selection by passing `strategy` explicitly:

```python
# Force stuff (even for long text — may fail if text exceeds context)
result = s.summarize(text, strategy="stuff")

# Force map_reduce (even for short text)
result = s.summarize(text, strategy="map_reduce")
```

---

## Map-Reduce Pipeline

When `map_reduce` is selected:

### 1. Split (Chunking)

Text is split into overlapping chunks using the core `TextSplitter` (with character-based fallback). Chunk size = `context_window × 0.6`, overlap = 200 characters.

### 2. Map Phase

Each chunk is sent to the LLM with the `SUMMARIZE_MAP` prompt:

```
Summarize this section concisely, preserving all key facts:
<chunk>{chunk}</chunk>
Section Summary:
```

- Sync: chunks are processed sequentially
- Async (`asummarize`): chunks are processed **concurrently** via `asyncio.gather()`

### 3. Reduce Phase

All chunk summaries are concatenated and sent to the LLM with the `SUMMARIZE_REDUCE` prompt:

```
Combine these section summaries into one coherent summary of {max_length} words or fewer.
Remove any redundancy.
<summaries>{summaries}</summaries>
Combined Summary:
```

### Cost Aggregation

The total `cost`, `input_tokens`, and `output_tokens` in `SummarizeResponse` are the **sum** across all map calls plus the reduce call.

---

## Prompt Templates

### Style Prompts

Each style uses a different user prompt template. All receive `{text}` and `{max_length}` placeholders:

| Style | Focus |
|-------|-------|
| **concise** | *"Key points and main conclusions"* |
| **detailed** | *"All major points, supporting evidence, and conclusions"* |
| **bullet_points** | *"One key point per bullet"* |
| **executive** | *"Decisions, actions, and outcomes"* |

### System Prompt

Default system prompt:

> *"You are an expert text summariser. Produce clear, accurate summaries that preserve the key facts and tone of the original text. Respond ONLY with the summary — no preamble."*

---

## Prompt Resolution

System prompts are resolved in this precedence order:

1. **Per-call override:** `s.summarize(text, system_prompt="...")`
2. **Constructor override:** `Summarizer(system_prompt="...")`
3. **Config file:** `CUSTOM_PROMPTS.summarize.system` in `aiforge.yaml`
4. **Built-in default:** The `SUMMARIZE_SYSTEM` constant (see above)

```python
# Per-call override
result = s.summarize(text, system_prompt="Summarize in the style of a news reporter.")

# Constructor override (applies to all calls)
s = Summarizer(system_prompt="Always include numerical data in summaries.")
```

---

## Instance Properties

After constructing a `Summarizer`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `s.style` | `SummarizeStyle` | Configured summary style. |
| `s.max_length` | `int` | Target summary length in words. |
| `s.temperature` | `float` | LLM sampling temperature. |
| `s.settings` | `AIForgeSettings` | Resolved configuration. |
| `s.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Basic Summarization

```python
from aiforge import Summarizer

s = Summarizer(style="concise", max_length=200)

article = """
Climate scientists have published new findings showing that global temperatures
rose by 1.5°C compared to pre-industrial levels for the first time in 2025.
The study, conducted by an international team of researchers, analyzed data
from over 10,000 weather stations worldwide. Key findings include accelerated
ice melt in Greenland, rising sea levels in coastal cities, and increased
frequency of extreme weather events...
"""

result = s.summarize(article)
print(result.summary)
# "Global temperatures reached 1.5°C above pre-industrial levels in 2025,
#  according to a study analyzing 10,000+ weather stations. Key impacts include
#  accelerated Greenland ice melt, coastal sea level rise, and more extreme weather."
print(result.strategy_used)     # "stuff"
print(f"Cost: ${result.cost:.4f}")
```

### Bullet Point Summary

```python
s = Summarizer(style="bullet_points", max_length=300)
result = s.summarize(meeting_notes)
print(result.summary)
# • Team approved the Q2 budget of $2.4M
# • New hiring plan: 5 engineers, 2 designers by June
# • Product launch moved from March to April 15
# • Action: Sarah to finalize vendor contracts by Friday
```

### Executive Summary

```python
s = Summarizer(style="executive", max_length=150)
result = s.summarize(quarterly_report)
print(result.summary)
# "Revenue grew 12% YoY to $45M. Key decisions: expand into APAC market,
#  increase R&D budget by 20%. Action items: hire regional VP by Q3,
#  complete compliance audit by end of month."
```

### Long Document (Auto Map-Reduce)

```python
s = Summarizer(style="detailed", max_length=1000)

# Read a large document
with open("annual_report_2025.txt") as f:
    long_text = f.read()  # 200,000 words

result = s.summarize(long_text)
print(result.strategy_used)     # "map_reduce"
print(result.chunks_processed)  # 8
print(result.cost)              # 0.032 (aggregated across 8 map + 1 reduce calls)
print(result.latency_ms)        # 12500 (sequential map calls)
```

### Async Map-Reduce (Faster)

```python
import asyncio
from aiforge import Summarizer

async def main():
    s = Summarizer(style="concise", max_length=500)

    with open("large_document.txt") as f:
        text = f.read()

    # Async map_reduce runs chunks concurrently
    result = await s.asummarize(text)
    print(result.summary)
    print(result.strategy_used)     # "map_reduce"
    print(result.latency_ms)        # Much faster than sync — parallel chunks

asyncio.run(main())
```

### Pipeline Integration

```python
from aiforge import Summarizer

s = Summarizer(style="concise")

# Pipeline-compatible interface
result = s.run("Long article text...")
next_input = result.to_text()  # summary string for downstream module
```

### Batch Summarization

```python
s = Summarizer(style="bullet_points", max_length=100)

articles = [
    "First article about technology...",
    "Second article about finance...",
    "Third article about healthcare...",
]

batch = s.run_batch(articles, concurrency=3)
for i, r in enumerate(batch.results):
    print(f"Article {i+1} ({r.strategy_used}):")
    print(r.summary)
    print()

print(f"Total cost: ${batch.total_cost:.4f}")
```

### Custom System Prompt

```python
s = Summarizer(
    style="concise",
    system_prompt="You are a financial analyst. Focus on numbers, percentages, and fiscal metrics.",
)

result = s.summarize(annual_report)
# Summary will emphasize financial data
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
MODEL: "gpt-4o-mini"
CUSTOM_PROMPTS:
  summarize:
    system: "Summarize in plain English suitable for a general audience."
```

```python
s = Summarizer()  # picks up aiforge.yaml automatically
result = s.summarize(text)
print(result.model)  # "gpt-4o-mini"
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Invalid `style`, `max_length < 1`, `temperature` out of range, empty `text`, invalid `strategy` | `Summarizer(style="unknown")` |
| `ProviderError` | LLM provider returns an error | Invalid API key |
| `RateLimitError` | Provider rate limit exceeded | Map-reduce with many chunks |
| `APITimeoutError` | Provider request timed out | Very long text |

```python
from aiforge.core.types import ProviderError

try:
    result = s.summarize("Some text")
except ProviderError as e:
    print(f"LLM call failed: {e}")

try:
    s = Summarizer(style="invalid")
except ValueError as e:
    print(f"Config error: {e}")
    # "style must be one of ['concise', 'detailed', 'bullet_points', 'executive']"
```

---

## File Structure

```
aiforge/modules/summarize/
├── __init__.py        # Exports: Summarizer, SummarizeResponse
├── module.py          # Summarizer class (stuff + map_reduce strategies)
├── prompts.py         # Prompt templates (SUMMARIZE_SYSTEM, 4 style prompts, MAP, REDUCE)
├── schemas.py         # Pydantic models (SummarizeResponse, SummarizeStyle)
└── widget.py          # Streamlit playground widget (style selector, strategy picker, metrics)
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic summarization example
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /summarize` endpoint
- [Configuration](../configuration.md) — Full settings reference
- [Playground](../playground.md) — Interactive summarization tab
