# Module: Extract

`aiforge.modules.extract` — Structured data extraction from unstructured text. Provide a Pydantic model or JSON schema, get back validated, typed data. Includes automatic retry with error feedback when validation fails.

---

## Installation

The extract module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge
```

---

## Import

```python
from aiforge import Extractor
from aiforge import ExtractResponse, ExtractExample  # types
```

---

## Quick Start

```python
from pydantic import BaseModel
from aiforge import Extractor

class Person(BaseModel):
    name: str
    age: int
    occupation: str

e = Extractor(output_schema=Person)
result = e.extract("John Smith is a 34-year-old software engineer at Google.")

print(result.data)               # {"name": "John Smith", "age": 34, "occupation": "software engineer"}
print(result.validation_passed)  # True
print(result.retry_count)        # 0
```

---

## Constructor

```python
Extractor(
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    examples: list[ExtractExample] | None = None,
    max_retries: int = 3,
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
| `output_schema` | `type[BaseModel] \| dict[str, Any] \| None` | `None` | The target schema for extraction. Accepts a Pydantic `BaseModel` subclass or a raw JSON schema dict. Can be overridden per call. Must be provided either at construction or call time. |
| `examples` | `list[ExtractExample] \| None` | `None` | Few-shot examples to guide extraction. Each example pairs input text with expected output. See [Few-Shot Examples](#few-shot-examples). |
| `max_retries` | `int` | `3` | Maximum retry attempts when LLM output fails schema validation. Minimum enforced to 1. On each retry, the validation error is fed back to the LLM for self-correction. |
| `temperature` | `float` | `0.0` | LLM temperature. Must be in range `[0.0, 2.0]`. Lower values (0.0) produce more deterministic, consistent extractions. |
| `system_prompt` | `str \| None` | `None` | Override the default system prompt. See [Prompt Resolution](#prompt-resolution). |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`. |

### Schema Types

The extractor accepts two types of schemas:

**Pydantic Model** (recommended) — gives you typed validation and clear field definitions:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    line_items: list[dict]
    date: str

e = Extractor(output_schema=Invoice)
```

**Raw JSON Schema dict** — useful when schemas are dynamic or loaded from config:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

e = Extractor(output_schema=schema)
```

### Examples

```python
from aiforge import Extractor, ExtractExample
from pydantic import BaseModel

# Minimal — schema only
e = Extractor(output_schema=Person)

# With few-shot examples for better accuracy
e = Extractor(
    output_schema=Person,
    examples=[
        ExtractExample(
            input="Alice is 28 and works as a data scientist.",
            output={"name": "Alice", "age": 28, "occupation": "data scientist"},
        ),
    ],
)

# With retries and specific model
e = Extractor(
    output_schema=Invoice,
    max_retries=5,
    temperature=0.0,
    MODEL="gpt-4o",
)

# Schema provided at call time instead of construction
e = Extractor()
result = e.extract("John is 30", output_schema=Person)

# Quality-based model routing
e = Extractor(output_schema=Person, quality="high")
```

---

## Methods

### `extract(text, output_schema=None, examples=None, *, system_prompt=None) -> ExtractResponse`

Extract structured data from text. Validates output against the schema and retries with error feedback on failure.

```python
def extract(
    self,
    text: str,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    examples: list[ExtractExample] | None = None,
    *,
    system_prompt: str | None = None,
) -> ExtractResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to extract from. Must not be empty or whitespace-only (raises `ValueError`). |
| `output_schema` | `type[BaseModel] \| dict \| None` | `None` | Per-call schema override. If `None`, uses the instance schema. Raises `ValueError` if neither is set. |
| `examples` | `list[ExtractExample] \| None` | `None` | Per-call examples override. If `None`, uses instance examples. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`ExtractResponse`](#extractresponse)

```python
from pydantic import BaseModel
from aiforge import Extractor

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

e = Extractor(output_schema=ContactInfo)

result = e.extract("Reach out to Jane Doe at jane@example.com or call 555-0123.")
print(result.data)
# {"name": "Jane Doe", "email": "jane@example.com", "phone": "555-0123"}
print(result.validation_passed)  # True
print(result.retry_count)        # 0
print(result.cost)               # 0.0001
print(result.model)              # "gpt-4o-mini"
```

---

### `aextract(text, output_schema=None, examples=None, *, system_prompt=None) -> ExtractResponse`

Async version of `extract()`.

```python
async def aextract(
    self,
    text: str,
    output_schema: type[BaseModel] | dict[str, Any] | None = None,
    examples: list[ExtractExample] | None = None,
    *,
    system_prompt: str | None = None,
) -> ExtractResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to extract from. |
| `output_schema` | `type[BaseModel] \| dict \| None` | `None` | Per-call schema override. |
| `examples` | `list[ExtractExample] \| None` | `None` | Per-call examples override. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`ExtractResponse`](#extractresponse)

```python
result = await e.aextract("Bob is 42 years old and is a chef.")
print(result.data)  # {"name": "Bob", "age": 42, "occupation": "chef"}
```

---

### `run(text, **kwargs) -> ExtractResponse`

BaseModule interface method. Delegates to `extract()`. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> ExtractResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to extract from. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`ExtractResponse`](#extractresponse)

```python
result = e.run("Alice is 28 and works as a data scientist.")
```

---

### `arun(text, **kwargs) -> ExtractResponse`

Async version of `run()`. Delegates to `aextract()`.

```python
async def arun(self, text: str, **kwargs) -> ExtractResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Extract from multiple texts concurrently. Inherited from `BaseModule`.

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
| `items` | `list[str]` | *(required)* | List of texts to extract from. |
| `concurrency` | `int` | `10` | Max concurrent LLM requests. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

```python
texts = [
    "John Smith, 34, software engineer at Google.",
    "Jane Doe, age 28, data scientist at Meta.",
    "Bob Wilson is a 45-year-old chef in NYC.",
]

batch = e.run_batch(texts, concurrency=3)

for r in batch.results:
    print(f"  {r.data['name']} — {r.data['occupation']} (valid: {r.validation_passed})")

print(f"Total cost: ${batch.total_cost:.4f}")
```

---

## Response Types

### `ExtractResponse`

Returned by `extract()`, `aextract()`, `run()`, and `arun()`.

```python
from aiforge import ExtractResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `data` | `dict[str, Any]` | Validated extracted data as a dictionary. Empty `{}` if validation failed after all retries. |
| `raw_json` | `str` | Raw JSON string returned by the LLM before validation. Useful for debugging. |
| `validation_passed` | `bool` | `True` if the LLM output passed schema validation. `False` if all retries were exhausted. |
| `retry_count` | `int` | Number of retry attempts (0 = succeeded on first attempt, 1 = needed one retry, etc.). |
| `cost` | `float` | Total cost in USD across all attempts (including retries). |
| `input_tokens` | `int` | Total input tokens consumed across all attempts. |
| `output_tokens` | `int` | Total output tokens generated across all attempts. |
| `model` | `str` | Model used (e.g. `"gpt-4o-mini"`). |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds (including retries). |

```python
result = e.extract("Jane is 28 and works at Meta.")

print(result.data)               # {"name": "Jane", "age": 28, "occupation": "..."}
print(result.raw_json)           # '{"name": "Jane", "age": 28, ...}'
print(result.validation_passed)  # True
print(result.retry_count)        # 0
print(result.cost)               # 0.00012
print(result.model)              # "gpt-4o-mini"
print(result.latency_ms)         # 412.3

# to_text() returns JSON string (used for pipeline chaining)
print(result.to_text())          # '{"name": "Jane", "age": 28, ...}'
# Returns "{}" if data is empty
```

---

### `ExtractExample`

A single few-shot example pairing input text with expected output.

```python
from aiforge import ExtractExample
```

| Field | Type | Description |
|-------|------|-------------|
| `input` | `str` | Example source text. Must not be empty. |
| `output` | `dict[str, Any]` | Expected extracted data for this text. |

```python
example = ExtractExample(
    input="Alice is 28 and works as a data scientist.",
    output={"name": "Alice", "age": 28, "occupation": "data scientist"},
)
```

---

## Retry Mechanism

The extractor automatically retries when LLM output fails schema validation. On each retry, the validation error message is fed back to the LLM so it can self-correct.

```
Attempt 1: LLM → JSON → validate → ✗ ("age must be integer, got string")
Attempt 2: LLM (with error feedback) → JSON → validate → ✓
```

```python
e = Extractor(output_schema=Person, max_retries=5)

result = e.extract("Bob is thirty years old")
print(result.retry_count)        # possibly 1 (LLM initially wrote "thirty" instead of 30)
print(result.validation_passed)  # True (self-corrected on retry)
print(result.cost)               # includes cost of all attempts
```

**Retry behavior:**
- `max_retries` controls the maximum number of attempts (minimum enforced to 1)
- Cost, tokens, and latency are accumulated across all attempts
- If all retries are exhausted, returns `ExtractResponse` with `validation_passed=False` and `data={}`
- Each retry prompt includes the previous invalid output and the specific validation error

---

## Few-Shot Examples

Providing examples improves extraction accuracy, especially for complex schemas or ambiguous text.

```python
from aiforge import Extractor, ExtractExample
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    date: str
    location: str
    attendees: int | None = None

e = Extractor(
    output_schema=Event,
    examples=[
        ExtractExample(
            input="The AI Summit will be held in San Francisco on June 15, 2026 with 500 expected attendees.",
            output={"name": "AI Summit", "date": "2026-06-15", "location": "San Francisco", "attendees": 500},
        ),
        ExtractExample(
            input="Join us for TechConf in Austin next March.",
            output={"name": "TechConf", "date": "March", "location": "Austin", "attendees": None},
        ),
    ],
)

result = e.extract("PyConf 2026 is happening in Chicago on Sept 20 with about 1200 people.")
print(result.data)
# {"name": "PyConf 2026", "date": "2026-09-20", "location": "Chicago", "attendees": 1200}
```

**Rules:**
- Example text and output are escaped to prevent prompt injection
- Each example's `input` must be non-empty
- More examples improve accuracy but increase input token count and cost
- Examples can be provided at construction or overridden per call

```python
# Per-call example override
result = e.extract(
    "Meeting with Bob at 3pm",
    examples=[
        ExtractExample(
            input="Call with Alice at noon",
            output={"name": "Call with Alice", "date": "noon", "location": "", "attendees": None},
        ),
    ],
)
```

---

## Per-Call Schema Override

The schema can be provided or changed at call time, allowing one `Extractor` instance to handle multiple schema types:

```python
from pydantic import BaseModel
from aiforge import Extractor

class Person(BaseModel):
    name: str
    age: int

class Company(BaseModel):
    name: str
    industry: str
    employees: int | None = None

e = Extractor()  # no schema at construction

# Extract a Person
person = e.extract("John is 34 years old.", output_schema=Person)
print(person.data)  # {"name": "John", "age": 34}

# Extract a Company from different text
company = e.extract(
    "Acme Corp is a tech company with 500 employees.",
    output_schema=Company,
)
print(company.data)  # {"name": "Acme Corp", "industry": "tech", "employees": 500}
```

---

## Using Raw JSON Schema

When you don't want to define a Pydantic model, pass a JSON schema dict:

```python
from aiforge import Extractor

schema = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string"},
        "price": {"type": "number"},
        "in_stock": {"type": "boolean"},
    },
    "required": ["product_name", "price"],
}

e = Extractor(output_schema=schema)
result = e.extract("The Widget Pro is $49.99 and currently available.")

print(result.data)
# {"product_name": "Widget Pro", "price": 49.99, "in_stock": True}
print(result.validation_passed)  # True
```

When using a raw dict schema, validation is performed via `jsonschema.validate()` instead of Pydantic.

---

## Prompt Resolution

System prompts are resolved in this precedence order (highest to lowest):

1. **Per-call override:** `e.extract(text, system_prompt="...")`
2. **Constructor override:** `Extractor(output_schema=..., system_prompt="...")`
3. **Config file:** `CUSTOM_PROMPTS.extract.system` in `aiforge.yaml`
4. **Built-in default:** *"You are a structured-data extraction engine. Given source text and a JSON Schema, extract the requested fields. Always respond ONLY with valid JSON — no markdown, no extra prose."*

---

## Instance Properties

After constructing an `Extractor`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `e.output_schema` | `type[BaseModel] \| dict \| None` | Configured output schema. |
| `e.examples` | `list[ExtractExample]` | Registered few-shot examples (empty list if none). |
| `e.max_retries` | `int` | Maximum retry attempts. |
| `e.temperature` | `float` | LLM temperature setting. |
| `e.settings` | `AIForgeSettings` | Resolved configuration. |
| `e.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Invoice Extraction

```python
from pydantic import BaseModel
from aiforge import Extractor

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    line_items: list[LineItem]
    subtotal: float
    tax: float | None = None
    total: float
    currency: str

e = Extractor(output_schema=Invoice, max_retries=5)

result = e.extract("""
Invoice #4521 from Acme Corp. Dated Jan 15, 2026.
Widget A x10 @ $5.00 = $50.00
Widget B x3 @ $12.00 = $36.00
Subtotal: $86.00  Tax: $6.88  Total: $92.88 USD
""")

print(result.data["vendor"])           # "Acme Corp"
print(result.data["total"])            # 92.88
print(result.data["line_items"][0])    # {"description": "Widget A", "quantity": 10, ...}
print(result.validation_passed)        # True
```

### Resume Parsing

```python
from pydantic import BaseModel
from aiforge import Extractor

class Experience(BaseModel):
    company: str
    role: str
    duration: str

class Resume(BaseModel):
    name: str
    email: str | None = None
    skills: list[str]
    experience: list[Experience]

e = Extractor(output_schema=Resume)

result = e.extract("""
John Smith | john@example.com
Skills: Python, React, AWS, Docker

Experience:
- Software Engineer at Google (2022-2024)
- Junior Developer at Startup Inc (2020-2022)
""")

print(result.data["name"])         # "John Smith"
print(result.data["skills"])       # ["Python", "React", "AWS", "Docker"]
print(len(result.data["experience"]))  # 2
```

### Batch Extraction

```python
from pydantic import BaseModel
from aiforge import Extractor

class Product(BaseModel):
    name: str
    price: float
    category: str

e = Extractor(output_schema=Product, quality="low")

descriptions = [
    "The UltraWidget costs $29.99 and falls under electronics.",
    "Organic Green Tea, priced at $12.50 — beverages section.",
    "Running Shoes Pro for $89.99 in the sports category.",
]

batch = e.run_batch(descriptions, concurrency=3)

for r in batch.results:
    p = r.data
    print(f"  {p['name']}: ${p['price']:.2f} ({p['category']})")

print(f"\nTotal cost: ${batch.total_cost:.4f}")
print(f"All valid: {all(r.validation_passed for r in batch.results)}")
```

### Pipeline Chaining

The `to_text()` method returns JSON, making `Extractor` composable in pipelines:

```python
from aiforge import Extractor, Summarizer
import json

# Step 1: Summarize a long document
summarizer = Summarizer()
summary = summarizer.summarize(long_document)

# Step 2: Extract key entities from the summary
e = Extractor(output_schema=KeyEntities)
extracted = e.extract(summary.to_text())

# to_text() returns JSON for downstream steps
next_input = extracted.to_text()  # '{"entities": [...]}'
```

### Handling Validation Failures

```python
result = e.extract("Some ambiguous text that might not extract cleanly")

if result.validation_passed:
    print(f"Extracted: {result.data}")
    print(f"Retries needed: {result.retry_count}")
else:
    print(f"Extraction failed after {result.retry_count} retries")
    print(f"Last LLM output: {result.raw_json}")
    print(f"Total cost (including retries): ${result.cost:.4f}")
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
MODEL: "gpt-4o-mini"
TEMPERATURE: 0.0
```

```python
# Picks up aiforge.yaml automatically
e = Extractor(output_schema=Person)
result = e.extract("Alice is 28.")
print(result.model)  # "gpt-4o-mini"
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Empty text, no schema provided (at construction or call time), temperature out of range | `e.extract("")` |
| `TypeError` | Schema is not a Pydantic model or dict | `Extractor(output_schema="invalid")` |
| `ProviderError` | LLM provider returns an error | API key invalid |
| `RateLimitError` | Provider rate limit exceeded | Too many requests |
| `APITimeoutError` | Provider request timed out | Slow network |

```python
from aiforge.core.types import ProviderError

try:
    result = e.extract("Some text")
except ValueError as e:
    print(f"Validation error: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

**Graceful degradation:** If the LLM returns malformed JSON or output that fails validation after all retries, the extractor does **not** raise. It returns an `ExtractResponse` with `validation_passed=False` and `data={}`.

---

## File Structure

```
aiforge/modules/extract/
├── __init__.py        # Exports: Extractor, ExtractResponse, ExtractExample
├── module.py          # Extractor class (constructor, extract/aextract, retry loop, validation)
├── prompts.py         # Prompt templates (EXTRACT_SYSTEM, EXTRACT_USER, EXTRACT_RETRY)
├── schemas.py         # Pydantic models (ExtractResponse, ExtractExample)
└── widget.py          # Streamlit playground widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Getting started with AIForge
- [Pipelines](../pipelines.md) — Chain extraction with other modules
- [API Server](../api-server.md) — `POST /extract` endpoint
- [Playground](../playground.md) — Interactive extraction tab
- [Configuration](../configuration.md) — Full settings reference
