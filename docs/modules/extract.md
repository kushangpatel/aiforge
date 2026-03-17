# Module: Extract

`aiforge.extract` — Structured Data Extraction

---

## Purpose

Extract structured data from unstructured text. Give it a Pydantic schema, get back validated, typed data.

---

## Public API

```python
from aiforge import Extractor
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    line_items: list[dict]
    date: str

e = Extractor(output_schema=Invoice)

result = e.run("Invoice #4521 from Acme Corp. Dated Jan 15, 2026. "
               "Widget A x10 @ $5.00 = $50.00. Widget B x3 @ $12.00 = $36.00. "
               "Total: $86.00 USD.")

print(result.data)
# Invoice(vendor="Acme Corp", total=86.0, currency="USD",
#         line_items=[{"item": "Widget A", "qty": 10, ...}, ...],
#         date="2026-01-15")
print(result.data.total)       # 86.0  ← typed, validated
print(result.raw_json)         # raw JSON string from LLM
print(result.cost)             # 0.001
```

### Async Support

```python
result = await e.arun(text)
```

---

## How It Works

```
1. User provides: text + Pydantic model (or JSON schema)
2. AIForge generates a prompt: "Extract the following fields: {schema_description}"
3. LLM returns JSON
4. AIForge validates against schema (Pydantic)
5. If validation fails → retry with error feedback (up to 3 attempts)
6. Return typed object
```

---

## Schemas

### Request

```python
class ExtractRequest(BaseModel):
    text: str
    output_schema: dict                 # JSON Schema
    examples: list[dict] | None = None  # few-shot examples
```

### Response

```python
class ExtractResponse(BaseModel):
    data: dict                          # validated structured data
    raw_json: str                       # raw LLM output
    validation_passed: bool
    retry_count: int
    cost: float
    trace_id: str

    def to_text(self) -> str:
        return self.data.model_dump_json() if hasattr(self.data, 'model_dump_json') else str(self.data)
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-EXT-001 | Schema-based extraction | P0 | Accept Pydantic model or JSON schema, return validated data |
| FR-EXT-002 | Auto-retry on validation failure | P0 | If LLM output fails schema validation, retry with error context |
| FR-EXT-003 | Few-shot examples | P1 | Accept examples to improve extraction quality |
| FR-EXT-004 | Batch extraction | P1 | Extract from multiple texts in one call |

---

## File Structure

```
aiforge/modules/extract/
├── __init__.py
├── module.py          # Extractor class
├── prompts.py         # Prompt templates
├── schemas.py         # ExtractRequest, ExtractResponse
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic extraction example
- [Pipelines](../pipelines.md) — Chain with other modules
- [API Server](../api-server.md) — `POST /extract` endpoint
