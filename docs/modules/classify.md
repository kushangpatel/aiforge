# Module: Classify

`aiforge.classify` — Text Classification

---

## Purpose

Classify text into custom categories. Supports zero-shot (just labels) and few-shot (labels + examples).

---

## Public API

```python
from aiforge import Classifier

# Zero-shot (just labels)
c = Classifier(labels=["bug", "feature_request", "question", "praise"])
result = c.run("The login button doesn't work on mobile")
print(result.label)            # "bug"
print(result.confidence)       # 0.94
print(result.all_scores)       # {"bug": 0.94, "feature_request": 0.03, ...}

# Few-shot (labels + examples)
c = Classifier(
    labels=["urgent", "normal", "low"],
    examples=[
        {"text": "Server is down, all users affected", "label": "urgent"},
        {"text": "Could you add dark mode?", "label": "low"},
        {"text": "Invoice not showing correct total", "label": "normal"},
    ]
)
result = c.run("Payment processing is failing for all customers")
print(result.label)            # "urgent"

# Batch classification
results = c.run_batch(["text1", "text2", "text3"])
```

### Async Support

```python
result = await c.arun(text)
```

---

## Batch Semantics

`run_batch()` on all modules follows these rules:

| Aspect | Behavior |
|--------|----------|
| **Execution** | N separate LLM calls (one per item), NOT a single mega-prompt. This ensures consistent per-item quality and proper cost attribution. |
| **Concurrency** | Up to 10 items processed concurrently via `asyncio.gather()` (respects rate limiter). Configurable: `run_batch(items, concurrency=5)`. |
| **Partial failure** | Returns `BatchResult` with `.succeeded: list[T]` and `.failed: list[BatchError]`. Does NOT abort on single-item failure. |
| **Batch size limit** | Default max 100 items per call. Raise `ValidationError` if exceeded. Override: `run_batch(items, max_batch=500)`. |
| **Cost tracking** | Aggregate cost across all items available via `batch_result.total_cost`. Individual costs on each result. |

---

## Schemas

### Request

```python
class ClassifyRequest(BaseModel):
    text: str
    labels: list[str]
    examples: list[dict] | None = None  # few-shot examples
    multi_label: bool = False           # allow multiple labels
```

### Response

```python
class ClassifyResponse(BaseModel):
    label: str | list[str]              # single or multi-label
    confidence: float
    all_scores: dict[str, float]        # scores per label
    cost: float
    trace_id: str

    def to_text(self) -> str:
        return self.label if isinstance(self.label, str) else ", ".join(self.label)
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-CLF-001 | Zero-shot classification | P0 | Classify with just labels, no examples |
| FR-CLF-002 | Few-shot classification | P1 | Improve accuracy with labeled examples |
| FR-CLF-003 | Multi-label | P1 | Return multiple labels when applicable |
| FR-CLF-004 | Confidence scores | P0 | Return per-label confidence |
| FR-CLF-005 | Batch classification | P1 | Classify multiple texts efficiently |

---

## File Structure

```
aiforge/modules/classify/
├── __init__.py
├── module.py          # Classifier class
├── prompts.py         # Prompt templates
├── schemas.py         # ClassifyRequest, ClassifyResponse
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic classification example
- [Pipelines](../pipelines.md) — Use as routing step in pipelines
- [API Server](../api-server.md) — `POST /classify` endpoint
