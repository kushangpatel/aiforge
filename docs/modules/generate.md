# Module: Generate

`aiforge.generate` — Template-Based Content Generation

---

## Purpose

Generate content from templates with variable substitution and guardrails. For emails, reports, descriptions, marketing copy — anything with a repeatable structure.

---

## Public API

```python
from aiforge import Generator

# Template-based
g = Generator(
    template="""Write a {tone} product description for:
    Product: {product_name}
    Features: {features}
    Target audience: {audience}
    Max length: {max_words} words.""",
    guardrails=["no_competitor_mentions", "no_pricing"]
)

result = g.run(
    product_name="AIForge",
    features=["plug-and-play AI", "8 modules", "provider-agnostic"],
    audience="developers",
    tone="professional",
    max_words=150,
)
print(result.text)             # Generated description
print(result.guardrail_flags)  # [] (empty = no violations)

# Without template (freeform with guardrails)
g = Generator(guardrails=["no_pii", "professional_tone"])
result = g.run(prompt="Write a follow-up email for a customer who complained about...")
```

### Async Support

```python
result = await g.arun(product_name="AIForge", tone="professional")

# Streaming
async for chunk in g.arun_stream(prompt="Generate a blog post about..."):
    print(chunk, end="")
```

---

## Guardrails

Guardrails are checked **after generation** (post-processing). If a guardrail is violated, the Generator retries with the violation fed back into the prompt (up to 2 retries).

| Guardrail | Detection Method | Performance Cost |
|-----------|-----------------|------------------|
| `no_pii` | Delegates to `Moderator.check(checks=["pii"])` — regex-first | ~5ms (regex only, no LLM call) |
| `no_competitor_mentions` | Regex match against configurable competitor list | ~1ms |
| `professional_tone` | LLM-based check (single classification call) | ~200ms + ~$0.0001 |
| `no_pricing` | Regex for currency patterns + LLM fallback | ~5ms or ~200ms |
| `no_hallucination` | LLM-based: compare output against provided context | ~500ms + ~$0.001 |
| Custom guardrail | User-defined function `(text: str) -> GuardrailResult` | Varies |

### Custom Guardrails

```python
from aiforge.guardrails import register_guardrail, GuardrailResult

@register_guardrail("max_sentences")
def check_max_sentences(text: str, max_n: int = 5) -> GuardrailResult:
    count = text.count(".") + text.count("!") + text.count("?")
    return GuardrailResult(passed=count <= max_n, message=f"Output has {count} sentences (max {max_n})")

g = Generator(guardrails=["max_sentences"])
```

---

## Schemas

### Request

```python
class GenerateRequest(BaseModel):
    template: str | None = None
    prompt: str | None = None
    variables: dict = {}
    guardrails: list[str] = []
    temperature: float = 0.7
```

### Response

```python
class GenerateResponse(BaseModel):
    text: str
    guardrail_flags: list[dict]
    retry_count: int
    cost: float
    trace_id: str

    def to_text(self) -> str:
        return self.text
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-GEN-001 | Template-based generation | P0 | Variable substitution + LLM generation |
| FR-GEN-002 | Freeform generation | P0 | Direct prompt with guardrails |
| FR-GEN-003 | Guardrails | P1 | Check output against rules (no PII, no competitors, etc.) |
| FR-GEN-004 | Streaming | P1 | Stream generated tokens |

---

## File Structure

```
aiforge/modules/generate/
├── __init__.py
├── module.py          # Generator class
├── prompts.py         # Prompt templates
├── schemas.py         # GenerateRequest, GenerateResponse
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic generation example
- [Pipelines](../pipelines.md) — Chain with other modules
- [Moderate](moderate.md) — Guardrail delegation to Moderator
- [API Server](../api-server.md) — `POST /generate` endpoint
