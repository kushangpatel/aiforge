# Pipelines

The killer feature: **chain modules together** into composable pipelines.

---

## Sequential Pipeline

```python
from aiforge import Pipeline, Extractor, Classifier, Summarizer

# Define pipeline
pipe = Pipeline([
    ("extract", Extractor(output_schema=TicketSchema)),
    ("classify", Classifier(labels=["critical", "high", "medium", "low"])),
    ("summarize", Summarizer(style="concise")),
])

# Run pipeline — output of each step feeds into the next
result = pipe.run("Long customer complaint email text here...")

print(result["extract"].data)      # Structured ticket data
print(result["classify"].label)    # "critical"
print(result["summarize"].summary) # One-line summary
```

---

## Conditional Routing

Route to different modules based on classification output:

```python
pipe = Pipeline([
    ("classify", Classifier(labels=["support", "sales", "spam"])),
    ("route", {
        "support": Summarizer(style="bullet_points"),
        "sales": Generator(template="Draft a sales response..."),
        "spam": None,  # drop
    }),
])
```

---

## Pipeline Data Flow Contract

When modules are chained in a pipeline, data flows between them using these rules:

| Source Module Output | Next Module Input | Transformation |
|---------------------|------------------|----------------|
| `SummarizeResponse.summary` | `str` (text) | Pass `.summary` as text |
| `QAResponse.text` | `str` (text) | Pass `.text` as text |
| `ExtractResponse.data` | `str` (text) | Call `model.model_dump_json()` → JSON string |
| `ClassifyResponse.label` | `str` (text) | Pass `.label` for routing; pass original text to next module |
| `SearchResponse.hits` | `str` (text) | Join `[hit.text for hit in hits]` with `\n\n` |
| `GenerateResponse.text` | `str` (text) | Pass `.text` as text |
| `ModerateResponse` | Pass-through | If `flagged=True` and pipeline `halt_on_flag=True`, raise `PipelineError` |

### The `to_text()` Protocol

Every module response implements a `.to_text() -> str` method that provides the canonical text representation for downstream consumption. Custom modules must implement this method.

```python
class SummarizeResponse(BaseModel):
    # ... fields ...
    def to_text(self) -> str:
        return self.summary

class ExtractResponse(BaseModel):
    # ... fields ...
    def to_text(self) -> str:
        return self.data.model_dump_json() if hasattr(self.data, 'model_dump_json') else str(self.data)
```

The `Pipeline` class calls `step_result.to_text()` to get the input for the next step. If a module receives a type it doesn't expect, a `PipelineError` is raised with a clear message indicating which step produced the incompatible output.

---

## Pipeline with Moderator Guardrail

```python
from aiforge import Pipeline, Moderator, Summarizer, Generator

pipe = Pipeline([
    ("moderate_input", Moderator(checks=["prompt_injection"])),
    ("summarize", Summarizer(style="concise")),
    ("generate", Generator(template="Write a response based on: {text}")),
    ("moderate_output", Moderator(checks=["pii"])),
])
```

---

## Custom Module in Pipeline

```python
from aiforge.core.module import BaseModule, ModuleResponse
from aiforge import Pipeline, Summarizer

class SentimentAnalyzer(BaseModule):
    """Custom module — follows the same interface as built-in modules."""

    def run(self, text: str) -> ModuleResponse:
        result = self._call_llm(f"Classify sentiment: {text}")
        return SentimentResponse(sentiment=result.text.strip(), cost=result.cost)

    def to_text(self, response: ModuleResponse) -> str:
        return response.sentiment

# Use in pipelines immediately
pipe = Pipeline([
    ("sentiment", SentimentAnalyzer()),
    ("summarize", Summarizer()),
])
```

---

## Pipeline Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-PIP-001 | Sequential pipeline | P0 | Chain modules in sequence |
| FR-PIP-002 | Conditional routing | P1 | Route to different modules based on classify output |
| FR-PIP-003 | Aggregate results | P0 | Return results from all pipeline steps |

---

## Error Handling

Pipeline errors raise `PipelineError` with context about which step failed:

```python
class PipelineError(AIForgeError):
    """Pipeline execution errors."""
```

If any step fails after all retries are exhausted, the pipeline stops and raises `PipelineError` with:
- The step name that failed
- The original error
- Results from all completed steps (partial results available)

---

## See Also

- [Quick Start](quickstart.md) — Basic pipeline example
- Module docs for individual module APIs
