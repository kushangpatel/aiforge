# Pipelines

Chain any combination of AIForge modules into **sequential, composable
pipelines** with optional conditional routing. The output of each step feeds
into the next via the `to_text()` protocol, and cost/token metrics are
automatically aggregated across every step.

---

## Table of Contents

- [Pipelines](#pipelines)
  - [Table of Contents](#table-of-contents)
  - [Why Pipelines?](#why-pipelines)
  - [Quick Start](#quick-start)
  - [Pipeline Constructor](#pipeline-constructor)
  - [Step Definitions](#step-definitions)
  - [Sequential Execution Flow](#sequential-execution-flow)
  - [The `to_text()` Protocol](#the-to_text-protocol)
    - [Built-in Response Mappings](#built-in-response-mappings)
    - [Custom `to_text()` Implementation](#custom-to_text-implementation)
  - [PipelineResult](#pipelineresult)
    - [Fields](#fields)
    - [Accessing Step Results](#accessing-step-results)
    - [Cost \& Token Aggregation](#cost--token-aggregation)
  - [Conditional Routing](#conditional-routing)
    - [Router Constructor](#router-constructor)
    - [Branch Resolution](#branch-resolution)
    - [Halting with `None` Branches](#halting-with-none-branches)
    - [Default Branch](#default-branch)
    - [`halt_on_flag` Control](#halt_on_flag-control)
  - [Async Pipelines](#async-pipelines)
  - [Pipeline with Guardrails](#pipeline-with-guardrails)
  - [Custom Modules in Pipelines](#custom-modules-in-pipelines)
  - [Error Handling](#error-handling)
  - [REST API — POST /pipeline](#rest-api--post-pipeline)
    - [Request Schema](#request-schema)
    - [Response Schema](#response-schema)
    - [Error Responses](#error-responses)
  - [Integration Examples](#integration-examples)
    - [Python SDK](#python-sdk)
    - [REST API (curl)](#rest-api-curl)
    - [Node.js](#nodejs)
    - [Go](#go)
    - [Java](#java)
  - [Playground](#playground)
  - [Troubleshooting](#troubleshooting)
  - [See Also](#see-also)

---

## Why Pipelines?

| Without pipelines | With AIForge pipelines |
|-------------------|------------------------|
| Manual glue code between each LLM call | Declarative step list, data flows automatically |
| Ad-hoc cost tracking across calls | `PipelineResult.total_cost` aggregates everything |
| No conditional branching | `Router` routes by classification label |
| Test each step separately, hope they compose | Pipeline handles type checking via `to_text()` |
| Separate sync/async implementations | `run()` and `arun()` with identical semantics |

---

## Quick Start

```python
from aiforge import Pipeline, Summarizer, Classifier

pipe = Pipeline([
    ("summarize", Summarizer(style="concise")),
    ("classify", Classifier(labels=["positive", "negative", "neutral"])),
])

result = pipe.run("Long customer feedback email text here...")

print(result.steps["summarize"].summary)    # Concise summary
print(result.steps["classify"].label)       # "positive"
print(f"Total cost: ${result.total_cost:.4f}")
print(f"Total tokens: {result.total_tokens}")
```

---

## Pipeline Constructor

```python
from aiforge import Pipeline

pipe = Pipeline(
    steps=[
        ("step_name", module_instance),
        ("step_name_2", module_instance_2),
        # ...
    ],
    halt_on_flag=True,   # default: stop when Router resolves to None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `list[tuple[str, BaseModule \| Router]]` | *required* | Ordered list of `(name, module_or_router)` tuples |
| `halt_on_flag` | `bool` | `True` | Stop execution when a Router branch resolves to `None` |

**Validation rules:**
- Step names must be **unique** — duplicate names raise `ValueError` at construction time
- Each step must be a `BaseModule` subclass or a `Router` instance
- An empty steps list is allowed (returns empty `PipelineResult`)

---

## Step Definitions

Each step is a `(name, module)` tuple:

```python
from aiforge import Summarizer, Classifier, Extractor, Generator, Router

steps = [
    ("extract", Extractor(output_schema=TicketSchema)),   # BaseModule
    ("classify", Classifier(labels=["bug", "feature"])), # BaseModule
    ("route", Router({                                     # Router
        "bug": Summarizer(style="bullet_points"),
        "feature": Generator(template="Draft a feature spec for: {text}"),
    })),
]
```

Step names are used as keys in the `PipelineResult.steps` dict and appear in
error messages — choose descriptive names.

---

## Sequential Execution Flow

```
Pipeline.run("input text")
    │
    ├─ step 1: module.run("input text")
    │     └─ response₁.to_text() → "text for step 2"
    │
    ├─ step 2: module.run("text for step 2")
    │     └─ response₂.to_text() → "text for step 3"
    │
    ├─ step 3 (Router): resolve(response₂.label, "text for step 3")
    │     └─ branch_module.run("text for step 3")
    │           └─ response₃.to_text() → "final output"
    │
    └─ return PipelineResult {
         steps: {"step1": response₁, "step2": response₂, "step3": response₃},
         total_cost: sum(costs),
         total_tokens: sum(tokens),
         halted: false,
         halted_at: null
       }
```

1. `current_input` is initialised to the input `text`
2. For each step in order:
   - **Module step:** calls `module.run(current_input)`
   - **Router step:** reads `.label` from the previous response, resolves to a
     branch module, calls `branch.run(current_input)`
3. The response's `to_text()` output becomes the next step's `current_input`
4. All responses are collected in an ordered dict keyed by step name
5. Cost and tokens are summed across all steps

---

## The `to_text()` Protocol

Every `ModuleResponse` subclass implements `to_text() -> str`, which extracts
the canonical text payload to feed into the next pipeline step. This is the
**only** contract between pipeline steps — modules don't need to know about
each other.

### Built-in Response Mappings

| Response Type | `to_text()` Returns | Description |
|--------------|---------------------|-------------|
| `SummarizeResponse` | `.summary` | The generated summary text |
| `QAResponse` | `.text` | The answer text |
| `ExtractResponse` | `.data.model_dump_json()` | Structured data serialised as JSON string |
| `ClassifyResponse` | `.text` (original input) | Original text passed through; `.label` used for routing |
| `SearchResponse` | `"\n\n".join(hit.text ...)` | Search hits joined as paragraphs |
| `GenerateResponse` | `.text` | The generated text |
| `ModerateResponse` | pass-through text | If `flagged=True` and `halt_on_flag=True`, halts pipeline |
| `ChatResponse` | `.text` | The assistant's reply |

### Custom `to_text()` Implementation

Custom module responses **must** implement `to_text()`:

```python
from aiforge.core.types import ModuleResponse

class SentimentResponse(ModuleResponse):
    sentiment: str = ""
    confidence: float = 0.0
    original_text: str = ""

    def to_text(self) -> str:
        # Return what the next step should receive as input
        return self.original_text
```

If `to_text()` raises `NotImplementedError`, the pipeline logs a warning and
keeps the previous input unchanged.

---

## PipelineResult

### Fields

```python
from aiforge.pipeline.schemas import PipelineResult

result: PipelineResult
```

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `dict[str, ModuleResponse]` | Results keyed by step name (insertion-ordered) |
| `total_cost` | `float` | Sum of `cost` across all steps (USD) |
| `total_tokens` | `int` | Sum of `input_tokens + output_tokens` across all steps |
| `halted` | `bool` | `True` if a Router's `None` branch stopped execution |
| `halted_at` | `str \| None` | Step name where the pipeline halted (or `None`) |

### Accessing Step Results

```python
result = pipe.run("Some input text")

# By step name
summary = result.steps["summarize"]        # ModuleResponse subclass
label = result.steps["classify"].label     # Access subclass fields

# Last step convenience
final = result.last                         # Last step's ModuleResponse

# Safe access
maybe = result.get("optional_step")         # None if step doesn't exist
```

### Cost & Token Aggregation

```python
result = pipe.run(text)

print(f"Total cost: ${result.total_cost:.4f}")
print(f"Total tokens: {result.total_tokens:,}")

# Per-step breakdown
for name, resp in result.steps.items():
    print(f"  {name}: ${resp.cost:.4f} | {resp.input_tokens + resp.output_tokens} tokens")
```

**Cost aggregation formula:**
```python
total_cost = sum(r.cost for r in result.steps.values())
total_tokens = sum(r.input_tokens + r.output_tokens for r in result.steps.values())
```

Pipeline costs are also recorded in the session-level `CostTracker` (each
module call records individually).

---

## Conditional Routing

The `Router` class enables **branch-based execution** where the previous step's
classification label determines which module runs next.

### Router Constructor

```python
from aiforge import Router

router = Router(
    branches={
        "support": Summarizer(style="bullet_points"),
        "sales": Generator(template="Draft a sales response for: {text}"),
        "spam": None,           # drop — halts pipeline
    },
    default=Summarizer(),       # optional fallback for unmapped labels
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branches` | `dict[str, BaseModule \| None]` | *required* | Label → module mapping; `None` = halt |
| `default` | `BaseModule \| None` | (no default) | Fallback module for unmapped labels |

### Branch Resolution

1. The Router reads the `.label` attribute from the **previous step's** response
2. Looks up the label in `branches`:
   - **Found → module:** calls `module.run(current_input)` and continues
   - **Found → `None`:** halts the pipeline (see below)
   - **Not found → `default`:** if a default module was provided, runs it
   - **Not found → no default:** raises `KeyError` with available branch names

```python
pipe = Pipeline([
    ("classify", Classifier(labels=["support", "sales", "spam"])),
    ("route", Router({
        "support": Summarizer(style="bullet_points"),
        "sales": Generator(template="Draft response: {text}"),
        "spam": None,
    })),
])

result = pipe.run("I need help with my account")
# classify.label → "support" → Router runs Summarizer
```

### Halting with `None` Branches

Setting a branch to `None` signals that the input should be **dropped**:

```python
router = Router({
    "spam": None,           # drop spam — pipeline stops here
    "real": Summarizer(),   # process real messages
})
```

When a `None` branch is hit:
- `PipelineResult.halted` is set to `True`
- `PipelineResult.halted_at` is set to the Router step name
- No further steps execute
- Completed steps are still in `result.steps`

### Default Branch

Handle unknown or unexpected labels with a fallback:

```python
router = Router(
    branches={
        "critical": Summarizer(style="detailed"),
        "low": Summarizer(style="concise"),
    },
    default=Summarizer(style="concise"),  # catches "medium", "high", etc.
)
```

Without a `default`, unmapped labels raise `KeyError`:
```
KeyError: "Label 'medium' not found in router branches. Available: ['critical', 'low']"
```

### `halt_on_flag` Control

When a Router resolves to `None`:

| `halt_on_flag` | Behaviour |
|----------------|-----------|
| `True` (default) | Pipeline stops; `halted=True`, `halted_at=step_name` |
| `False` | Pipeline **continues** — step produces no output, next step receives previous input |

```python
# Continue processing even if some branches return None
pipe = Pipeline(
    steps=[
        ("classify", classifier),
        ("route", Router({"spam": None, "real": summarizer})),
        ("generate", generator),  # still runs even after "spam" → None
    ],
    halt_on_flag=False,
)
```

---

## Async Pipelines

Every pipeline supports async execution via `arun()`. Each step calls
`module.arun()` instead of `module.run()` — same semantics, fully non-blocking:

```python
import asyncio
from aiforge import Pipeline, Summarizer, Classifier

pipe = Pipeline([
    ("summarize", Summarizer(style="concise")),
    ("classify", Classifier(labels=["positive", "negative"])),
])

async def main():
    result = await pipe.arun("Long text to process...")
    print(result.steps["summarize"].summary)
    print(result.steps["classify"].label)
    print(f"Cost: ${result.total_cost:.4f}")

asyncio.run(main())
```

Router steps use `resolve_async()` internally, which calls `module.arun()` on
the resolved branch module.

---

## Pipeline with Guardrails

Use the `Moderator` module at the beginning and/or end of a pipeline to enforce
content safety:

```python
from aiforge import Pipeline, Moderator, Summarizer, Generator

pipe = Pipeline([
    ("guard_input", Moderator(checks=["prompt_injection", "jailbreak"])),
    ("summarize", Summarizer(style="concise")),
    ("generate", Generator(template="Write a response based on: {text}")),
    ("guard_output", Moderator(checks=["pii", "toxicity"])),
])

result = pipe.run("User-submitted content here...")
# If input is flagged → pipeline halts at "guard_input"
# If output is flagged → pipeline halts at "guard_output"
```

The `Moderator` module sets `flagged=True` on its response when content
violates any of the configured checks. Combined with `halt_on_flag=True`
(the default), this stops the pipeline before unsafe content propagates.

---

## Custom Modules in Pipelines

Any `BaseModule` subclass works in pipelines. Register with the plugin system
for REST API access and discoverability:

```python
from aiforge.core.types import BaseModule, ModuleResponse
from aiforge.registry import register_module
from aiforge import Pipeline, Summarizer

class SentimentResponse(ModuleResponse):
    sentiment: str = ""
    confidence: float = 0.0
    text: str = ""

    def to_text(self) -> str:
        return self.text    # pass original text to next step

class SentimentAnalyzer(BaseModule):
    def run(self, text: str, **kwargs) -> SentimentResponse:
        result = self._call_llm(f"Classify sentiment of: {text}")
        return SentimentResponse(
            sentiment=result.text.strip(),
            confidence=0.95,
            text=text,
            cost=result.cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
        )

    async def arun(self, text: str, **kwargs) -> SentimentResponse:
        result = await self._acall_llm(f"Classify sentiment of: {text}")
        return SentimentResponse(
            sentiment=result.text.strip(),
            confidence=0.95,
            text=text,
            cost=result.cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
        )

# Register for REST API + discovery
register_module("sentiment", SentimentAnalyzer)

# Use in a pipeline
pipe = Pipeline([
    ("sentiment", SentimentAnalyzer()),
    ("summarize", Summarizer(style="concise")),
])
result = pipe.run("I absolutely love this product!")
print(result.steps["sentiment"].sentiment)   # "positive"
print(result.steps["summarize"].summary)     # summarised text
```

**Checklist for custom pipeline modules:**
1. Extend `BaseModule`
2. Implement `run()` and `arun()`
3. Return a `ModuleResponse` subclass
4. Implement `to_text()` on the response class
5. Include `cost`, `input_tokens`, `output_tokens`, `model` in the response

---

## Error Handling

| Scenario | Exception | Details |
|----------|-----------|---------|
| Duplicate step names | `ValueError` | Raised at `Pipeline()` construction |
| Module execution failure | Original exception | Propagates from the module (not wrapped) |
| Router without `.label` on previous step | `PipelineError` | Previous step's response must have a `label` attribute |
| Unknown router label, no default | `KeyError` | Includes available branch names in message |
| `to_text()` not implemented | Warning logged | Pipeline continues with previous input |

```python
from aiforge.core.types import PipelineError

try:
    result = pipe.run("input text")
except PipelineError as e:
    print(f"Pipeline failed: {e}")
except KeyError as e:
    print(f"Router label not found: {e}")
```

Completed steps are **not** available in the exception — if you need partial
results on failure, wrap individual steps in try/except or use individual
module calls.

---

## REST API — POST /pipeline

The API server exposes a dynamic pipeline endpoint that constructs and
executes pipelines from JSON step definitions.

### Request Schema

```
POST /pipeline
Content-Type: application/json
Authorization: Bearer <api_key>
```

```json
{
  "text": "Long customer complaint email text...",
  "steps": [
    {
      "name": "summarize",
      "module": "Summarizer",
      "config": { "style": "concise" }
    },
    {
      "name": "classify",
      "module": "Classifier",
      "config": { "categories": ["bug", "feature", "question"] }
    }
  ],
  "context": {
    "customer_id": "C-12345"
  }
}
```

**Step definition fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique step name |
| `module` | `str` | Yes | Registered module class name (e.g., `"Summarizer"`, `"Classifier"`) |
| `config` | `dict` | No | kwargs passed to the module constructor |

**Top-level fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | Yes | Input text for the first step |
| `steps` | `list[StepDef]` | Yes | Ordered list of step definitions |
| `context` | `dict` | No | Shared metadata (not modified by modules) |

Module names are resolved via the registry (`get_module()`), so custom
registered modules work too.

### Response Schema

```json
{
  "steps": {
    "summarize": {
      "summary": "Customer reports billing issue...",
      "cost": 0.003,
      "input_tokens": 1200,
      "output_tokens": 150,
      "model": "gpt-4o",
      "trace_id": "",
      "latency_ms": 820.5
    },
    "classify": {
      "label": "bug",
      "confidence": 0.94,
      "cost": 0.0015,
      "input_tokens": 200,
      "output_tokens": 10,
      "model": "gpt-4o",
      "trace_id": "",
      "latency_ms": 340.2
    }
  },
  "total_cost": 0.0045,
  "total_tokens": 1560,
  "halted": false,
  "halted_at": null
}
```

Response subclass fields (e.g., `summary`, `label`, `confidence`) are
preserved via a custom serializer that calls each subclass's `model_dump()`.

### Error Responses

| Status | Condition | Body |
|--------|-----------|------|
| `200` | Success | `PipelineResult` JSON |
| `422` | Empty steps list | `{"detail": "Steps list cannot be empty"}` |
| `422` | Unknown module name | `{"detail": "Unknown module: 'BadName'"}` |
| `422` | Invalid config | `{"detail": "ValueError: ..."}` |
| `500` | Execution error | `{"detail": "Pipeline execution failed: ..."}` |

---

## Integration Examples

### Python SDK

```python
from aiforge import Pipeline, Summarizer, Classifier, Router, Generator

# ── Sequential pipeline ──
pipe = Pipeline([
    ("summarize", Summarizer(style="concise")),
    ("classify", Classifier(labels=["positive", "negative", "neutral"])),
])
result = pipe.run("Customer feedback text...")
print(result.steps["classify"].label)
print(f"Cost: ${result.total_cost:.4f}")

# ── Conditional routing ──
pipe = Pipeline([
    ("classify", Classifier(labels=["support", "sales", "spam"])),
    ("route", Router({
        "support": Summarizer(style="bullet_points"),
        "sales": Generator(template="Draft response for: {text}"),
        "spam": None,
    })),
])
result = pipe.run("I need help with my order")
if result.halted:
    print(f"Pipeline halted at: {result.halted_at}")
else:
    print(result.last.to_text())

# ── Async ──
import asyncio

async def process():
    result = await pipe.arun("Async input text...")
    return result

asyncio.run(process())
```

### REST API (curl)

```bash
# Sequential pipeline
curl -s http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AIFORGE_API_KEY" \
  -d '{
    "text": "Long customer email text...",
    "steps": [
      {"name": "summarize", "module": "Summarizer", "config": {"style": "concise"}},
      {"name": "classify", "module": "Classifier", "config": {"categories": ["bug", "feature"]}}
    ]
  }' | jq .

# With context metadata
curl -s http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AIFORGE_API_KEY" \
  -d '{
    "text": "Input text",
    "steps": [
      {"name": "extract", "module": "Extractor", "config": {}},
      {"name": "summarize", "module": "Summarizer", "config": {"style": "bullet_points"}}
    ],
    "context": {"source": "email", "customer_id": "C-123"}
  }' | jq '{total_cost, total_tokens, halted}'
```

### Node.js

```javascript
const BASE = "http://localhost:8000";
const HEADERS = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${process.env.AIFORGE_API_KEY}`,
};

async function runPipeline(text, steps, context = null) {
  const body = { text, steps };
  if (context) body.context = context;

  const res = await fetch(`${BASE}/pipeline`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`Pipeline failed (${res.status}): ${err.detail}`);
  }
  return res.json();
}

// Usage
const result = await runPipeline(
  "Long customer feedback...",
  [
    { name: "summarize", module: "Summarizer", config: { style: "concise" } },
    { name: "classify", module: "Classifier", config: { categories: ["positive", "negative"] } },
  ]
);

console.log("Summary:", result.steps.summarize.summary);
console.log("Label:", result.steps.classify.label);
console.log(`Total cost: $${result.total_cost.toFixed(4)}`);
console.log(`Halted: ${result.halted}`);
```

### Go

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type StepDef struct {
	Name   string         `json:"name"`
	Module string         `json:"module"`
	Config map[string]any `json:"config,omitempty"`
}

type PipelineRequest struct {
	Text    string         `json:"text"`
	Steps   []StepDef      `json:"steps"`
	Context map[string]any `json:"context,omitempty"`
}

type PipelineResult struct {
	Steps       map[string]map[string]any `json:"steps"`
	TotalCost   float64                   `json:"total_cost"`
	TotalTokens int                       `json:"total_tokens"`
	Halted      bool                      `json:"halted"`
	HaltedAt    *string                   `json:"halted_at"`
}

func runPipeline(text string, steps []StepDef) (*PipelineResult, error) {
	reqBody := PipelineRequest{Text: text, Steps: steps}
	body, _ := json.Marshal(reqBody)

	req, _ := http.NewRequest("POST", "http://localhost:8000/pipeline",
		bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result PipelineResult
	json.NewDecoder(resp.Body).Decode(&result)
	fmt.Printf("Cost: $%.4f | Tokens: %d | Halted: %v\n",
		result.TotalCost, result.TotalTokens, result.Halted)
	return &result, nil
}

func main() {
	result, _ := runPipeline("Customer feedback text...", []StepDef{
		{Name: "summarize", Module: "Summarizer", Config: map[string]any{"style": "concise"}},
		{Name: "classify", Module: "Classifier", Config: map[string]any{
			"categories": []string{"positive", "negative"},
		}},
	})
	fmt.Println("Summary:", result.Steps["summarize"]["summary"])
	fmt.Println("Label:", result.Steps["classify"]["label"])
}
```

### Java

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.*;
import java.util.*;

public class AiForgePipelineExample {
    private static final String BASE = "http://localhost:8000";
    private static final String API_KEY = System.getenv("AIFORGE_API_KEY");
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final HttpClient client = HttpClient.newHttpClient();

    public static Map<String, Object> runPipeline(
            String text, List<Map<String, Object>> steps) throws Exception {

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text", text);
        body.put("steps", steps);

        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(BASE + "/pipeline"))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer " + API_KEY)
            .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
            .build();

        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new RuntimeException("Pipeline failed: " + resp.body());
        }

        Map<String, Object> result = mapper.readValue(resp.body(), Map.class);
        System.out.printf("Cost: $%.4f | Tokens: %s | Halted: %s%n",
            ((Number) result.get("total_cost")).doubleValue(),
            result.get("total_tokens"),
            result.get("halted"));
        return result;
    }

    public static void main(String[] args) throws Exception {
        List<Map<String, Object>> steps = List.of(
            Map.of("name", "summarize", "module", "Summarizer",
                   "config", Map.of("style", "concise")),
            Map.of("name", "classify", "module", "Classifier",
                   "config", Map.of("categories", List.of("bug", "feature")))
        );

        Map<String, Object> result = runPipeline("Customer complaint...", steps);
        Map<String, Object> stepsResult = (Map<String, Object>) result.get("steps");
        Map<String, Object> summary = (Map<String, Object>) stepsResult.get("summarize");
        Map<String, Object> classify = (Map<String, Object>) stepsResult.get("classify");
        System.out.println("Summary: " + summary.get("summary"));
        System.out.println("Label: " + classify.get("label"));
    }
}
```

---

## Playground

The Streamlit pipeline playground provides an interactive UI for building and
testing pipelines without writing code:

```bash
aiforge playground
# Opens at http://localhost:8501 → Pipeline page
```

**Features:**
- Add/remove steps dynamically
- Choose from available modules (Summarizer, Classifier, Generator, Moderator)
- Configure module-specific parameters per step
- Execute pipeline and view per-step results
- View cost/token breakdown for each step and totals
- See halt status if a Router's `None` branch was triggered

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ValueError: Duplicate step name` | Two steps share the same name | Use unique names for every step |
| `PipelineError: Router requires label` | Previous step's response has no `.label` attribute | Ensure the step before a Router is a `Classifier` or similar label-producing module |
| `KeyError: Label 'X' not found` | Router has no branch for the label | Add the label to `branches` or provide a `default` module |
| `NotImplementedError: to_text()` | Custom response missing `to_text()` | Override `to_text()` on your `ModuleResponse` subclass |
| Pipeline halts unexpectedly | Router matched a `None` branch | Check classification labels; set `halt_on_flag=False` to continue |
| `422` from REST API | Unknown module name or empty steps | Verify module names match registered names (case-sensitive) |
| Cost shows `0.0` for a step | Module returned cached result or used unknown model | Check cache settings; add model to `custom_pricing` if needed |
| Step result missing subclass fields | Serialization issue | Ensure your response class extends `ModuleResponse` and calls `super().__init__()` |

---

## See Also

- [Configuration](configuration.md) — Full configuration reference
- [Cost Tracking](cost-tracking.md) — Cost aggregation and pricing
- [API Server](api-server.md) — REST API endpoints
- [Modules](modules/) — Detailed documentation for each built-in module
- [Quick Start](quickstart.md) — Getting started with pipelines
