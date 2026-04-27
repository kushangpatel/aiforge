# Module: Classify

`aiforge.modules.classify` — Text classification into custom categories with confidence scores. Supports zero-shot (just labels), few-shot (labels + examples), single-label, and multi-label modes.

---

## Installation

The classify module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge
```

---

## Import

```python
from aiforge import Classifier
from aiforge import ClassifyResponse, ClassifyExample, LabelScore  # types
```

---

## Quick Start

```python
from aiforge import Classifier

c = Classifier(labels=["bug", "feature_request", "question"])
result = c.classify("The login button doesn't work on mobile")
print(result.label)       # "bug"
print(result.scores)      # [LabelScore(label="bug", confidence=0.94)]
print(result.rationale)   # "The text describes a broken UI element."
```

---

## Constructor

```python
Classifier(
    labels: list[str],
    examples: list[ClassifyExample] | None = None,
    multi_label: bool = False,
    temperature: float = 0.0,
    min_confidence: float = 0.0,
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
| `labels` | `list[str]` | *(required)* | The taxonomy — list of label strings to classify into. Must not be empty. Max 100 labels. Duplicates are silently removed (order preserved). |
| `examples` | `list[ClassifyExample] \| None` | `None` | Few-shot examples to improve accuracy. Each example's label must exist in `labels`. See [Few-Shot Examples](#few-shot-examples). |
| `multi_label` | `bool` | `False` | When `True`, the LLM may return multiple applicable labels per text instead of a single best match. See [Multi-Label Mode](#multi-label-mode). |
| `temperature` | `float` | `0.0` | LLM temperature. Must be in range `[0.0, 2.0]`. Lower values (0.0) produce deterministic results; higher values add variability. |
| `min_confidence` | `float` | `0.0` | Minimum confidence threshold. Labels with confidence below this value are filtered out. Must be in range `[0.0, 1.0]`. |
| `system_prompt` | `str \| None` | `None` | Override the default system prompt. See [Prompt Resolution](#prompt-resolution). |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`. |

### Validation Rules

The constructor raises `ValueError` for:
- Empty `labels` list
- More than 100 labels
- `temperature` outside `[0.0, 2.0]`
- `min_confidence` outside `[0.0, 1.0]`
- Any example whose label is not in the `labels` taxonomy

### Examples

```python
from aiforge import Classifier, ClassifyExample

# Zero-shot — just labels, no examples needed
c = Classifier(labels=["positive", "negative", "neutral"])

# Few-shot — provide examples for better accuracy
c = Classifier(
    labels=["urgent", "normal", "low"],
    examples=[
        ClassifyExample(text="Server is down, all users affected", label="urgent"),
        ClassifyExample(text="Could you add dark mode?", label="low"),
        ClassifyExample(text="Invoice not showing correct total", label="normal"),
    ],
)

# Multi-label — text can belong to multiple categories
c = Classifier(
    labels=["politics", "economy", "technology", "health", "sports"],
    multi_label=True,
)

# With confidence threshold — only return high-confidence labels
c = Classifier(
    labels=["spam", "not_spam"],
    min_confidence=0.8,
)

# Specific model + low temperature for deterministic results
c = Classifier(
    labels=["positive", "negative", "neutral"],
    temperature=0.0,
    MODEL="gpt-4o",
)

# Quality-based model routing
c = Classifier(labels=["bug", "feature", "question"], quality="low")
```

---

## Methods

### `classify(text, *, system_prompt=None) -> ClassifyResponse`

Classify a single text into one or more labels from the taxonomy.

```python
def classify(
    self,
    text: str,
    *,
    system_prompt: str | None = None,
) -> ClassifyResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to classify. Must not be empty or whitespace-only (raises `ValueError`). |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`ClassifyResponse`](#classifyresponse)

```python
c = Classifier(labels=["bug", "feature_request", "question", "praise"])

result = c.classify("The login button doesn't work on mobile")
print(result.label)       # "bug"
print(result.scores[0])   # LabelScore(label="bug", confidence=0.94)
print(result.rationale)   # "The text describes a broken UI element."
print(result.cost)        # 0.00008
print(result.model)       # "gpt-4o-mini"
```

---

### `aclassify(text, *, system_prompt=None) -> ClassifyResponse`

Async version of `classify()`.

```python
async def aclassify(
    self,
    text: str,
    *,
    system_prompt: str | None = None,
) -> ClassifyResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to classify. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`ClassifyResponse`](#classifyresponse)

```python
result = await c.aclassify("Payment processing is failing for all customers")
print(result.label)  # "urgent"
```

---

### `run(text, **kwargs) -> ClassifyResponse`

BaseModule interface method. Delegates to `classify()`. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> ClassifyResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to classify. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`ClassifyResponse`](#classifyresponse)

```python
# Equivalent to c.classify("Hello")
result = c.run("Hello")

# In a pipeline context
result = c.run(previous_step.to_text())
```

---

### `arun(text, **kwargs) -> ClassifyResponse`

Async version of `run()`. Delegates to `aclassify()`.

```python
async def arun(self, text: str, **kwargs) -> ClassifyResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Classify multiple texts concurrently. Inherited from `BaseModule`.

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
| `items` | `list[str]` | *(required)* | List of texts to classify. |
| `concurrency` | `int` | `10` | Max concurrent LLM requests. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

Each item makes a separate LLM call (not a single mega-prompt), ensuring consistent per-item quality and cost attribution.

```python
tickets = [
    "Login button is broken on mobile",
    "Can you add dark mode support?",
    "How do I reset my password?",
    "Great product, love the UI!",
]

batch = c.run_batch(tickets, concurrency=4)

for r in batch.results:
    print(f"{r.label} ({r.scores[0].confidence:.0%}): {r.rationale}")
# bug (94%): The text describes a broken UI element.
# feature_request (91%): The user is requesting a new feature.
# question (88%): The user is asking for help.
# praise (96%): The user is expressing satisfaction.

print(f"Total cost: ${batch.total_cost:.4f}")
print(f"Failed: {len(batch.failed)}")
```

---

## Response Types

### `ClassifyResponse`

Returned by `classify()`, `aclassify()`, `run()`, and `arun()`.

```python
from aiforge import ClassifyResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str \| None` | Top predicted label, or `None` if the LLM could not determine one or confidence was below `min_confidence`. |
| `scores` | `list[LabelScore]` | All returned labels with confidence scores, sorted by confidence descending. Empty if no label matched. |
| `rationale` | `str` | Brief explanation from the LLM for the classification decision. |
| `cost` | `float` | Cost in USD for this call. |
| `input_tokens` | `int` | Input tokens consumed. |
| `output_tokens` | `int` | Output tokens generated. |
| `model` | `str` | Model used (e.g. `"gpt-4o-mini"`). |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds. |

```python
result = c.classify("The server is completely down!")

print(result.label)          # "urgent"
print(result.scores)         # [LabelScore(label="urgent", confidence=0.97)]
print(result.rationale)      # "Complete service outage affecting all users."
print(result.cost)           # 0.00008
print(result.model)          # "gpt-4o-mini"
print(result.latency_ms)     # 312.5

# to_text() returns the label (used for pipeline chaining)
print(result.to_text())      # "urgent"
# Returns "unknown" if label is None
```

---

### `LabelScore`

Represents a single label prediction with its confidence score.

```python
from aiforge import LabelScore
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `label` | `str` | *(required)* | The predicted label string. |
| `confidence` | `float` | `1.0` | Confidence score in range `[0.0, 1.0]`. |

```python
for score in result.scores:
    print(f"  {score.label}: {score.confidence:.0%}")
# In multi-label mode, there will be multiple entries
```

---

### `ClassifyExample`

A labeled example for few-shot classification.

```python
from aiforge import ClassifyExample
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Example input text. Must not be empty. |
| `label` | `str` | The correct label for this text. Must not be empty. Must be present in the classifier's `labels` taxonomy. |

```python
examples = [
    ClassifyExample(text="Server is down, all users affected", label="urgent"),
    ClassifyExample(text="Could you add dark mode?", label="low"),
    ClassifyExample(text="Invoice total is wrong", label="normal"),
]

c = Classifier(labels=["urgent", "normal", "low"], examples=examples)
```

---

## Few-Shot Examples

Providing examples improves classification accuracy, especially for ambiguous categories or domain-specific taxonomies.

```python
from aiforge import Classifier, ClassifyExample

c = Classifier(
    labels=["billing", "technical", "account", "general"],
    examples=[
        ClassifyExample(text="I was charged twice for my subscription", label="billing"),
        ClassifyExample(text="The app crashes when I open settings", label="technical"),
        ClassifyExample(text="How do I change my email address?", label="account"),
        ClassifyExample(text="What are your business hours?", label="general"),
    ],
)

result = c.classify("My credit card was declined but I still got charged")
print(result.label)  # "billing"
```

**Rules:**
- Every example label must exist in the `labels` list (validated at construction time)
- Example text and labels are escaped to prevent prompt injection
- More examples generally improve accuracy, but add to input token count and cost

---

## Multi-Label Mode

When `multi_label=True`, the classifier can assign multiple labels to a single text. Useful for content tagging, topic detection, etc.

```python
c = Classifier(
    labels=["politics", "economy", "technology", "health", "sports"],
    multi_label=True,
)

result = c.classify("The government announced a new tech tax on digital health platforms")
print(result.label)     # "politics" (top label)
print(result.scores)
# [
#   LabelScore(label="politics", confidence=0.95),
#   LabelScore(label="technology", confidence=0.88),
#   LabelScore(label="health", confidence=0.72),
#   LabelScore(label="economy", confidence=0.65),
# ]
```

**Behavior:**
- The `label` field contains the single top-scoring label
- The `scores` list contains all applicable labels sorted by confidence (descending)
- Labels below `min_confidence` are filtered out

```python
# Combine multi-label with confidence threshold
c = Classifier(
    labels=["politics", "economy", "technology", "health", "sports"],
    multi_label=True,
    min_confidence=0.7,  # only labels the model is ≥70% confident about
)

result = c.classify("The government announced a new tech tax on digital health platforms")
print(result.scores)
# Only labels with confidence >= 0.7:
# [
#   LabelScore(label="politics", confidence=0.95),
#   LabelScore(label="technology", confidence=0.88),
#   LabelScore(label="health", confidence=0.72),
# ]
```

---

## Confidence Threshold

The `min_confidence` parameter filters out low-confidence predictions. When the top label's confidence is below the threshold, **`label` is set to `None`**.

```python
c = Classifier(
    labels=["spam", "not_spam"],
    min_confidence=0.8,
)

result = c.classify("Maybe check out our sale?")  # ambiguous text
if result.label is None:
    print("Not confident enough to classify")
    print(f"Best guess was: {result.rationale}")
else:
    print(f"Classified as: {result.label}")
```

---

## Sentiment Analysis

The Classifier also provides built-in sentiment analysis via `sentiment()`. This mode detects overall polarity, emotion types, and optional aspect-level breakdown — no label configuration required.

### Quick Start

```python
from aiforge import Classifier

c = Classifier(labels=["positive", "negative", "neutral"])

# Basic sentiment
result = c.sentiment("I absolutely love this product! Best purchase ever.")
print(result.overall)     # "positive"
print(result.score)       # 0.95
print(result.emotions)    # [EmotionScore(emotion="joy", intensity=0.9), ...]
print(result.rationale)   # "Text expresses strong positive sentiment."
```

### Aspect-Level Analysis

```python
result = c.sentiment(
    "The food was amazing but the service was terrible and the price was too high.",
    aspects=["food", "service", "price"],
)
print(result.overall)     # "mixed"
print(result.score)       # -0.1
for a in result.aspects:
    print(f"  {a.aspect}: {a.sentiment} ({a.score})")
    # food: positive (0.9)
    # service: negative (-0.8)
    # price: negative (-0.6)
```

### `sentiment(text, *, aspects=None, system_prompt=None) → SentimentResponse`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Input text to analyze. |
| `aspects` | `list[str] \| None` | `None` | Aspects to evaluate individually. |
| `system_prompt` | `str \| None` | `None` | Override the system prompt. |

`asentiment()` is the async equivalent.

### SentimentResponse

| Field | Type | Description |
|-------|------|-------------|
| `overall` | `Literal["positive", "negative", "neutral", "mixed"]` | Overall sentiment. |
| `score` | `float` | Score from -1.0 (very negative) to 1.0 (very positive). |
| `aspects` | `list[AspectSentiment]` | Per-aspect results (when aspects provided). |
| `emotions` | `list[EmotionScore]` | Detected emotions with intensity. |
| `rationale` | `str` | Brief explanation. |

Plus inherited: `cost`, `input_tokens`, `output_tokens`, `model`, `trace_id`, `latency_ms`.

### CLI

```bash
# Basic sentiment
aiforge sentiment --text "I love this product!"

# With aspects
aiforge sentiment --text "Great food but slow service." --aspects "food,service"
```

### API

```
POST /classify/sentiment
```

```json
{
  "text": "The food was great but service was slow.",
  "aspects": ["food", "service"]
}
```

---

## Prompt Resolution

System prompts are resolved in this precedence order (highest to lowest):

1. **Per-call override:** `c.classify(text, system_prompt="...")`
2. **Constructor override:** `Classifier(labels=..., system_prompt="...")`
3. **Config file:** `CUSTOM_PROMPTS.classify.system` in `aiforge.yaml`
4. **Built-in default:** *"You are a text classification engine. Assign the most accurate label from the provided taxonomy to the given text. Be precise and consistent. Always respond ONLY with valid JSON — no markdown, no extra prose."*

---

## Label Matching

The classifier uses smart label matching to handle minor LLM inconsistencies:

1. **Exact match** — tries exact string match first
2. **Case-insensitive fallback** — if no exact match, tries case-insensitive (logs a warning)
3. **Unknown label** — if the LLM returns a label not in the taxonomy, it's dropped and `label` is set to `None`

This means your taxonomy labels are case-sensitive by convention, but the classifier is forgiving of LLM output case variations.

---

## Instance Properties

After constructing a `Classifier`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `c.labels` | `list[str]` | Deduplicated label taxonomy. |
| `c.examples` | `list[ClassifyExample]` | Registered few-shot examples (empty list if none). |
| `c.multi_label` | `bool` | Whether multi-label mode is active. |
| `c.temperature` | `float` | LLM temperature setting. |
| `c.min_confidence` | `float` | Confidence threshold. |
| `c.settings` | `AIForgeSettings` | Resolved configuration. |
| `c.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Zero-Shot Ticket Routing

```python
from aiforge import Classifier

router = Classifier(labels=["billing", "technical", "account", "general"])

result = router.classify("I can't log in to my account")
print(result.label)       # "account"
print(result.rationale)   # "The user is having an account access issue."
```

### Few-Shot Sentiment Analysis

```python
from aiforge import Classifier, ClassifyExample

sentiment = Classifier(
    labels=["positive", "negative", "neutral"],
    examples=[
        ClassifyExample(text="This product is amazing!", label="positive"),
        ClassifyExample(text="Terrible experience, would not recommend", label="negative"),
        ClassifyExample(text="It works as expected", label="neutral"),
    ],
    temperature=0.0,
)

reviews = [
    "Best purchase I've made this year!",
    "The product broke after one week",
    "Shipping was fast, product is okay",
]

batch = sentiment.run_batch(reviews, concurrency=3)
for r in batch.results:
    print(f"  {r.label} ({r.scores[0].confidence:.0%})")
# positive (95%)
# negative (92%)
# neutral (78%)
```

### Multi-Label Content Tagging

```python
from aiforge import Classifier

tagger = Classifier(
    labels=["python", "javascript", "tutorial", "advanced", "beginner", "web", "data-science"],
    multi_label=True,
    min_confidence=0.5,
)

result = tagger.classify(
    "Learn how to build a REST API with FastAPI and deploy it to AWS"
)

print(result.label)  # "python" (top label)
tags = [s.label for s in result.scores]
print(tags)  # ["python", "web", "tutorial", "beginner"]
```

### Pipeline Routing

The `to_text()` method returns the label string, making `Classifier` a natural routing step in pipelines:

```python
from aiforge import Classifier, Summarizer, ChatAgent

# Step 1: Classify the input
router = Classifier(labels=["question", "task", "feedback"])
classification = router.classify(user_input)

# Step 2: Route to the right handler
if classification.to_text() == "question":
    agent = ChatAgent(system_prompt="Answer the user's question.")
    response = agent.chat(user_input)
elif classification.to_text() == "task":
    # handle task...
    pass
elif classification.to_text() == "unknown":
    # classification failed — handle gracefully
    pass
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
c = Classifier(labels=["positive", "negative", "neutral"])
result = c.classify("Great service!")
print(result.model)  # "gpt-4o-mini"
```

### Model Override per Classifier

```python
# High-accuracy classifier
precise = Classifier(labels=["spam", "not_spam"], MODEL="gpt-4o", temperature=0.0)

# Cost-efficient bulk classifier
cheap = Classifier(labels=["spam", "not_spam"], quality="low")
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Empty labels, >100 labels, empty text, invalid temperature/min_confidence, example label not in taxonomy | `Classifier(labels=[])` |
| `ProviderError` | LLM provider returns an error | API key invalid |
| `RateLimitError` | Provider rate limit exceeded | Too many requests |
| `APITimeoutError` | Provider request timed out | Slow network |

```python
from aiforge.core.types import ProviderError

try:
    result = c.classify("Some text")
except ValueError as e:
    print(f"Validation error: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

**Graceful degradation:** If the LLM returns malformed JSON or an unrecognized label, the classifier does not raise an error. Instead, it returns a `ClassifyResponse` with `label=None` and empty `scores`.

---

## File Structure

```
aiforge/modules/classify/
├── __init__.py        # Exports: Classifier, ClassifyResponse, ClassifyExample, LabelScore
├── module.py          # Classifier class (constructor, classify/aclassify, label matching, parsing)
├── prompts.py         # Prompt templates (CLASSIFY_SYSTEM, zero-shot/few-shot, single/multi-label)
├── schemas.py         # Pydantic models (ClassifyResponse, ClassifyExample, LabelScore)
└── widget.py          # Streamlit playground widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Getting started with AIForge
- [Pipelines](../pipelines.md) — Use classification as a routing step
- [API Server](../api-server.md) — `POST /classify` endpoint
- [Playground](../playground.md) — Interactive classification tab
- [Configuration](../configuration.md) — Full settings reference
