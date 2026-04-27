# Module: Generate

`aiforge.modules.generate` — Template-based and freeform content generation with built-in guardrails. For emails, reports, product descriptions, marketing copy — anything with a repeatable structure or safety requirements.

---

## Installation

The generate module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge
```

---

## Import

```python
from aiforge import Generator
from aiforge import GenerateResponse          # response type
from aiforge import register_guardrail        # custom guardrail decorator
from aiforge import GuardrailResult           # guardrail return type
```

---

## Quick Start

```python
from aiforge import Generator

# Freeform
g = Generator()
result = g.generate(prompt="Write a professional email thanking a client for their order.")
print(result.text)

# Template-based
g = Generator(template="Write a {tone} description for {product}.")
result = g.generate(tone="professional", product="AIForge SDK")
print(result.text)
```

---

## Constructor

```python
Generator(
    template: str | None = None,
    guardrails: list[str] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    competitors: list[str] | None = None,
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
| `template` | `str \| None` | `None` | Template string with `{variable}` placeholders. Variables are substituted at call time via keyword arguments. Mutually exclusive with `prompt` at call time. |
| `guardrails` | `list[str] \| None` | `None` | List of guardrail names to enforce on generated output. Built-in: `"no_pii"`, `"no_competitor_mentions"`, `"no_pricing"`, `"professional_tone"`. See [Guardrails](#guardrails). |
| `max_tokens` | `int` | `1024` | Maximum tokens for LLM generation. Must be ≥ 1. |
| `temperature` | `float` | `0.7` | LLM sampling temperature. Must be in range `[0.0, 2.0]`. Higher values produce more creative output. |
| `competitors` | `list[str] \| None` | `None` | List of competitor names for the `no_competitor_mentions` guardrail. Required when using that guardrail. |
| `system_prompt` | `str \| None` | `None` | Override the default system prompt. See [Prompt Resolution](#prompt-resolution). |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, etc.). Falls back to `settings.PROVIDER`. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`. |

### Examples

```python
from aiforge import Generator

# Freeform with guardrails
g = Generator(
    guardrails=["no_pii", "professional_tone"],
)

# Template-based with competitor protection
g = Generator(
    template="Write a {tone} comparison of {product} vs the competition.",
    guardrails=["no_competitor_mentions", "no_pricing"],
    competitors=["CompetitorA", "CompetitorB"],
)

# High creativity with more tokens
g = Generator(
    temperature=1.2,
    max_tokens=2048,
)

# Specific model
g = Generator(
    template="Draft a {length} blog post about {topic}.",
    MODEL="gpt-4o",
    TEMPERATURE=0.8,
)

# Quality-based model routing
g = Generator(quality="high")  # best model for creative work
g = Generator(quality="low")   # cheapest model for drafts
```

---

## Methods

### `generate(prompt=None, *, system_prompt=None, **variables) -> GenerateResponse`

Generate content from a template (with variables) or a freeform prompt.

```python
def generate(
    self,
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    **variables: Any,
) -> GenerateResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| None` | `None` | Freeform generation prompt. Mutually exclusive with template mode. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |
| `**variables` | `Any` | | Template variable substitutions. Keys must match `{placeholders}` in the template. List values are automatically joined with `", "`. |

**Returns:** [`GenerateResponse`](#generateresponse)

**Rules:**
- If a `template` was provided at construction, pass template variables as keyword arguments (do NOT pass `prompt`)
- If no `template` was set, pass `prompt` as a freeform string
- Raises `ValueError` if both `prompt` and template variables are provided, or if neither is available

```python
# Template mode — pass variables as kwargs
g = Generator(template="Write a {tone} email about {topic}.")
result = g.generate(tone="friendly", topic="product launch")
print(result.text)

# Freeform mode — pass prompt string
g = Generator()
result = g.generate(prompt="Write a haiku about software engineering")
print(result.text)

# With guardrails
g = Generator(guardrails=["no_pii", "professional_tone"])
result = g.generate(prompt="Draft a customer response email")
print(result.text)
print(result.guardrail_flags)  # [] = all guardrails passed
print(result.retry_count)      # 0 = passed on first attempt
```

---

### `agenerate(prompt=None, *, system_prompt=None, **variables) -> GenerateResponse`

Async version of `generate()`.

```python
async def agenerate(
    self,
    prompt: str | None = None,
    *,
    system_prompt: str | None = None,
    **variables: Any,
) -> GenerateResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| None` | `None` | Freeform generation prompt. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |
| `**variables` | `Any` | | Template variable substitutions. |

**Returns:** [`GenerateResponse`](#generateresponse)

```python
result = await g.agenerate(tone="casual", topic="team outing")
print(result.text)
```

---

### `run(text, **kwargs) -> GenerateResponse`

BaseModule interface method. Treats `text` as a freeform prompt. Used by pipelines.

```python
def run(self, text: str, **kwargs) -> GenerateResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Freeform prompt. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`GenerateResponse`](#generateresponse)

```python
result = g.run("Write a product announcement for the new SDK release")
```

---

### `arun(text, **kwargs) -> GenerateResponse`

Async version of `run()`. Delegates to `agenerate()`.

```python
async def arun(self, text: str, **kwargs) -> GenerateResponse
```

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Generate content for multiple prompts concurrently. Inherited from `BaseModule`.

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
| `items` | `list[str]` | *(required)* | List of freeform prompts. |
| `concurrency` | `int` | `10` | Max concurrent LLM requests. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

```python
prompts = [
    "Write a tagline for a coffee shop",
    "Write a tagline for a bookstore",
    "Write a tagline for a gym",
]

batch = g.run_batch(prompts, concurrency=3)
for r in batch.results:
    print(f"  {r.text}")
print(f"Total cost: ${batch.total_cost:.4f}")
```

---

## Response Types

### `GenerateResponse`

Returned by `generate()`, `agenerate()`, `run()`, and `arun()`.

```python
from aiforge import GenerateResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The generated content. |
| `guardrail_flags` | `list[str]` | Names of guardrails still violated after all retries. Empty list = all checks passed. |
| `retry_count` | `int` | Number of retry attempts (0 = passed on first try). |
| `cost` | `float` | Total cost in USD (including retries and guardrail LLM checks). |
| `input_tokens` | `int` | Total input tokens consumed (including retries). |
| `output_tokens` | `int` | Total output tokens generated (including retries). |
| `model` | `str` | Model used (e.g. `"gpt-4o-mini"`). |
| `trace_id` | `str` | Unique trace ID (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds (including retries). |

```python
result = g.generate(prompt="Write a professional bio for a software engineer")

print(result.text)              # "Jane is a senior software engineer..."
print(result.guardrail_flags)   # [] = clean
print(result.retry_count)       # 0
print(result.cost)              # 0.0003
print(result.model)             # "gpt-4o-mini"
print(result.latency_ms)        # 523.4

# to_text() returns the generated content (for pipeline chaining)
print(result.to_text())         # same as result.text
```

---

### `GuardrailResult`

Return type for custom guardrail functions. A dataclass, not a Pydantic model.

```python
from aiforge import GuardrailResult
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `passed` | `bool` | *(required)* | `True` if the guardrail check passed. |
| `message` | `str` | `""` | Optional explanation of why it passed or failed. |

```python
result = GuardrailResult(passed=True, message="All good")
result = GuardrailResult(passed=False, message="Contains competitor name")
```

---

## Template Mode

Templates use `{variable}` placeholders that are substituted at call time. List values are automatically joined with `", "`.

```python
g = Generator(
    template="""Write a {tone} product description for:
Product: {product_name}
Features: {features}
Target audience: {audience}
Max length: {max_words} words.""",
)

result = g.generate(
    product_name="AIForge",
    features=["plug-and-play AI", "8 modules", "provider-agnostic"],
    audience="developers",
    tone="professional",
    max_words=150,
)
print(result.text)
```

The rendered prompt sent to the LLM would be:
```
Write a professional product description for:
Product: AIForge
Features: plug-and-play AI, 8 modules, provider-agnostic
Target audience: developers
Max length: 150 words.
```

**Template rules:**
- Use `{variable_name}` syntax for placeholders
- Pass matching keyword arguments to `generate()`
- List values are joined with `", "`
- Template mode and `prompt` are mutually exclusive — use one or the other

---

## Freeform Mode

When no template is set, pass a freeform `prompt` directly:

```python
g = Generator(guardrails=["no_pii", "professional_tone"])

result = g.generate(prompt="Write a follow-up email for a customer who complained about shipping delays")
print(result.text)
```

---

## Rewrite Mode

The Generator also supports rewriting existing text. Use `rewrite()` to transform text while preserving meaning:

```python
g = Generator()

# Basic rewrite with instructions
result = g.rewrite(
    "The product is good. You should buy it.",
    instructions="Make it more compelling and professional",
)
print(result.text)

# Tone control
result = g.rewrite(
    "Dear Sir, I wish to formally request...",
    tone="casual",
)

# Reading level adjustment
result = g.rewrite(
    "The mitochondria is the powerhouse of the cell...",
    reading_level="5th grade",
)

# Combined
result = g.rewrite(
    original_text,
    instructions="Shorten to 2 paragraphs",
    tone="formal",
    reading_level="college",
)
```

### `rewrite(text, *, instructions=None, tone=None, reading_level=None, system_prompt=None) → GenerateResponse`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Original text to rewrite. Must not be empty (raises `ValueError`). |
| `instructions` | `str \| None` | `None` | Free-form rewriting directives. |
| `tone` | `str \| None` | `None` | Target tone (e.g. `"formal"`, `"casual"`, `"friendly"`). |
| `reading_level` | `str \| None` | `None` | Target reading level (e.g. `"5th grade"`, `"college"`). |
| `system_prompt` | `str \| None` | `None` | Override the system prompt for this call. |

`arewrite()` is the async equivalent.

Rewrite mode uses the same guardrail loop as `generate()`, so any configured guardrails are enforced on the rewritten output.

If none of `instructions`, `tone`, or `reading_level` are provided, the rewriter defaults to "Improve clarity and readability".

### CLI

```bash
aiforge rewrite --text "The product is good." --instructions "Make it professional" --tone formal
aiforge rewrite --file draft.txt --reading-level "5th grade"
```

### API

```
POST /generate/rewrite
```

```json
{
  "text": "The product is good. You should buy it.",
  "instructions": "Make it more compelling",
  "tone": "professional",
  "reading_level": null,
  "guardrails": ["professional_tone"]
}
```

---

## Guardrails

Guardrails are enforced **after** generation. If any guardrail is violated, the Generator retries with the violation feedback included in the prompt (up to 2 retries, 3 total attempts).

### Built-in Guardrails

| Guardrail | Detection | What It Catches | Cost |
|-----------|-----------|----------------|------|
| `no_pii` | Regex via Moderator | Personal data (emails, phones, SSNs, etc.) | ~0ms (no LLM call) |
| `no_competitor_mentions` | Regex (case-insensitive) | Competitor names from the `competitors` list | ~0ms |
| `no_pricing` | Regex patterns | Currency amounts like `$99`, `€50`, `£25`, `100 USD` | ~0ms |
| `professional_tone` | LLM classification | Slang, informal language, offensive content | ~200ms + ~$0.0001 |

### Using Built-in Guardrails

```python
# PII protection
g = Generator(guardrails=["no_pii"])
result = g.generate(prompt="Write a welcome email for a new user")
# If the LLM includes "john@email.com" or "555-0123", it retries

# Competitor protection
g = Generator(
    guardrails=["no_competitor_mentions"],
    competitors=["Acme AI", "RivalCorp", "CompetitorX"],
)
result = g.generate(prompt="Write a product comparison")
# If the LLM mentions any competitor by name, it retries

# Pricing protection
g = Generator(guardrails=["no_pricing"])
result = g.generate(prompt="Write a product overview")
# If the LLM includes "$99.99" or "€50", it retries

# Professional tone
g = Generator(guardrails=["professional_tone"])
result = g.generate(prompt="Write a client communication")
# Uses a separate LLM call to check tone; retries if unprofessional

# Multiple guardrails
g = Generator(
    guardrails=["no_pii", "no_competitor_mentions", "professional_tone"],
    competitors=["Acme AI"],
)
```

### Guardrail Retry Flow

```
Attempt 1: Generate → check guardrails → ✗ (violated "no_pii")
Attempt 2: Regenerate (with violation feedback) → check → ✗ (still violated)
Attempt 3: Regenerate (with feedback) → check → ✓ (passed)
→ Return result with retry_count=2
```

If all 3 attempts are exhausted:
- The last generated text is returned
- `guardrail_flags` lists the violations that remain
- No error is raised

```python
result = g.generate(prompt="Write about our pricing plans")

if result.guardrail_flags:
    print(f"Warning: guardrails still violated: {result.guardrail_flags}")
    print(f"After {result.retry_count} retries")
else:
    print("All guardrails passed!")
```

### Custom Guardrails

Register custom guardrail functions using the `@register_guardrail` decorator:

```python
from aiforge import register_guardrail, GuardrailResult, Generator

@register_guardrail("max_sentences")
def check_max_sentences(text: str) -> GuardrailResult:
    count = text.count(".") + text.count("!") + text.count("?")
    if count > 5:
        return GuardrailResult(passed=False, message=f"Too many sentences: {count}")
    return GuardrailResult(passed=True)

@register_guardrail("no_emojis")
def check_no_emojis(text: str) -> GuardrailResult:
    import re
    emoji_pattern = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]")
    if emoji_pattern.search(text):
        return GuardrailResult(passed=False, message="Contains emojis")
    return GuardrailResult(passed=True)

# Use custom guardrails alongside built-in ones
g = Generator(guardrails=["no_pii", "max_sentences", "no_emojis"])
result = g.generate(prompt="Write a concise product summary")
```

**Custom guardrail rules:**
- Function signature must be `(text: str) -> GuardrailResult`
- Must return `GuardrailResult(passed=True/False, message="...")`
- Registration is thread-safe
- Custom guardrails can be mixed with built-in ones
- Unknown guardrail names raise `ValueError` at check time

---

## Prompt Resolution

System prompts are resolved in this precedence order (highest to lowest):

1. **Per-call override:** `g.generate(prompt=..., system_prompt="...")`
2. **Constructor override:** `Generator(system_prompt="...")`
3. **Config file:** `CUSTOM_PROMPTS.generate.system` in `aiforge.yaml`
4. **Built-in default:** *"You are a versatile content generator. Produce high-quality text that precisely follows the user's instructions. Respond ONLY with the generated content — no preamble, no commentary."*

---

## Instance Properties

After constructing a `Generator`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `g.template` | `str \| None` | Configured template string. |
| `g.guardrails` | `list[str]` | Guardrail names to enforce. |
| `g.max_tokens` | `int` | Max token limit for generation. |
| `g.temperature` | `float` | LLM temperature setting. |
| `g.competitors` | `list[str]` | Competitor names for `no_competitor_mentions`. |
| `g.settings` | `AIForgeSettings` | Resolved configuration. |
| `g.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Product Description Generator

```python
from aiforge import Generator

g = Generator(
    template="""Write a {tone} product description for {product}.

Key features: {features}
Target audience: {audience}
Word limit: {word_limit} words.""",
    guardrails=["no_competitor_mentions", "no_pricing"],
    competitors=["CompetitorA", "RivalCorp"],
    temperature=0.8,
)

result = g.generate(
    product="AIForge SDK",
    features=["8 AI modules", "provider-agnostic", "type-safe responses"],
    audience="Python developers",
    tone="professional",
    word_limit=100,
)

print(result.text)
print(f"Guardrails passed: {not result.guardrail_flags}")
print(f"Cost: ${result.cost:.4f}")
```

### Email Drafter with Safety

```python
from aiforge import Generator

g = Generator(
    guardrails=["no_pii", "professional_tone"],
    max_tokens=512,
)

result = g.generate(
    prompt="Write a follow-up email to a client who reported a billing discrepancy. "
           "Be empathetic and offer to resolve the issue.",
)

print(result.text)
if result.guardrail_flags:
    print(f"⚠ Safety issues: {result.guardrail_flags}")
```

### Batch Content Generation

```python
from aiforge import Generator

g = Generator(
    guardrails=["professional_tone"],
    quality="low",  # cheapest model for bulk work
)

topics = [
    "Write a tagline for a coffee shop",
    "Write a tagline for a bookstore",
    "Write a tagline for a fitness app",
    "Write a tagline for a pet grooming service",
]

batch = g.run_batch(topics, concurrency=4)

for r in batch.results:
    flags = f" [flagged: {r.guardrail_flags}]" if r.guardrail_flags else ""
    print(f"  {r.text}{flags}")

print(f"\nTotal cost: ${batch.total_cost:.4f}")
```

### Custom Guardrail with Pipeline

```python
from aiforge import Generator, Summarizer, register_guardrail, GuardrailResult

@register_guardrail("word_count")
def check_word_count(text: str) -> GuardrailResult:
    count = len(text.split())
    if count > 200:
        return GuardrailResult(passed=False, message=f"Too long: {count} words (max 200)")
    return GuardrailResult(passed=True)

# Step 1: Summarize a document
summarizer = Summarizer()
summary = summarizer.summarize(long_document)

# Step 2: Generate a newsletter blurb from the summary
g = Generator(
    template="Write a newsletter blurb based on this summary: {summary}",
    guardrails=["word_count", "professional_tone"],
)
result = g.generate(summary=summary.to_text())
print(result.text)
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
MODEL: "gpt-4o-mini"
TEMPERATURE: 0.7
```

```python
g = Generator(template="Write a {tone} blog intro about {topic}.")
result = g.generate(tone="casual", topic="AI trends")
print(result.model)  # "gpt-4o-mini"
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Both template and prompt provided, neither available, `max_tokens < 1`, temperature out of range, unknown guardrail name | `g.generate(prompt="x", topic="y")` when template is set |
| `ProviderError` | LLM provider returns an error | API key invalid |
| `RateLimitError` | Provider rate limit exceeded | Too many requests |
| `APITimeoutError` | Provider request timed out | Slow network |

```python
from aiforge.core.types import ProviderError

try:
    result = g.generate(prompt="Write something")
except ValueError as e:
    print(f"Config error: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

**Graceful degradation:** If guardrails are still violated after all retries, the Generator does **not** raise. It returns the last generated text with `guardrail_flags` listing the remaining violations.

---

## File Structure

```
aiforge/modules/generate/
├── __init__.py        # Exports: Generator, register_guardrail, GenerateResponse, GuardrailResult
├── module.py          # Generator class (constructor, generate/agenerate, guardrail loop, retry logic)
├── prompts.py         # Prompt templates (GENERATE_SYSTEM, GENERATE_RETRY, PROFESSIONAL_TONE_CHECK)
├── schemas.py         # Pydantic model (GenerateResponse) + dataclass (GuardrailResult)
└── widget.py          # Streamlit playground widget (template + freeform modes)
```

---

## See Also

- [Quick Start](../quickstart.md) — Getting started with AIForge
- [Moderate](moderate.md) — Content moderation (PII detection used by `no_pii` guardrail)
- [Pipelines](../pipelines.md) — Chain generation with other modules
- [API Server](../api-server.md) — `POST /generate` endpoint
- [Playground](../playground.md) — Interactive generation tab
- [Configuration](../configuration.md) — Full settings reference
