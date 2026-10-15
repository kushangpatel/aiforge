# Module: Translate

`aiforge.modules.translate` — Multilingual text translation with auto language detection, formality control, and domain-specific hints.

---

## Installation

The translate module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import Translator
from aiforge import TranslateResponse   # response type
```

---

## Quick Start

```python
from aiforge import Translator

# Basic translation (auto-detect source language)
t = Translator(target_language="Spanish")
result = t.translate("Hello, how are you?")
print(result.text)              # "Hola, ¿cómo estás?"
print(result.source_language)   # "English"

# Explicit source language
t = Translator(target_language="French", source_language="English")
result = t.translate("Good morning!")
print(result.text)              # "Bonjour !"
```

---

## Constructor

```python
Translator(
    target_language: str = "English",
    source_language: str = "auto",
    formality: Literal["formal", "informal", "neutral"] = "neutral",
    domain: str | None = None,
    temperature: float = 0.3,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_language` | `str` | `"English"` | Language to translate into. |
| `source_language` | `str` | `"auto"` | Source language. `"auto"` triggers auto-detection. |
| `formality` | `str` | `"neutral"` | Tone: `"formal"`, `"informal"`, or `"neutral"`. |
| `domain` | `str \| None` | `None` | Domain hint (e.g. `"legal"`, `"medical"`, `"technical"`). |
| `temperature` | `float` | `0.3` | Sampling temperature. |

---

## Methods

### `translate(text, *, system_prompt=None) → TranslateResponse`

Translate text synchronously.

```python
t = Translator(target_language="German", formality="formal")
result = t.translate("Can you help me with this?")
print(result.text)              # Formal German translation
print(result.formality)         # "formal"
```

### `atranslate(text, *, system_prompt=None) → TranslateResponse`

Async version of `translate()`.

```python
import asyncio
from aiforge import Translator

async def main():
    t = Translator(target_language="Japanese")
    result = await t.atranslate("Welcome to our platform!")
    print(result.text)

asyncio.run(main())
```

### Pipeline Interface

`run(text)` and `arun(text)` are also available for pipeline chaining. The translated text is passed downstream via `to_text()`.

---

## Auto Language Detection

When `source_language="auto"` (the default), the Translator makes a preliminary LLM call to detect the source language before translating. The detected language is included in the response.

```python
t = Translator(target_language="English")
result = t.translate("Bonjour le monde!")
print(result.source_language)   # "French"
print(result.text)              # "Hello world!"
```

---

## Domain Hints

Provide a domain to get terminology-aware translations:

```python
t = Translator(target_language="Spanish", domain="medical")
result = t.translate("The patient presents with acute myocardial infarction.")
```

---

## Formality Control

```python
# Informal
t = Translator(target_language="German", formality="informal")
result = t.translate("Could you send me the report?")

# Formal
t = Translator(target_language="German", formality="formal")
result = t.translate("Could you send me the report?")
```

---

## Response Schema

`TranslateResponse` inherits from `ModuleResponse` and adds:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Translated text. |
| `source_language` | `str` | Detected or specified source language. |
| `target_language` | `str` | Target language. |
| `formality` | `str` | Formality level used. |

Plus inherited: `cost`, `input_tokens`, `output_tokens`, `model`, `trace_id`, `latency_ms`.

---

## CLI

```bash
# Basic translation
aiforge translate --text "Hello world" --target Spanish

# From file with formality
aiforge translate --file input.txt --target German --formality formal

# With domain hint
aiforge translate --text "The patient has hypertension." --target French --domain medical
```

---

## API

```
POST /translate
```

```json
{
  "text": "Hello, how are you?",
  "target_language": "Spanish",
  "source_language": "auto",
  "formality": "neutral",
  "domain": null
}
```

---

## Custom Prompts

Override the system prompt via `aiforge.yaml`:

```yaml
custom_prompts:
  translate:
    system: "You are a certified translator specializing in legal documents..."
```

Or per-call:

```python
result = t.translate("Hello", system_prompt="Translate casually with slang.")
```
