# Module: Vision

`aiforge.modules.vision` — Image analysis using multimodal LLMs. Supports description, visual Q&A, text extraction, and image classification.

---

## Installation

The vision module is included with the base `aiforge` install. No extras required. A vision-capable model (e.g. GPT-4o, Claude 3) is needed at runtime.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import Vision
from aiforge import VisionResponse   # response type
```

---

## Quick Start

```python
from aiforge import Vision

# Describe an image from a URL
v = Vision(task="describe")
result = v.analyze(source="https://example.com/photo.jpg")
print(result.text)

# Describe a local file
result = v.analyze(source="/path/to/image.png")
print(result.text)

# Ask a question about an image
v = Vision(task="qa")
result = v.analyze(source="photo.jpg", question="How many people are in the image?")
print(result.text)
```

---

## Constructor

```python
Vision(
    task: Literal["describe", "qa", "extract", "classify"] = "describe",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"describe"` | Analysis task. See Tasks below. |
| `temperature` | `float` | `0.3` | Sampling temperature. |
| `max_tokens` | `int` | `1024` | Maximum output tokens. |

---

## Tasks

| Task | Description | Extra Parameters |
|------|-------------|------------------|
| `describe` | Generate a detailed description of the image. | `focus` — optional area to focus on. |
| `qa` | Answer a question about the image. | `question` — required. |
| `extract` | Extract text/data visible in the image. | — |
| `classify` | Classify the image into provided labels. | `labels` — list of labels. |

---

## Methods

### `analyze(source, *, question=None, focus=None, labels=None, system_prompt=None) → VisionResponse`

Analyze an image synchronously.

```python
# Describe with focus
v = Vision(task="describe")
result = v.analyze(source="building.jpg", focus="architectural style")
print(result.text)

# Extract text from a screenshot
v = Vision(task="extract")
result = v.analyze(source="screenshot.png")
print(result.text)

# Classify
v = Vision(task="classify")
result = v.analyze(source="animal.jpg", labels=["cat", "dog", "bird", "fish"])
print(result.text)
```

### `aanalyze(source, *, question=None, focus=None, labels=None, system_prompt=None) → VisionResponse`

Async version of `analyze()`.

```python
import asyncio
from aiforge import Vision

async def main():
    v = Vision(task="qa")
    result = await v.aanalyze("chart.png", question="What's the highest value?")
    print(result.text)

asyncio.run(main())
```

### Pipeline Interface

`run(text)` and `arun(text)` treat the input `text` as an image source (path or URL).

---

## Image Sources

The Vision module accepts:

| Source | Format |
|--------|--------|
| **URL** | `https://example.com/image.jpg` |
| **Local file** | `/path/to/image.png` or `./image.jpg` |

Supported formats: JPEG, PNG, GIF, WebP.

---

## Response Schema

`VisionResponse` inherits from `ModuleResponse` and adds:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Analysis result text. |
| `task` | `str` | Task that was performed. |
| `image_source` | `str` | The image source that was analyzed. |

Plus inherited: `cost`, `input_tokens`, `output_tokens`, `model`, `trace_id`, `latency_ms`.

---

## CLI

```bash
# Describe an image
aiforge vision --source https://example.com/photo.jpg

# Ask a question
aiforge vision --source photo.jpg --task qa --question "What color is the car?"

# Extract text
aiforge vision --source screenshot.png --task extract

# Classify
aiforge vision --source animal.jpg --task classify --labels "cat,dog,bird"
```

---

## API

```
POST /vision
```

```json
{
  "source": "https://example.com/photo.jpg",
  "task": "describe",
  "question": null,
  "focus": null,
  "labels": []
}
```

---

## Custom Prompts

Override prompts via `aiforge.yaml`:

```yaml
custom_prompts:
  vision:
    system: "You are a medical imaging specialist..."
```

Or per-call:

```python
result = v.analyze("xray.jpg", system_prompt="Analyze this medical image for abnormalities.")
```
