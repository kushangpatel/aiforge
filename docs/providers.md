# Providers

AIForge is provider-agnostic. Swap between OpenAI, Anthropic, Ollama, and
Azure OpenAI with a single config change — zero code changes required. All
four providers implement identical `LLMProvider` and `EmbeddingProvider`
interfaces, and a lazy-loaded registry means optional dependencies are only
imported when you actually use a provider.

---

## Table of Contents

- [Why Provider Abstraction?](#why-provider-abstraction)
- [Supported Providers](#supported-providers)
- [Quick Start — Swapping Providers](#quick-start--swapping-providers)
  - [Via YAML](#via-yaml)
  - [Via Environment Variables](#via-environment-variables)
  - [Via Code](#via-code)
- [OpenAI](#openai)
  - [Configuration](#openai-configuration)
  - [Supported Models](#openai-supported-models)
  - [Pricing (USD per 1M tokens)](#openai-pricing)
- [Anthropic](#anthropic)
  - [Configuration](#anthropic-configuration)
  - [Supported Models](#anthropic-supported-models)
  - [Pricing (USD per 1M tokens)](#anthropic-pricing)
  - [Embeddings via Voyage AI](#embeddings-via-voyage-ai)
- [Ollama (Local)](#ollama-local)
  - [Prerequisites](#ollama-prerequisites)
  - [Configuration](#ollama-configuration)
  - [Supported Models](#ollama-supported-models)
  - [Token Counting Fallback](#token-counting-fallback)
- [Azure OpenAI](#azure-openai)
  - [Configuration](#azure-openai-configuration)
  - [Deployment Names vs Model Names](#deployment-names-vs-model-names)
- [Provider-Embedding Compatibility](#provider-embedding-compatibility)
  - [Compatibility Map](#compatibility-map)
  - [Mismatch Error Example](#mismatch-error-example)
- [Provider Interfaces](#provider-interfaces)
  - [LLMProvider](#llmprovider)
  - [EmbeddingProvider](#embeddingprovider)
  - [CompletionResult](#completionresult)
- [Provider Registry](#provider-registry)
  - [Built-in Registry](#built-in-registry)
  - [Listing Providers](#listing-providers)
  - [Custom Provider Registration](#custom-provider-registration)
- [Model Router](#model-router)
  - [How Routing Works](#how-routing-works)
  - [Quality Tiers](#quality-tiers)
  - [Built-in Model Catalogue](#built-in-model-catalogue)
  - [Usage Examples](#model-router-usage-examples)
- [Error Handling & Retries](#error-handling--retries)
  - [Error Hierarchy](#error-hierarchy)
  - [Retry Behaviour](#retry-behaviour)
  - [Provider-Specific Error Mapping](#provider-specific-error-mapping)
- [Rate Limiting](#rate-limiting)
- [Optional Dependencies](#optional-dependencies)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Why Provider Abstraction?

| Scenario | How AIForge helps |
|----------|------------------|
| Vendor lock-in risk | Change `provider` in YAML — all module code stays identical |
| Cost optimisation | Use the Model Router to auto-select the cheapest model per quality tier |
| Local development | Switch to Ollama for zero-cost, offline inference |
| Enterprise compliance | Use Azure OpenAI deployments for data-residency requirements |
| Embedding mismatch | Startup validation catches provider/embedding incompatibilities |
| Transient failures | Built-in retry with exponential backoff across all providers |
| Rate limit protection | Token-bucket rate limiter blocks (not fails) when limits are hit |

---

## Supported Providers

| Provider | Install Extra | LLM Models | Embedding Models | Local? | API Keys Required |
|----------|--------------|-----------|-----------------|--------|-------------------|
| **OpenAI** | *(included by default)* | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini, o3 | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 | No | `OPENAI_API_KEY` |
| **Anthropic** | `pip install "aiforge[anthropicai]"` | claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku, claude-3-haiku, claude-3-opus | voyage-3, voyage-3-lite, voyage-large-2, voyage-code-2 | No | `ANTHROPIC_API_KEY` + `VOYAGE_API_KEY` |
| **Ollama** | `pip install "aiforge[local]"` | llama3, llama3.1, llama3.2, llama3.3, mistral, mistral-nemo, codellama, phi-3, phi-4, gemma2, qwen2.5 | nomic-embed-text, mxbai-embed-large, all-minilm | Yes | None |
| **Azure OpenAI** | *(shares langchain-openai)* | Same as OpenAI (via deployment names) | Same as OpenAI (via deployment names) | No | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |

---

## Quick Start — Swapping Providers

### Via YAML

```yaml
# aiforge.yaml — change these two lines, nothing else
provider: anthropic
model: claude-3-5-sonnet-20241022
```

### Via Environment Variables

```bash
export AIFORGE_PROVIDER=anthropic
export AIFORGE_MODEL=claude-3-5-sonnet-20241022
```

### Via Code

```python
from aiforge import Summarizer

# Provider override at module level — no global config change needed
result = Summarizer(provider="anthropic", model="claude-3-5-sonnet-20241022").run(
    "Long document text here..."
)
```

All three approaches produce identical results. Zero code changes needed in
consuming applications.

---

## OpenAI

The default provider. Included in the base install — no extras required.

### OpenAI Configuration

```bash
# .env or shell
AIFORGE_PROVIDER=openai
AIFORGE_MODEL=gpt-4o
AIFORGE_OPENAI_API_KEY=sk-...
AIFORGE_EMBEDDING_MODEL=text-embedding-3-small
AIFORGE_TEMPERATURE=0.0
AIFORGE_MAX_TOKENS=            # Optional — omit for model default
```

```yaml
# aiforge.yaml
provider: openai
model: gpt-4o
embedding_model: text-embedding-3-small
temperature: 0.0
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `AIFORGE_MODEL` | No | `gpt-4o` | LLM model identifier |
| `AIFORGE_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `AIFORGE_TEMPERATURE` | No | `0.0` | Sampling temperature (0.0–2.0) |
| `AIFORGE_MAX_TOKENS` | No | `None` | Hard cap on output tokens |

### OpenAI Supported Models

**LLM:**

| Model | Quality Tier | Context Window | Input / 1M | Output / 1M |
|-------|-------------|----------------|------------|-------------|
| gpt-4o | High | 128,000 | $5.00 | $15.00 |
| gpt-4o-mini | Medium | 128,000 | $0.15 | $0.60 |
| gpt-4-turbo | High | 128,000 | $10.00 | $30.00 |
| gpt-3.5-turbo | Low | 16,385 | $0.50 | $1.50 |
| o1 | High | 200,000 | $15.00 | $60.00 |
| o1-mini | Medium | 128,000 | $3.00 | $12.00 |
| o3-mini | Medium | 200,000 | $1.10 | $4.40 |
| o3 | High | 200,000 | $10.00 | $40.00 |

**Embeddings:**

| Model | Dimensions | Input / 1M |
|-------|-----------|------------|
| text-embedding-3-small | 1,536 | $0.02 |
| text-embedding-3-large | 3,072 | $0.13 |
| text-embedding-ada-002 | 1,536 | $0.10 |

---

## Anthropic

Requires the `anthropicai` extra.

```bash
pip install "aiforge[anthropicai]"
```

This installs `langchain-anthropic` (LLM) and `langchain-voyageai` (embeddings).

### Anthropic Configuration

```bash
AIFORGE_PROVIDER=anthropic
AIFORGE_MODEL=claude-3-5-sonnet-20241022
AIFORGE_ANTHROPIC_API_KEY=sk-ant-...
AIFORGE_VOYAGE_API_KEY=pa-...            # Required for embeddings
AIFORGE_EMBEDDING_MODEL=voyage-3
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `AIFORGE_VOYAGE_API_KEY` | Yes (for embeddings) | — | Voyage AI key for embeddings |
| `AIFORGE_MODEL` | No | `claude-3-5-sonnet-20241022` | LLM model |
| `AIFORGE_EMBEDDING_MODEL` | No | `voyage-3` | Embedding model |
| `AIFORGE_MAX_TOKENS` | No | `4096` | Default 4096 if not explicitly set |

### Anthropic Supported Models

**LLM:**

| Model | Quality Tier | Context Window | Input / 1M | Output / 1M |
|-------|-------------|----------------|------------|-------------|
| claude-3-7-sonnet-20250219 | High | 200,000 | $3.00 | $15.00 |
| claude-3-5-sonnet-20241022 | High | 200,000 | $3.00 | $15.00 |
| claude-3-5-haiku-20241022 | Medium | 200,000 | $0.80 | $4.00 |
| claude-3-haiku-20240307 | Medium | 200,000 | $0.25 | $1.25 |
| claude-3-opus-20240229 | High | 200,000 | $15.00 | $75.00 |

### Embeddings via Voyage AI

Anthropic has **no native embedding API**. When you select `provider=anthropic`,
AIForge automatically routes embedding calls to **Voyage AI**. You must set
both API keys:

```bash
AIFORGE_ANTHROPIC_API_KEY=sk-ant-...    # For LLM calls
AIFORGE_VOYAGE_API_KEY=pa-...           # For embedding calls
```

**Voyage AI Embedding Models:**

| Model | Input / 1M |
|-------|------------|
| voyage-3 | $0.06 |
| voyage-3-lite | $0.02 |
| voyage-large-2 | $0.12 |
| voyage-code-2 | $0.12 |

---

## Ollama (Local)

Zero-cost, fully offline inference. Requires a running Ollama server.

```bash
pip install "aiforge[local]"
```

### Ollama Prerequisites

1. Install Ollama: https://ollama.ai
2. Pull models before use:

```bash
ollama pull llama3            # LLM
ollama pull nomic-embed-text  # Embeddings
```

3. Verify the server is running:

```bash
curl http://localhost:11434/api/tags
```

### Ollama Configuration

```bash
AIFORGE_PROVIDER=ollama
AIFORGE_MODEL=llama3
AIFORGE_EMBEDDING_MODEL=nomic-embed-text
AIFORGE_OLLAMA_BASE_URL=http://localhost:11434   # Default
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server endpoint |
| `AIFORGE_MODEL` | No | `llama3` | LLM model (must be pulled) |
| `AIFORGE_EMBEDDING_MODEL` | No | `nomic-embed-text` | Embedding model (must be pulled) |

### Ollama Supported Models

**LLM:**

| Model | Quality Tier | Context Window | Cost |
|-------|-------------|----------------|------|
| llama3 | Medium | 8,192 | Free |
| llama3.1 | Medium | 131,072 | Free |
| llama3.2 | Low | 131,072 | Free |
| llama3.3 | Medium | 131,072 | Free |
| mistral | Medium | 32,768 | Free |
| mistral-nemo | Medium | 128,000 | Free |
| codellama | Low | 16,384 | Free |
| phi-3 | Low | 128,000 | Free |
| phi-4 | Medium | 16,384 | Free |
| gemma2 | Medium | 8,192 | Free |
| qwen2.5 | Medium | 131,072 | Free |

**Embeddings:**

| Model | Cost |
|-------|------|
| nomic-embed-text | Free |
| mxbai-embed-large | Free |
| all-minilm | Free |

> Any model available in your Ollama installation can be used — the table
> above lists the pre-configured models in AIForge's pricing/context-window
> registry. Unknown models default to 8,192 context and $0 cost.

### Token Counting Fallback

Ollama doesn't reliably report token usage. AIForge uses a three-tier
fallback:

1. `response.usage_metadata` (if the Ollama version provides it)
2. `tiktoken` cl100k_base encoding estimate (tiktoken is a core dependency)
3. Word-count heuristic (words × 1.3) as a last resort

This ensures cost tracking and rate limiting still function with local models.

---

## Azure OpenAI

Uses the same `langchain-openai` package as the OpenAI provider (no extra
install). Azure uses **deployment names** instead of model identifiers.

### Azure OpenAI Configuration

```bash
AIFORGE_PROVIDER=azure
AIFORGE_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AIFORGE_AZURE_OPENAI_API_KEY=...
AIFORGE_AZURE_OPENAI_DEPLOYMENT=my-gpt4o       # Optional — falls back to MODEL
AIFORGE_AZURE_OPENAI_API_VERSION=2024-02-01     # Default
AIFORGE_MODEL=gpt-4o                            # Used if DEPLOYMENT not set
AIFORGE_EMBEDDING_MODEL=my-embedding-deployment
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_AZURE_OPENAI_ENDPOINT` | Yes | — | Azure resource URL |
| `AIFORGE_AZURE_OPENAI_API_KEY` | Yes | — | Azure API key |
| `AIFORGE_AZURE_OPENAI_DEPLOYMENT` | No | Falls back to `MODEL` | LLM deployment name |
| `AIFORGE_AZURE_OPENAI_API_VERSION` | No | `2024-02-01` | API version |
| `AIFORGE_EMBEDDING_MODEL` | No | — | Must be a deployed embedding model name |

> If `AZURE_OPENAI_ENDPOINT` is not set, AIForge raises a `ConfigError` at
> startup with a clear message.

### Deployment Names vs Model Names

Azure OpenAI uses deployment names that you define when deploying a model in
your Azure resource. The deployment name can be anything — it doesn't have to
match the underlying model name:

```bash
# Azure portal: deployed "gpt-4o" as "my-production-model"
AIFORGE_AZURE_OPENAI_DEPLOYMENT=my-production-model
```

The same applies to embeddings — set `EMBEDDING_MODEL` to the **deployment
name**, not the base model name.

---

## Provider-Embedding Compatibility

Not all LLM providers include their own embedding API. AIForge validates
provider/embedding combinations at startup and raises `ConfigError` with a
helpful message if they don't match.

### Compatibility Map

| Provider | Compatible Embedding Models | Notes |
|----------|---------------------------|-------|
| `openai` | text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 | Both from OpenAI — works out of the box |
| `anthropic` | voyage-3, voyage-3-lite, voyage-large-2, voyage-code-2 | Auto-routed to Voyage AI — requires `VOYAGE_API_KEY` |
| `ollama` | nomic-embed-text, mxbai-embed-large, all-minilm | Must be pulled locally before use |
| `azure` | *(Any deployed embedding model)* | Must match a deployed model in your Azure resource |

### Mismatch Error Example

Setting `provider=ollama` with `embedding_model=text-embedding-3-small`:

```
ConfigError: Embedding model 'text-embedding-3-small' is not available
with provider 'ollama'.
Available Ollama embedding models: nomic-embed-text, mxbai-embed-large, all-minilm.
Set AIFORGE_EMBEDDING_MODEL to a compatible model, or run:
  ollama pull nomic-embed-text
```

---

## Provider Interfaces

All providers implement two abstract base classes defined in
`aiforge/core/provider.py`.

### LLMProvider

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict[str, str]], **kwargs) -> CompletionResult:
        """Synchronous completion."""

    @abstractmethod
    async def complete_async(self, messages: list[dict[str, str]], **kwargs) -> CompletionResult:
        """Asynchronous completion."""

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Token-by-token streaming."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""
```

Messages use the OpenAI-style format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarise this document..."},
]
```

All providers internally convert these to LangChain message objects via
`to_lc_messages()` (SystemMessage, HumanMessage, AIMessage).

### EmbeddingProvider

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts."""

    @abstractmethod
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    @abstractmethod
    async def embed_query_async(self, text: str) -> list[float]:
        """Async single-query embedding."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured embedding model identifier."""
```

### CompletionResult

Returned by every `complete()` / `complete_async()` call:

```python
class CompletionResult(BaseModel):
    text: str                           # Generated text content
    input_tokens: int                   # Prompt tokens consumed
    output_tokens: int                  # Completion tokens generated
    model: str                          # Model identifier (e.g., "gpt-4o-2024-08-06")
    finish_reason: str                  # "stop" | "length" | "tool_calls" | "content_filter"
    latency_ms: float                   # Wall-clock time for this call
    raw_response: dict | None = None    # Provider-specific raw response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The generated content |
| `input_tokens` | `int` | Tokens consumed by the prompt |
| `output_tokens` | `int` | Tokens generated in the response |
| `model` | `str` | Exact model version used |
| `finish_reason` | `str` | Why generation stopped |
| `latency_ms` | `float` | Wall-clock time in milliseconds |
| `raw_response` | `dict \| None` | Full provider-specific payload |
| `total_tokens` | `int` (property) | `input_tokens + output_tokens` |

---

## Provider Registry

Providers are managed through a lazy-loaded registry that avoids importing
optional dependencies until they are actually needed.

### Built-in Registry

```python
# aiforge/providers/__init__.py (simplified)
_REGISTRY = {
    "openai":    (OpenAIProvider,    OpenAIEmbeddingProvider),
    "anthropic": (AnthropicProvider, AnthropicEmbeddingProvider),   # if langchain-anthropic installed
    "ollama":    (OllamaProvider,    OllamaEmbeddingProvider),      # if langchain-ollama installed
    "azure":     (AzureProvider,     AzureEmbeddingProvider),       # shares langchain-openai
}
```

The registry is populated lazily by `_ensure_registered()` on first access.
Providers whose dependencies are missing are silently skipped.

### Listing Providers

```python
from aiforge.providers import list_providers

print(list_providers())
# ["openai", "anthropic", "ollama", "azure"]  — only those with deps installed
```

### Custom Provider Registration

Register your own LLM and embedding provider at runtime:

```python
from aiforge.core.provider import LLMProvider, EmbeddingProvider, CompletionResult
from aiforge.providers import register_provider


class GoogleProvider(LLMProvider):
    def __init__(self, settings):
        # Initialize with your Gemini client
        ...

    def complete(self, messages, **kwargs) -> CompletionResult:
        ...

    async def complete_async(self, messages, **kwargs) -> CompletionResult:
        ...

    async def stream(self, messages, **kwargs):
        ...

    @property
    def model_name(self) -> str:
        return "gemini-pro"


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings):
        ...

    def embed(self, texts):
        ...

    async def embed_async(self, texts):
        ...

    def embed_query(self, text):
        ...

    async def embed_query_async(self, text):
        ...

    @property
    def model_name(self) -> str:
        return "text-embedding-004"


# Register — both classes receive AIForgeSettings in __init__
register_provider("google", GoogleProvider, GoogleEmbeddingProvider)

# Now usable everywhere
from aiforge import Summarizer
s = Summarizer(provider="google", model="gemini-pro")
result = s.run("Your text here...")
```

### Provider Instantiation Flow

```
User code: Summarizer(provider="openai", model="gpt-4o")
  ↓
AIForgeSettings resolved (YAML → .env → env vars → code overrides)
  ↓
get_provider(settings)  [aiforge/providers/__init__.py]
  ↓
_ensure_registered()  — lazy-load built-in providers
  ↓
_REGISTRY["openai"] → (OpenAIProvider, OpenAIEmbeddingProvider)
  ↓
OpenAIProvider(settings) → configured with API key, model, temperature
  ↓
LLMProvider instance ready for .complete() / .stream()
```

---

## Model Router

The Model Router automatically selects the cheapest model that meets a given
quality tier. It enables cost-optimised workflows where you specify *what
quality you need*, not *which model to use*.

### How Routing Works

1. Filter candidate models by **quality tier** (high, medium, low)
2. Filter by **provider** (if explicitly requested)
3. Filter by **available providers** (those with installed dependencies)
4. Filter by **minimum context window** (if specified)
5. Sort by: `prefer_local` (Ollama first), then by input cost ascending
6. Return the **cheapest qualifying model**

### Quality Tiers

| Tier | When to use | Example models |
|------|-------------|----------------|
| `high` | Complex reasoning, important outputs | gpt-4o, claude-3-5-sonnet, claude-3-opus |
| `medium` | General tasks, good quality/cost balance | gpt-4o-mini, claude-3-haiku, llama3.1 |
| `low` | Simple tasks, maximum throughput | gpt-3.5-turbo, llama3.2 |

### Built-in Model Catalogue

| Model | Provider | Tier | Input / 1M | Output / 1M | Context |
|-------|----------|------|------------|-------------|---------|
| gpt-4o | openai | high | $5.00 | $15.00 | 128K |
| gpt-4o-mini | openai | medium | $0.15 | $0.60 | 128K |
| gpt-3.5-turbo | openai | low | $0.50 | $1.50 | 16K |
| claude-3-5-sonnet-20241022 | anthropic | high | $3.00 | $15.00 | 200K |
| claude-3-haiku-20240307 | anthropic | medium | $0.25 | $1.25 | 200K |
| llama3.1 | ollama | medium | $0.00 | $0.00 | 131K |
| llama3.2 | ollama | low | $0.00 | $0.00 | 131K |

You can extend this with `CUSTOM_MODELS` in configuration.

### Model Router Usage Examples

```python
from aiforge.core.model_router import ModelRouter

router = ModelRouter()

# Cheapest high-quality model
model = router.select(quality="high")
# → "claude-3-5-sonnet-20241022" ($3.00 input — cheaper than gpt-4o's $5.00)

# Cheapest model from a specific provider
model = router.select(quality="medium", provider="openai")
# → "gpt-4o-mini"

# Prefer local (free) models
model = router.select(quality="medium", prefer_local=True)
# → "llama3.1" (free, Ollama)

# Require large context window
model = router.select(quality="high", min_context=150_000)
# → "claude-3-5-sonnet-20241022" (200K context)

# Estimate cost before calling
cost = router.estimate_cost("gpt-4o", input_tokens=5000, output_tokens=1000)
# → 0.04  (USD)

# List all models in a tier
models = router.list_models(quality="high")
```

**Configuration:**

```yaml
# aiforge.yaml
model_router_enabled: true
default_quality: medium
prefer_local: false
custom_models:
  - name: my-fine-tuned-gpt4
    provider: openai
    tier: high
    input_cost_per_m: 6.00
    output_cost_per_m: 18.00
    context_window: 128000
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_ROUTER_ENABLED` | `bool` | `true` | Enable/disable the router |
| `DEFAULT_QUALITY` | `str` | `medium` | Default quality tier |
| `PREFER_LOCAL` | `bool` | `false` | Prefer Ollama/local models |
| `CUSTOM_MODELS` | `list[dict]` | `[]` | Additional models for the catalogue |

---

## Error Handling & Retries

All provider calls are wrapped with retry logic and provider-specific error
mapping. Transient errors are retried automatically; permanent errors
propagate immediately.

### Error Hierarchy

```
AIForgeError (base)
├── ProviderError           ← catch-all for provider failures
│   ├── RateLimitError      ← 429 / rate limit exceeded
│   ├── APITimeoutError     ← request timed out
│   └── APIConnectionError  ← network / connection failure
├── ConfigError             ← invalid configuration
├── ValidationError         ← input validation
├── VectorStoreError        ← vector store operations
└── PipelineError           ← pipeline execution
```

### Retry Behaviour

Retries use **Tenacity** with exponential jitter backoff:

| Decorator | Max Attempts | Min Wait | Max Wait | Retryable Errors |
|-----------|-------------|----------|----------|-----------------|
| `@retry_llm_call()` | 3 | 1.0s | 10.0s | RateLimitError, APITimeoutError, APIConnectionError |
| `@retry_embedding_call()` | 3 | 0.5s | 8.0s | RateLimitError, APITimeoutError, APIConnectionError |

Non-retryable errors (authentication failures, invalid model, validation)
propagate immediately without retry.

### Provider-Specific Error Mapping

Each provider maps native SDK exceptions to the AIForge error hierarchy:

**OpenAI / Azure** (`_wrap_openai_error()`):

| Native Error | Mapped To |
|-------------|-----------|
| RateLimit* | `RateLimitError` |
| Timeout* | `APITimeoutError` |
| APIConnection*, Connection* | `APIConnectionError` |
| Other | `ProviderError` |

**Anthropic** (`_wrap_anthropic_error()`):

| Native Error | Mapped To |
|-------------|-----------|
| RateLimit* | `RateLimitError` |
| Timeout* | `APITimeoutError` |
| Connection* | `APIConnectionError` |
| Other | `ProviderError` |

**Ollama** (`_wrap_ollama_error()`):

| Native Error | Mapped To | Extra Help |
|-------------|-----------|------------|
| Connection*, *refused* | `APIConnectionError` | "Is the Ollama server running?" |
| Other | `ProviderError` | — |

---

## Rate Limiting

AIForge includes a built-in **token-bucket rate limiter** (singleton, thread-safe)
that protects against provider rate limits. When a limit is hit, the limiter
**blocks (sleeps)** until capacity is available — it never raises an error.

| Setting | Default | Description |
|---------|---------|-------------|
| `RATE_LIMIT_RPM` | 500 | Requests per minute |
| `RATE_LIMIT_TPM` | 150,000 | Tokens per minute |

Both buckets use a **fixed 60-second refill window**.

```python
from aiforge.infra.rate_limiter import RateLimiter

limiter = RateLimiter.get_instance(rpm=100, tpm=50_000)
limiter.acquire(tokens=2000)  # Blocks if capacity exhausted
```

For Ollama (local inference), rate limiting still applies but is typically
unnecessary at the default limits. Set higher values or disable via
configuration if local throughput is a bottleneck.

---

## Optional Dependencies

| Extra | Packages Installed | Use Case |
|-------|--------------------|----------|
| *(base)* | langchain-openai, chromadb, tenacity, tiktoken | OpenAI + ChromaDB (default) |
| `anthropicai` | langchain-anthropic, langchain-voyageai | Anthropic LLM + Voyage AI embeddings |
| `local` | langchain-ollama | Ollama local inference |
| `pinecone` | pinecone-client≥3.0.0 | Pinecone vector store |
| `qdrant` | qdrant-client≥1.7.0 | Qdrant vector store |
| `pgvector` | psycopg[binary]≥3.1.0 | PostgreSQL/pgvector |
| `observability` | opentelemetry-api, opentelemetry-sdk, langsmith | Tracing & monitoring |
| `server` | fastapi, uvicorn | REST API server |
| `playground` | streamlit≥1.40.0 | Web UI playground |
| `docs` | pypdf, python-docx | Document processing |
| `dev` | pytest, pytest-asyncio, mypy, ruff, httpx | Testing & development |
| `all` | Everything above | Full install |

**Install examples:**

```bash
pip install aiforge                                      # Core + OpenAI + ChromaDB
pip install "aiforge[anthropicai]"                       # + Anthropic + Voyage
pip install "aiforge[local]"                             # + Ollama
pip install "aiforge[anthropicai,pgvector,server]"       # Custom combo
pip install "aiforge[all]"                               # Everything
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ConfigError: provider 'anthropic' not registered` | Missing dependency | `pip install "aiforge[anthropicai]"` |
| `ConfigError: provider 'ollama' not registered` | Missing dependency | `pip install "aiforge[local]"` |
| `ConfigError: Embedding model '...' not available with provider '...'` | Provider/embedding mismatch | Set `EMBEDDING_MODEL` to a compatible model (see [Compatibility Map](#compatibility-map)) |
| `ConfigError: AZURE_OPENAI_ENDPOINT is required` | Azure endpoint not set | Set `AIFORGE_AZURE_OPENAI_ENDPOINT` |
| `APIConnectionError: Is the Ollama server running?` | Ollama not running | Start Ollama: `ollama serve` |
| `RateLimitError` after retries | Provider rate limit exceeded | Reduce `RATE_LIMIT_RPM`/`RATE_LIMIT_TPM`, or upgrade provider plan |
| Token counts are 0 with Ollama | Ollama version doesn't report usage | Normal — AIForge falls back to tiktoken estimates |
| `ProviderError: model not found` (Ollama) | Model not pulled | `ollama pull <model-name>` |
| Embeddings fail with Anthropic | Missing Voyage API key | Set `AIFORGE_VOYAGE_API_KEY` (separate from Anthropic key) |
| Cost tracking shows $0 for all calls | Using Ollama | Expected — local inference is free |
| Model Router returns unexpected model | Custom models not registered | Add models to `custom_models` in config |
| Slow first request | Provider lazy-loading | Expected on first call — subsequent calls reuse the cached provider |

---

## See Also

- [Configuration](configuration.md) — Full configuration reference
- [Cost Tracking](cost-tracking.md) — Pricing tables and cost tracking details
- [Model Router usage in Pipelines](pipelines.md) — Using quality tiers in pipeline steps
- [API Server](api-server.md) — REST API (passes provider config through)
- [Playground](playground.md) — Interactive UI for testing providers
