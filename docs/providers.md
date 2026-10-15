# Providers

AIForge is provider-agnostic. Swap between OpenAI, Anthropic, Ollama,
Azure OpenAI, Google Gemini, Amazon Bedrock and Mistral AI with a single config
change — zero code changes required. All seven providers implement identical
`LLMProvider` and `EmbeddingProvider` interfaces, and a lazy-loaded registry means optional dependencies are only
imported when you actually use a provider.

---

## Table of Contents

- [Providers](#providers)
  - [Table of Contents](#table-of-contents)
  - [Why Provider Abstraction?](#why-provider-abstraction)
  - [Supported Providers](#supported-providers)
  - [Quick Start — Swapping Providers](#quick-start--swapping-providers)
    - [Via YAML](#via-yaml)
    - [Via Environment Variables](#via-environment-variables)
    - [Via Code](#via-code)
  - [OpenAI](#openai)
    - [OpenAI Configuration](#openai-configuration)
    - [OpenAI Supported Models](#openai-supported-models)
  - [Anthropic](#anthropic)
    - [Anthropic Configuration](#anthropic-configuration)
    - [Anthropic Supported Models](#anthropic-supported-models)
    - [Embeddings via Voyage AI](#embeddings-via-voyage-ai)
  - [Ollama (Local)](#ollama-local)
    - [Ollama Prerequisites](#ollama-prerequisites)
    - [Ollama Configuration](#ollama-configuration)
    - [Ollama Supported Models](#ollama-supported-models)
    - [Token Counting Fallback](#token-counting-fallback)
  - [Azure OpenAI](#azure-openai)
    - [Azure OpenAI Configuration](#azure-openai-configuration)
    - [Deployment Names vs Model Names](#deployment-names-vs-model-names)
  - [Google Gemini](#google-gemini)
    - [Gemini Configuration](#gemini-configuration)
    - [Gemini Supported Models](#gemini-supported-models)
  - [Amazon Bedrock](#amazon-bedrock)
    - [Bedrock Configuration](#bedrock-configuration)
    - [Bedrock Supported Models](#bedrock-supported-models)
    - [Bedrock Prompt Caching](#bedrock-prompt-caching)
  - [Mistral AI](#mistral-ai)
    - [Mistral Configuration](#mistral-configuration)
    - [Mistral Supported Models](#mistral-supported-models)
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
    - [Provider Instantiation Flow](#provider-instantiation-flow)
  - [Model Router](#model-router)
    - [How Routing Works](#how-routing-works)
    - [Quality Tiers](#quality-tiers)
    - [Built-in Model Catalogue](#built-in-model-catalogue)
    - [Model Router Usage Examples](#model-router-usage-examples)
  - [Error Handling \& Retries](#error-handling--retries)
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
| **Anthropic** | `pip install "aiforge-kit[anthropicai]"` | claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5, claude-fable-5-1 | voyage-3, voyage-3-lite, voyage-large-2, voyage-code-2 | No | `ANTHROPIC_API_KEY` + `VOYAGE_API_KEY` |
| **Ollama** | `pip install "aiforge-kit[local]"` | llama3, llama3.1, llama3.2, llama3.3, mistral, mistral-nemo, codellama, phi-3, phi-4, gemma2, qwen2.5 | nomic-embed-text, mxbai-embed-large, all-minilm | Yes | None |
| **Azure OpenAI** | *(shares langchain-openai)* | Same as OpenAI (via deployment names) | Same as OpenAI (via deployment names) | No | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| **Google Gemini** | `pip install "aiforge-kit[gemini]"` | gemini-3.8-flash, gemini-3.5-flash-lite, gemini-3.1-pro-preview, gemini-4-argon, gemini-2.5-* | gemini-embedding-2, gemini-embedding-001 | No | `GOOGLE_API_KEY` |
| **Amazon Bedrock** | `pip install "aiforge-kit[bedrock]"` | Claude (anthropic.claude-*), Amazon Nova (amazon.nova-*), any Converse model | amazon.titan-embed-text-v2:0, amazon.titan-embed-text-v1 | No | AWS credential chain + `AWS_REGION` |
| **Mistral AI** | `pip install "aiforge-kit[mistral]"` | mistral-large-4, mistral-medium-3-5, mistral-small-2603, codestral-2508 | mistral-embed, codestral-embed | No | `MISTRAL_API_KEY` |

---

## Quick Start — Swapping Providers

### Via YAML

```yaml
# aiforge.yaml — change these two lines, nothing else
provider: anthropic
model: claude-sonnet-5-5
```

### Via Environment Variables

```bash
export AIFORGE_PROVIDER=anthropic
export AIFORGE_MODEL=claude-sonnet-5-5
```

### Via Code

```python
from aiforge import Summarizer

# Provider override at module level — no global config change needed
result = Summarizer(provider="anthropic", model="claude-sonnet-5-5").run(
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
pip install "aiforge-kit[anthropicai]"
```

This installs `langchain-anthropic` (LLM) and `langchain-voyageai` (embeddings).

### Anthropic Configuration

```bash
AIFORGE_PROVIDER=anthropic
AIFORGE_MODEL=claude-sonnet-5-5
AIFORGE_ANTHROPIC_API_KEY=sk-ant-...
AIFORGE_VOYAGE_API_KEY=pa-...            # Required for embeddings
AIFORGE_EMBEDDING_MODEL=voyage-3
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `AIFORGE_VOYAGE_API_KEY` | Yes (for embeddings) | — | Voyage AI key for embeddings |
| `AIFORGE_MODEL` | No | `claude-sonnet-5-5` | LLM model |
| `AIFORGE_EMBEDDING_MODEL` | No | `voyage-3` | Embedding model |
| `AIFORGE_MAX_TOKENS` | No | `4096` | Default 4096 if not explicitly set |

### Anthropic Supported Models

**LLM:**

| Model | Quality Tier | Context Window | Input / 1M | Output / 1M |
|-------|-------------|----------------|------------|-------------|
| claude-opus-5-5 | High | 1,000,000 | $4.00 | $20.00 |
| claude-sonnet-5-5 | High | 1,000,000 | $2.00 | $10.00 |
| claude-haiku-5-5 | Medium | 1,000,000 | $0.10 | $0.50 |
| claude-fable-5-1 | High | 1,000,000 | $10.00 | $50.00 |

Claude 3.x models are retired. Current Claude models reject `temperature` /
`top_p` / `top_k`; AIForge drops those parameters automatically for them.

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
pip install "aiforge-kit[local]"
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

## Google Gemini

Gemini models through the Gemini Developer API, using `langchain-google-genai`
(`ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings`).

```bash
pip install "aiforge-kit[gemini]"
```

### Gemini Configuration

```bash
AIFORGE_PROVIDER=gemini
AIFORGE_GOOGLE_API_KEY=...                  # or GOOGLE_API_KEY / GEMINI_API_KEY
AIFORGE_MODEL=gemini-3.8-flash              # Optional, this is the default
AIFORGE_EMBEDDING_MODEL=gemini-embedding-2  # Optional, this is the default
```

```yaml
# aiforge.yaml
provider: gemini
model: gemini-3.8-flash
embedding_model: gemini-embedding-2
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_GOOGLE_API_KEY` | Yes | — | Gemini API key from Google AI Studio. Falls back to `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| `AIFORGE_MODEL` | No | `gemini-3.8-flash` | LLM model, used when `MODEL` is not set anywhere |
| `AIFORGE_EMBEDDING_MODEL` | No | `gemini-embedding-2` | Embedding model |
| `AIFORGE_MAX_TOKENS` | No | model default | Sent as `max_output_tokens` |

Google recommends leaving `temperature` at 1.0 for Gemini 3 and later models;
AIForge's default of 0.0 is sent unchanged, so set `temperature: 1.0` if you
see repetitive output.

### Gemini Supported Models

Prices are the Standard paid tier per 1M tokens, checked 2026-10-15
([Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)).
"Cached" is the price of input tokens served from Gemini's implicit cache.

| Model | Context Window | Input / 1M | Cached / 1M | Output / 1M |
|-------|----------------|------------|-------------|-------------|
| gemini-4-argon | 1,048,576 | $2.00 ¹ | $0.10 | $10.00 ¹ |
| gemini-3.1-pro-preview | 1,048,576 | $2.00 ² | $0.20 | $12.00 ² |
| gemini-3.8-flash | 1,048,576 | $0.75 ³ | $0.075 | $3.75 ³ |
| gemini-3.6-flash | 1,048,576 | $0.75 ³ | $0.075 | $3.75 ³ |
| gemini-3.5-flash-lite | 1,048,576 | $0.30 | $0.03 | $2.50 |
| gemini-3.1-flash-lite | 1,048,576 | $0.25 | $0.025 | $1.50 |
| gemini-2.5-pro | 1,048,576 | $1.25 ² | $0.125 | $10.00 ² |
| gemini-2.5-flash | 1,048,576 | $0.30 | $0.03 | $2.50 |
| gemini-2.5-flash-lite | 1,048,576 | $0.10 | $0.01 | $0.40 |

¹ Rises to $4.00 / $20.00 (cached $0.20) on 2026-12-01.
² Prompts up to 200K tokens; longer prompts are billed at a higher rate that
the pricing table does not model.
³ Rises to $1.50 / $7.50 (cached $0.15) on 2027-01-01.

Update the table with `custom_pricing` when prices change (see
[Configuration](configuration.md)).

**Embeddings:**

| Model | Max input | Dimensions | Input / 1M |
|-------|-----------|------------|------------|
| gemini-embedding-2 | 8,192 tokens | 3,072 (128–3,072) | $0.20 (text) |
| gemini-embedding-001 | 2,048 tokens | 3,072 (128–3,072) | not in the pricing table |

Gemini caches repeated prompt prefixes implicitly; hits are reported on
`cache_read_tokens`. `PROMPT_CACHE` has no effect for Gemini.

---

## Amazon Bedrock

Any Bedrock chat model through the Converse API, using `langchain-aws`
(`ChatBedrockConverse` and `BedrockEmbeddings`). Claude and Amazon Nova are
priced out of the box.

```bash
pip install "aiforge-kit[bedrock]"
```

### Bedrock Configuration

AIForge never reads or stores AWS secrets. Credentials come from the standard
AWS chain: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`,
`~/.aws/credentials`, SSO, or an instance / container / Lambda role.

```bash
AIFORGE_PROVIDER=bedrock
AIFORGE_AWS_REGION=us-east-1                            # or the AWS default region
AIFORGE_AWS_PROFILE=my-profile                          # Optional named profile
AIFORGE_MODEL=global.anthropic.claude-sonnet-5-5        # Optional, this is the default
AIFORGE_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0    # Optional, this is the default
```

```yaml
# aiforge.yaml
provider: bedrock
aws_region: us-east-1
model: us.amazon.nova-pro-v1:0
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_AWS_REGION` | No | AWS default region | Bedrock region, e.g. `us-east-1` |
| `AIFORGE_AWS_PROFILE` | No | default profile | Named profile from `~/.aws/config` |
| `AIFORGE_MODEL` | No | `global.anthropic.claude-sonnet-5-5` | Model ID or inference-profile ID |
| `AIFORGE_EMBEDDING_MODEL` | No | `amazon.titan-embed-text-v2:0` | Embedding model ID |

`MODEL` is a Bedrock model ID or inference-profile ID. Recent Claude models
are only served through inference profiles: `global.` (any commercial region,
cheapest) or a geography prefix such as `us.`, `eu.`, `jp.` or `apac.`
(data stays in that geography, 10% surcharge for Claude). Current Claude
models reject `temperature` / `top_p` / `top_k` on Bedrock too; AIForge drops
them automatically.

### Bedrock Supported Models

On-demand prices for US East (N. Virginia), checked 2026-10-15 through the
AWS Price List API ([Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)).
The pricing table understands every ID form: `us.anthropic.claude-haiku-4-5-20251001-v1:0`
is priced as `anthropic.claude-haiku-4-5`.

| Model (ID without prefix) | Context | `global.` in / out per 1M | Geo / in-region in / out per 1M |
|-------|---------|------------|------------|
| anthropic.claude-opus-5-5 | 1M | $4.00 / $20.00 | $4.40 / $22.00 |
| anthropic.claude-sonnet-5-5 | 1M | $2.00 / $10.00 | $2.20 / $11.00 |
| anthropic.claude-haiku-5-5 ¹ | 1M | $0.10 / $0.50 | $0.11 / $0.55 |
| anthropic.claude-fable-5-1 | 1M | $10.00 / $50.00 | $11.00 / $55.00 |
| anthropic.claude-sonnet-4-6 | 1M | $3.00 / $15.00 | $3.30 / $16.50 |
| anthropic.claude-haiku-4-5-20251001-v1:0 | 200K | $1.00 / $5.00 | $1.10 / $5.50 |
| amazon.nova-2-lite-v1:0 | 1M | $0.30 / $2.50 | $0.33 / $2.75 |
| amazon.nova-pro-v1:0 | 300K | — | $0.80 / $3.20 |
| amazon.nova-lite-v1:0 | 300K | — | $0.06 / $0.24 |
| amazon.nova-micro-v1:0 | 128K | — | $0.035 / $0.14 |

¹ Prompts up to 100K tokens; longer prompts cost 5x.

Cache reads and writes use each model's own Bedrock rates (Claude Sonnet 5.5 reads
are 0.05x input; Nova cache writes are free).

**Embeddings:**

| Model | Max input | Input / 1M |
|-------|-----------|------------|
| amazon.titan-embed-text-v2:0 | 8K tokens | $0.02 |
| amazon.titan-embed-text-v1 | 8K tokens | $0.10 |

### Bedrock Prompt Caching

Bedrock caches prompts at `cachePoint` blocks, on Claude 3.7 and later and on
Amazon Nova. AIForge places them from `PROMPT_CACHE`:

| Value | Behaviour |
|-------|-----------|
| `system` (default) | Cache point after the system prompt (tools + system prompt are reused) |
| `auto` | `langchain-aws` places cache points on the system prompt, tools and conversation end |
| `off` | No cache points |

Content blocks marked with `cache_control` (for example
`messages.system(text, cache=True)`) become cache points in every mode. On
models without prompt caching the markers are dropped.

---

## Mistral AI

Mistral's hosted models, using `langchain-mistralai` (`ChatMistralAI` and
`MistralAIEmbeddings`).

```bash
pip install "aiforge-kit[mistral]"
```

### Mistral Configuration

```bash
AIFORGE_PROVIDER=mistral
AIFORGE_MISTRAL_API_KEY=...              # or MISTRAL_API_KEY
AIFORGE_MODEL=mistral-medium-3-5         # Optional, this is the default
AIFORGE_EMBEDDING_MODEL=mistral-embed    # Optional, this is the default
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AIFORGE_MISTRAL_API_KEY` | Yes | — | Mistral API key. Falls back to `MISTRAL_API_KEY` |
| `AIFORGE_MODEL` | No | `mistral-medium-3-5` | LLM model |
| `AIFORGE_EMBEDDING_MODEL` | No | `mistral-embed` | Embedding model |

Mistral accepts temperatures from 0.0 to 1.0; higher values are capped at 1.0.

### Mistral Supported Models

Prices per 1M tokens, checked 2026-10-15
([Mistral pricing](https://docs.mistral.ai/inference/pricing)). Cached input
is billed at 10% of the input price.

| Model | Context Window | Input / 1M | Cached / 1M | Output / 1M |
|-------|----------------|------------|-------------|-------------|
| mistral-large-4 ¹ | 1,000,000 | $0.68 | $0.07 | $2.09 |
| mistral-large-2512 (Large 3) | 256,000 | $0.50 | $0.05 | $1.50 |
| mistral-medium-3-5 | 256,000 | $1.50 | $0.15 | $7.50 |
| mistral-small-2603 (Small 4) | 256,000 | $0.15 | $0.015 | $0.60 |
| codestral-2508 | 128,000 | $0.30 | $0.03 | $0.90 |

¹ Public preview, at its sale price (list price $1.36 / $4.18).

**Embeddings:**

| Model | Max input | Dimensions | Input / 1M |
|-------|-----------|------------|------------|
| mistral-embed (mistral-embed-2312) | 8K tokens | 1,024 | $0.10 |
| codestral-embed (codestral-embed-2505) | 8K tokens | 1,536 (up to 3,072) | $0.15 |

Mistral caches repeated prompt prefixes; hits are reported on
`cache_read_tokens`. To improve the hit rate, pass the same
`prompt_cache_key=` on calls that share a prefix. `PROMPT_CACHE` has no effect
for Mistral.

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
| `gemini` | gemini-embedding-2, gemini-embedding-001 | Same `GOOGLE_API_KEY` as the LLM |
| `bedrock` | amazon.titan-embed-text-v2:0, amazon.titan-embed-text-v1 | Same AWS region and credentials as the LLM |
| `mistral` | mistral-embed, mistral-embed-2312, codestral-embed, codestral-embed-2505 | Same `MISTRAL_API_KEY` as the LLM |

`gemini`, `bedrock` and `mistral` fill in their default `MODEL` and
`EMBEDDING_MODEL` when you don't set them, so `provider: gemini` alone is a
complete configuration.

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
from typing import Any, AsyncGenerator

class LLMProvider(ABC):
    supports_tools: bool = False              # accepts tools= / tool_choice=
    supports_structured_output: bool = False  # accepts response_format=

    @abstractmethod
    def complete(self, messages: list[dict[str, Any]], **kwargs) -> CompletionResult:
        """Synchronous completion."""

    @abstractmethod
    async def complete_async(self, messages: list[dict[str, Any]], **kwargs) -> CompletionResult:
        """Asynchronous completion."""

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        """Token-by-token streaming."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""
```

All built-in providers support tools and structured output. Custom providers
leave the flags `False` and modules fall back to prompt-based JSON.

### Messages

Messages use the OpenAI-style format. `content` is a string or a list of
content blocks, and `aiforge.core.messages` has builders for every shape:

```python
from aiforge.core import messages as m

msgs = [
    m.system("You are a helpful assistant."),
    m.user("What does this invoice say?", m.file(path="invoice.pdf")),
    m.user("And this photo?", m.image(path="receipt.jpg")),
]
```

Assistant turns can carry tool calls (`m.assistant(text, [ToolCall(...)])`)
and tool results go back as `m.tool_result(call_id, output)`. Providers
convert everything to LangChain messages via `to_lc_messages()`, and each
LangChain chat model translates the blocks to its own API.

### Native tool calling

```python
result = provider.complete(msgs, tools=[my_langchain_tool])
for call in result.tool_calls:          # list[ToolCall(id, name, arguments)]
    ...
```

`ChatAgent` uses this automatically on providers that support it. Claude
Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced `tool_choice` values, so
leave `tool_choice` unset for them.

### Native structured output

```python
class Invoice(BaseModel):
    vendor: str
    total: float

result = provider.complete(msgs, response_format=Invoice)  # or a JSON-schema dict
result.parsed   # {"vendor": "...", "total": 12.5}
result.text     # the same object as JSON
```

The provider's native `json_schema` mode constrains generation to the schema
(OpenAI and Azure response formats, Claude `output_config.format`, Ollama
`format`). If a model or schema isn't accepted, the call is retried once
without `response_format` and the caller's own validation takes over.
`Extractor`, `Classifier` (labels restricted to the taxonomy) and `Moderator`
use this path.

### Prompt caching

`PROMPT_CACHE` controls Anthropic prompt caching:

| Value | Behaviour |
|-------|-----------|
| `system` (default) | Cache breakpoint on the system prompt, so tools + system prompt are reused |
| `auto` | Top-level automatic caching, which also reuses a growing conversation history |
| `off` | No breakpoints |

Amazon Bedrock uses the same three modes with `cachePoint` blocks (see
[Bedrock Prompt Caching](#bedrock-prompt-caching)). OpenAI, Azure, Gemini and
Mistral cache long prompts automatically. Either way, cached input
tokens are reported on `CompletionResult.cache_read_tokens` /
`cache_write_tokens` and priced at the discounted rate (Claude: reads 0.1x,
writes 1.25x; OpenAI: reads 0.5x; Gemini, Bedrock and Mistral models carry
their own cached-input prices). Override with `cache_read` / `cache_write`
keys in `custom_pricing`. See [Cost Tracking](cost-tracking.md#prompt-caching--cost-savings).

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
    tool_calls: list[ToolCall] = []     # Native tool calls requested by the model
    parsed: dict | None = None          # Schema-validated object (response_format=)
    cache_read_tokens: int = 0          # Input tokens served from the prompt cache
    cache_write_tokens: int = 0         # Input tokens written to the prompt cache
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
| `tool_calls` | `list[ToolCall]` | Tool calls requested by the model |
| `parsed` | `dict \| None` | Structured output, when `response_format` was used |
| `cache_read_tokens` / `cache_write_tokens` | `int` | Prompt-cache usage, included in `input_tokens` |
| `latency_ms` | `float` | Wall-clock time in milliseconds |
| `raw_response` | `dict \| None` | Full provider-specific payload |
| `total_tokens` | `int` (property) | `input_tokens + output_tokens` |
| `cost` | `float` (property) | USD cost, with cached tokens at their own rates |

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
    "gemini":    (GeminiProvider,    GeminiEmbeddingProvider),      # if langchain-google-genai installed
    "bedrock":   (BedrockProvider,   BedrockEmbeddingProvider),     # if langchain-aws installed
    "mistral":   (MistralProvider,   MistralEmbeddingProvider),     # if langchain-mistralai installed
}
```

The registry is populated lazily by `_ensure_registered()` on first access.
Providers whose dependencies are missing are silently skipped.

### Listing Providers

```python
from aiforge.providers import list_providers

print(list_providers())
# ["openai", "anthropic", "ollama", "azure", "gemini", "bedrock", "mistral"]
# — gemini, bedrock and mistral only when their extra is installed
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
| `high` | Complex reasoning, important outputs | gpt-4o, claude-opus-5-5, claude-sonnet-5-5 |
| `medium` | General tasks, good quality/cost balance | gpt-4o-mini, claude-haiku-5-5, llama3.1 |
| `low` | Simple tasks, maximum throughput | gpt-3.5-turbo, llama3.2 |

### Built-in Model Catalogue

| Model | Provider | Tier | Input / 1M | Output / 1M | Context |
|-------|----------|------|------------|-------------|---------|
| gpt-4o | openai | high | $2.50 | $10.00 | 128K |
| gpt-4o-mini | openai | medium | $0.15 | $0.60 | 128K |
| claude-opus-5-5 | anthropic | high | $4.00 | $20.00 | 1M |
| claude-sonnet-5-5 | anthropic | high | $2.00 | $10.00 | 1M |
| claude-haiku-5-5 | anthropic | medium | $0.10 | $0.50 | 1M |
| llama3.1 | ollama | medium | $0.00 | $0.00 | 131K |
| llama3.2 | ollama | low | $0.00 | $0.00 | 131K |

You can extend this with `CUSTOM_MODELS` in configuration.

### Model Router Usage Examples

```python
from aiforge.core.model_router import ModelRouter

router = ModelRouter()

# Cheapest high-quality model
model = router.select(quality="high")
# → "claude-sonnet-5-5" ($2.00 input — cheaper than gpt-4o's $2.50)

# Cheapest model from a specific provider
model = router.select(quality="medium", provider="openai")
# → "gpt-4o-mini"

# Prefer local (free) models
model = router.select(quality="medium", prefer_local=True)
# → "llama3.1" (free, Ollama)

# Require large context window
model = router.select(quality="high", min_context=150_000)
# → "claude-sonnet-5-5" (1M context)

# Estimate cost before calling
cost = router.estimate_cost("gpt-4o", input_tokens=5000, output_tokens=1000)
# → 0.0225  (USD)

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

**Gemini / Bedrock / Mistral** (`wrap_sdk_error()` in `aiforge/providers/_errors.py`),
which also follows the exception's `__cause__` chain:

| Native Error | Mapped To |
|-------------|-----------|
| HTTP 429, *RateLimit*, AWS `ThrottlingException` / `ServiceQuotaExceededException` | `RateLimitError` |
| HTTP 408 / 504, *Timeout*, AWS `ModelTimeoutException`, botocore `ReadTimeoutError` | `APITimeoutError` |
| *Connect* (httpx `ConnectError`, botocore `EndpointConnectionError`) | `APIConnectionError` |
| Other | `ProviderError` |

Embedding errors from these three providers are mapped the same way, so
`@retry_embedding_call()` retries them too.

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
| `gemini` | langchain-google-genai ≥4.0, <5 | Google Gemini LLM + embeddings |
| `bedrock` | langchain-aws ≥1.5, <2 | Amazon Bedrock LLM + embeddings |
| `mistral` | langchain-mistralai ≥1.0, <2 | Mistral AI LLM + embeddings |
| `pinecone` | pinecone≥5.1.0 | Pinecone vector store |
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
pip install aiforge-kit                                      # Core + OpenAI + ChromaDB
pip install "aiforge-kit[anthropicai]"                       # + Anthropic + Voyage
pip install "aiforge-kit[local]"                             # + Ollama
pip install "aiforge-kit[gemini]"                            # + Google Gemini
pip install "aiforge-kit[bedrock]"                           # + Amazon Bedrock
pip install "aiforge-kit[mistral]"                           # + Mistral AI
pip install "aiforge-kit[anthropicai,pgvector,server]"       # Custom combo
pip install "aiforge-kit[all]"                               # Everything
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ConfigError: provider 'anthropic' not registered` | Missing dependency | `pip install "aiforge-kit[anthropicai]"` |
| `ConfigError: provider 'ollama' not registered` | Missing dependency | `pip install "aiforge-kit[local]"` |
| `ConfigError: Provider 'gemini' is not available` | Missing dependency | `pip install "aiforge-kit[gemini]"` (likewise `bedrock`, `mistral`) |
| Bedrock: `Could not load credentials` / `NoRegionError` | No AWS credentials or region | Configure the AWS credential chain (`aws configure`, SSO, role) and set `AIFORGE_AWS_REGION` |
| Bedrock: `ValidationException` about on-demand throughput | Bare model ID for a profile-only model | Use an inference profile ID, e.g. `global.anthropic.claude-sonnet-5-5` |
| Gemini output repeats itself | Low temperature on Gemini 3+ | Set `temperature: 1.0` |
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
