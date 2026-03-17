# Provider Abstraction Layer

AIForge is provider-agnostic. Swap between OpenAI, Anthropic, Ollama, and Azure OpenAI with a single config change — zero code changes required.

---

## Provider Interface

```python
# aiforge/core/provider.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> CompletionResult:
        """Synchronous completion."""

    @abstractmethod
    async def complete_async(self, messages: list[dict], **kwargs) -> CompletionResult:
        """Async completion."""

    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        """Streaming completion."""


class CompletionResult(BaseModel):
    """Returned by every LLMProvider.complete() / complete_async() call."""
    text: str                           # the generated text content
    input_tokens: int                   # prompt tokens consumed
    output_tokens: int                  # completion tokens generated
    model: str                          # actual model used (e.g., "gpt-4o-2024-08-06")
    finish_reason: str                  # "stop" | "length" | "tool_calls" | "content_filter"
    raw_response: dict | None = None    # provider-specific raw response (opt-in via config)
    latency_ms: float                   # wall-clock time for this call


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
```

---

## Supported Providers

| Provider | LLM Models | Embedding Models | Local? |
|----------|-----------|-----------------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo | text-embedding-3-small, text-embedding-3-large | No |
| **Anthropic** | claude-3.5-sonnet, claude-3-haiku | voyage-3 (via Voyage AI) | No |
| **Ollama** | llama3, mistral, codellama, phi-3 | nomic-embed-text, mxbai-embed-large | Yes |
| **Azure OpenAI** | Same as OpenAI (deployed models) | Same as OpenAI | No |

---

## Provider-Embedding Compatibility

Not all LLM providers ship their own embeddings. AIForge validates provider/embedding combinations on startup and raises `ConfigError` with a helpful message if they don't match.

| `AIFORGE_PROVIDER` | Default `AIFORGE_EMBEDDING_MODEL` | Notes |
|--------------------|----------------------------------|-------|
| `openai` | `text-embedding-3-small` | Both from OpenAI. Works out of the box. |
| `anthropic` | `voyage-3` | Anthropic has no embedding API. AIForge auto-routes to Voyage AI. Requires `AIFORGE_VOYAGE_API_KEY`. |
| `ollama` | `nomic-embed-text` | Must be pulled locally: `ollama pull nomic-embed-text`. If the model isn't available, `ConfigError` is raised. |
| `azure` | Deployed embedding model name | Must match a deployed model in your Azure OpenAI resource. |

### Mismatch Error Example

If the user sets `provider=ollama` but leaves `embedding_model=text-embedding-3-small` (an OpenAI model), AIForge raises:

```
ConfigError: Embedding model 'text-embedding-3-small' is not available with provider 'ollama'.
Available Ollama embedding models: nomic-embed-text, mxbai-embed-large.
Set AIFORGE_EMBEDDING_MODEL to a compatible model, or run: ollama pull nomic-embed-text
```

---

## Swapping Providers

### Via Config (YAML)

```yaml
# aiforge.yaml — just change these 2 lines
provider: anthropic
model: claude-3-5-sonnet-20241022
```

### Via Code (Runtime)

```python
from aiforge import Summarizer
s = Summarizer(provider="anthropic", model="claude-3-5-sonnet-20241022")
```

Zero code changes required in the consuming application.

---

## Extending AIForge

### Custom Provider Registration

```python
from aiforge.core.provider import LLMProvider, CompletionResult

class MyCustomProvider(LLMProvider):
    def complete(self, messages, **kwargs) -> CompletionResult:
        # Your implementation here
        ...

    async def complete_async(self, messages, **kwargs) -> CompletionResult:
        # Your async implementation
        ...

    async def stream(self, messages, **kwargs):
        # Your streaming implementation
        ...

# Register for use
from aiforge.providers import register_provider
register_provider("my_provider", MyCustomProvider)

# Now usable everywhere
s = Summarizer(provider="my_provider")
```

### Custom Vector Store Adapter

```python
from aiforge.core.vectorstore import VectorStoreAdapter

class PgVectorAdapter(VectorStoreAdapter):
    def add_documents(self, docs, embeddings): ...
    def query(self, embedding, top_k, filter): ...
    def delete_collection(self, name): ...

from aiforge.vectorstores import register_vectorstore
register_vectorstore("pgvector", PgVectorAdapter)

qa = DocQA(vector_store="pgvector", pgvector_dsn="postgresql://...")
```

---

## Provider-Specific Configuration

### OpenAI

```bash
export AIFORGE_PROVIDER=openai
export AIFORGE_MODEL=gpt-4o
export AIFORGE_OPENAI_API_KEY=sk-...
export AIFORGE_EMBEDDING_MODEL=text-embedding-3-small
```

### Anthropic

```bash
export AIFORGE_PROVIDER=anthropic
export AIFORGE_MODEL=claude-3-5-sonnet-20241022
export AIFORGE_ANTHROPIC_API_KEY=sk-ant-...
export AIFORGE_VOYAGE_API_KEY=pa-...    # Required for embeddings
export AIFORGE_EMBEDDING_MODEL=voyage-3
```

### Ollama (Local)

```bash
export AIFORGE_PROVIDER=ollama
export AIFORGE_MODEL=llama3
export AIFORGE_OLLAMA_BASE_URL=http://localhost:11434
export AIFORGE_EMBEDDING_MODEL=nomic-embed-text

# Ensure models are pulled
ollama pull llama3
ollama pull nomic-embed-text
```

### Azure OpenAI

```bash
export AIFORGE_PROVIDER=azure
export AIFORGE_MODEL=your-deployed-model-name
export AIFORGE_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AIFORGE_AZURE_OPENAI_API_KEY=...
export AIFORGE_EMBEDDING_MODEL=your-deployed-embedding-model
```

---

## See Also

- [Configuration](configuration.md) — Full configuration reference
- [Cost Tracking](cost-tracking.md) — Provider-specific pricing
