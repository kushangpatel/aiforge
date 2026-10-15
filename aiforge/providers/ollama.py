"""
Ollama provider: local LLMs (Llama 3, Mistral, …) + local embeddings.

Requires:
    pip install langchain-ollama
    Running Ollama server at AIFORGE_OLLAMA_BASE_URL (default: http://localhost:11434)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiforge.core.provider import EmbeddingProvider
from aiforge.core.types import APIConnectionError, ProviderError
from aiforge.infra.retry import retry_embedding_call
from aiforge.providers._langchain import LangChainLLMProvider

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class OllamaProvider(LangChainLLMProvider):
    """LLM provider backed by Ollama (local models)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                'langchain-ollama not installed. Run: pip install "aiforge-kit[local]"'
            ) from exc

        self._model_name = settings.MODEL
        self._llm = ChatOllama(
            model=settings.MODEL,
            temperature=settings.TEMPERATURE,
            base_url=settings.OLLAMA_BASE_URL,
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return _wrap_ollama_error(exc)

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return dict(kwargs)

    def _usage(self, response: Any, messages: list[dict[str, Any]], text: str) -> tuple[int, int]:
        messages_text = " ".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
        return _extract_tokens(response, messages_text, text)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Ollama local embeddings."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                'langchain-ollama not installed. Run: pip install "aiforge-kit[local]"'
            ) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry_embedding_call()
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    @retry_embedding_call()
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)

    @retry_embedding_call()
    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    @retry_embedding_call()
    async def embed_query_async(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)


def _count_tokens_fallback(text: str) -> int:
    """Estimate token count using tiktoken (cl100k_base) when the model doesn't report usage."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        # If tiktoken is unavailable fall back to a rough word-count estimate
        return max(1, len(text.split()))


def _extract_tokens(response: Any, messages_text: str, response_text: str) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from an Ollama response, with fallback."""
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if input_tokens == 0 and output_tokens == 0:
        input_tokens = _count_tokens_fallback(messages_text)
        output_tokens = _count_tokens_fallback(response_text)
    return input_tokens, output_tokens


def _wrap_ollama_error(exc: Exception) -> ProviderError:
    """Convert an Ollama exception into the appropriate AIForge error type."""
    msg = str(exc).lower()
    if "connection" in msg or "refused" in msg:
        return APIConnectionError(f"Cannot connect to Ollama server. Is it running? Error: {exc}")
    return ProviderError(str(exc))
