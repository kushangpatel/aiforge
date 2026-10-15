"""
Anthropic provider: Claude (LLM) + Voyage AI (embeddings).

Requires:
    pip install langchain-anthropic langchain-voyageai
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiforge.core.messages import CACHE_BREAKPOINT
from aiforge.core.messages import text as text_block
from aiforge.core.provider import EmbeddingProvider
from aiforge.core.types import ProviderError
from aiforge.infra.retry import retry_embedding_call
from aiforge.providers._langchain import LangChainLLMProvider, wrap_error_by_class_name
from aiforge.providers._utils import supports_sampling_params

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class AnthropicProvider(LangChainLLMProvider):
    """LLM provider backed by Anthropic Claude."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                'langchain-anthropic not installed. Run: pip install "aiforge-kit[anthropicai]"'
            ) from exc

        self._settings = settings
        self._model_name = settings.MODEL
        self._llm = ChatAnthropic(  # type: ignore[call-arg]
            model=settings.MODEL,
            temperature=settings.TEMPERATURE if supports_sampling_params(settings.MODEL) else None,
            max_tokens=settings.MAX_TOKENS or 4096,
            api_key=settings.ANTHROPIC_API_KEY,  # type: ignore[arg-type]
        )

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        call_kwargs = super()._request_kwargs(kwargs)
        # PROMPT_CACHE="auto": a top-level breakpoint that follows the end of
        # every request, which suits growing multi-turn conversations.
        if self._settings.PROMPT_CACHE == "auto" and "cache_control" not in call_kwargs:
            call_kwargs["cache_control"] = dict(CACHE_BREAKPOINT)
        return call_kwargs

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Put a cache breakpoint on the last system message (``PROMPT_CACHE="system"``).

        Tools and system prompt form the stable prefix of most AIForge calls,
        so caching them is cheap insurance: prefixes below the model's minimum
        cacheable length are simply not cached and cost nothing extra.
        """
        if self._settings.PROMPT_CACHE != "system":
            return messages
        idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "system"),
            None,
        )
        if idx is None:
            return messages
        content = messages[idx].get("content") or ""
        blocks = [text_block(content)] if isinstance(content, str) else list(content)
        if not blocks or any("cache_control" in b for b in blocks):
            return messages
        blocks[-1] = {**blocks[-1], "cache_control": dict(CACHE_BREAKPOINT)}
        marked = list(messages)
        marked[idx] = {**messages[idx], "content": blocks}
        return marked


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Voyage AI (recommended for Anthropic)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                'langchain-voyageai not installed. Run: pip install "aiforge-kit[anthropicai]"'
            ) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = VoyageAIEmbeddings(  # type: ignore[call-arg]
            voyage_api_key=settings.VOYAGE_API_KEY,
            model=settings.EMBEDDING_MODEL,
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


def _wrap_anthropic_error(exc: Exception) -> ProviderError:
    return wrap_error_by_class_name(exc)
