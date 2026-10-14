"""
Anthropic provider: Claude (LLM) + Voyage AI (embeddings).

Requires:
    pip install langchain-anthropic langchain-voyageai
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from aiforge.core.provider import EmbeddingProvider, LLMProvider
from aiforge.core.types import (
    APIConnectionError,
    APITimeoutError,
    CompletionResult,
    ProviderError,
    RateLimitError,
)
from aiforge.infra.retry import retry_embedding_call, retry_llm_call
from aiforge.providers._utils import drop_unsupported_sampling, supports_sampling_params
from aiforge.providers._utils import to_lc_messages as _to_lc_messages

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class AnthropicProvider(LLMProvider):
    """LLM provider backed by Anthropic Claude."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                'langchain-anthropic not installed. Run: pip install "aiforge[anthropicai]"'
            ) from exc

        self._settings = settings
        self._model_name = settings.MODEL
        self._llm = ChatAnthropic(  # type: ignore[call-arg]
            model=settings.MODEL,
            temperature=settings.TEMPERATURE if supports_sampling_params(settings.MODEL) else None,
            max_tokens=settings.MAX_TOKENS or 4096,
            api_key=settings.ANTHROPIC_API_KEY,  # type: ignore[arg-type]
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry_llm_call()
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        lc_messages = _to_lc_messages(messages)
        start = time.monotonic()
        try:
            response = self._llm.invoke(
                lc_messages, **drop_unsupported_sampling(self._model_name, kwargs)
            )
        except Exception as exc:
            raise _wrap_anthropic_error(exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, "usage_metadata", None) or {}
        return CompletionResult(
            text=str(response.content),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=self._model_name,
            finish_reason=getattr(response, "response_metadata", {}).get("stop_reason", "stop"),
            latency_ms=latency_ms,
        )

    @retry_llm_call()
    async def complete_async(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> CompletionResult:
        lc_messages = _to_lc_messages(messages)
        start = time.monotonic()
        try:
            response = await self._llm.ainvoke(
                lc_messages, **drop_unsupported_sampling(self._model_name, kwargs)
            )
        except Exception as exc:
            raise _wrap_anthropic_error(exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, "usage_metadata", None) or {}
        return CompletionResult(
            text=str(response.content),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=self._model_name,
            finish_reason=getattr(response, "response_metadata", {}).get("stop_reason", "stop"),
            latency_ms=latency_ms,
        )

    async def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        lc_messages = _to_lc_messages(messages)
        async for chunk in self._llm.astream(
            lc_messages, **drop_unsupported_sampling(self._model_name, kwargs)
        ):
            yield str(chunk.content)


class AnthropicEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Voyage AI (recommended for Anthropic)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                'langchain-voyageai not installed. Run: pip install "aiforge[anthropicai]"'
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
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name:
        return RateLimitError(str(exc))
    if "Timeout" in cls_name:
        return APITimeoutError(str(exc))
    if "Connection" in cls_name:
        return APIConnectionError(str(exc))
    return ProviderError(str(exc))
