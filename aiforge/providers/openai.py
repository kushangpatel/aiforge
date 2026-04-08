"""
OpenAI provider: GPT-4o (LLM) + text-embedding-3-small (embeddings).

Requires: pip install langchain-openai
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
from aiforge.providers._utils import to_lc_messages as _to_lc_messages

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class OpenAIProvider(LLMProvider):
    """LLM provider backed by OpenAI Chat Completions."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from exc

        self._settings = settings
        self._llm = ChatOpenAI(
            model=settings.MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,  # type: ignore[call-arg]
            api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
        )
        self._model_name = settings.MODEL

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry_llm_call()
    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResult:
        lc_messages = _to_lc_messages(messages)
        start = time.monotonic()
        try:
            response = self._llm.invoke(lc_messages, **kwargs)
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, "usage_metadata", None) or {}
        return CompletionResult(
            text=str(response.content),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=self._model_name,
            finish_reason=getattr(response, "response_metadata", {}).get("finish_reason", "stop"),
            latency_ms=latency_ms,
        )

    @retry_llm_call()
    async def complete_async(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResult:
        lc_messages = _to_lc_messages(messages)
        start = time.monotonic()
        try:
            response = await self._llm.ainvoke(lc_messages, **kwargs)
        except Exception as exc:
            raise _wrap_openai_error(exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = getattr(response, "usage_metadata", None) or {}
        return CompletionResult(
            text=str(response.content),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=self._model_name,
            finish_reason=getattr(response, "response_metadata", {}).get("finish_reason", "stop"),
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        lc_messages = _to_lc_messages(messages)
        async for chunk in self._llm.astream(lc_messages, **kwargs):
            yield str(chunk.content)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI Embeddings."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Re-export for backward compatibility (azure.py imports _wrap_openai_error from here)
def _wrap_openai_error(exc: Exception) -> ProviderError:
    """Map openai SDK exceptions to AIForge error types."""
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name:
        return RateLimitError(str(exc))
    if "Timeout" in cls_name:
        return APITimeoutError(str(exc))
    if "APIConnection" in cls_name or "Connection" in cls_name:
        return APIConnectionError(str(exc))
    return ProviderError(str(exc))
