"""
Azure OpenAI provider.

Requires:
    pip install langchain-openai
    Environment variables:
        AIFORGE_AZURE_OPENAI_API_KEY
        AIFORGE_AZURE_OPENAI_ENDPOINT
        AIFORGE_AZURE_OPENAI_DEPLOYMENT  (deployment name)
        AIFORGE_AZURE_OPENAI_API_VERSION (default: 2024-02-01)
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from aiforge.core.provider import EmbeddingProvider, LLMProvider
from aiforge.core.types import CompletionResult, ConfigError, ProviderError
from aiforge.infra.retry import retry_embedding_call, retry_llm_call
from aiforge.providers._utils import to_lc_messages as _to_lc_messages
from aiforge.providers.openai import _wrap_openai_error

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class AzureProvider(LLMProvider):
    """LLM provider backed by Azure OpenAI."""

    def __init__(self, settings: "AIForgeSettings") -> None:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai not installed. Run: pip install langchain-openai"
            ) from exc

        if not settings.AZURE_OPENAI_ENDPOINT:
            from aiforge.core.types import ConfigError

            raise ConfigError(
                "AIFORGE_AZURE_OPENAI_ENDPOINT must be set when provider=azure"
            )

        self._model_name = settings.AZURE_OPENAI_DEPLOYMENT or settings.MODEL
        self._llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT or settings.MODEL,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,  # type: ignore[arg-type]
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry_llm_call()
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
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
            finish_reason=getattr(response, "response_metadata", {}).get(
                "finish_reason", "stop"
            ),
            latency_ms=latency_ms,
        )

    @retry_llm_call()
    async def complete_async(
        self, messages: list[dict[str, str]], **kwargs: Any
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
            finish_reason=getattr(response, "response_metadata", {}).get(
                "finish_reason", "stop"
            ),
            latency_ms=latency_ms,
        )

    async def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        lc_messages = _to_lc_messages(messages)
        async for chunk in self._llm.astream(lc_messages, **kwargs):
            yield str(chunk.content)


class AzureEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Azure OpenAI Embeddings."""

    def __init__(self, settings: "AIForgeSettings") -> None:
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-openai not installed. Run: pip install langchain-openai"
            ) from exc

        if not settings.AZURE_OPENAI_ENDPOINT:
            raise ConfigError(
                "AIFORGE_AZURE_OPENAI_ENDPOINT must be set when provider=azure"
            )

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.EMBEDDING_MODEL,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,  # type: ignore[arg-type]
            api_version=settings.AZURE_OPENAI_API_VERSION,
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
