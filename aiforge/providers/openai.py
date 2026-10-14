"""
OpenAI provider: GPT-4o (LLM) + text-embedding-3-small (embeddings).

Requires: pip install langchain-openai
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiforge.core.provider import EmbeddingProvider
from aiforge.core.types import ProviderError
from aiforge.infra.retry import retry_embedding_call
from aiforge.providers._langchain import LangChainLLMProvider, wrap_error_by_class_name
from aiforge.providers._utils import supports_sampling_params

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings


class OpenAIProvider(LangChainLLMProvider):
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
            temperature=settings.TEMPERATURE if supports_sampling_params(settings.MODEL) else None,
            max_tokens=settings.MAX_TOKENS,  # type: ignore[call-arg]
            api_key=settings.OPENAI_API_KEY,  # type: ignore[arg-type]
        )
        self._model_name = settings.MODEL

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return _wrap_openai_error(exc)


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


def _wrap_openai_error(exc: Exception) -> ProviderError:
    """Map openai SDK exceptions to AIForge error types (also used by azure.py)."""
    return wrap_error_by_class_name(exc)
