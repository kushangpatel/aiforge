"""
Google Gemini provider: Gemini chat models + Gemini embeddings (Gemini Developer API).

Requires:
    pip install "aiforge-kit[gemini]"
    AIFORGE_GOOGLE_API_KEY (or GOOGLE_API_KEY / GEMINI_API_KEY in the environment)

Gemini caches repeated prompt prefixes implicitly; cached tokens are reported
as ``cache_read_tokens`` and priced at the model's cached-input rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiforge.core.types import ProviderError
from aiforge.providers._errors import wrap_sdk_error
from aiforge.providers._langchain import LangChainEmbeddingProvider, LangChainLLMProvider

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings

_MISSING = 'langchain-google-genai is not installed. Run: pip install "aiforge-kit[gemini]"'


def _api_key(settings: AIForgeSettings) -> dict[str, Any]:
    """Pass the key only when configured, so the SDK's own env lookup still works."""
    return {"google_api_key": settings.GOOGLE_API_KEY} if settings.GOOGLE_API_KEY else {}


class GeminiProvider(LangChainLLMProvider):
    """LLM provider backed by Google Gemini (``langchain-google-genai``)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._settings = settings
        self._model_name = settings.MODEL
        self._llm = ChatGoogleGenerativeAI(
            model=settings.MODEL,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_TOKENS,
            **_api_key(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)


class GeminiEmbeddingProvider(LangChainEmbeddingProvider):
    """Embedding provider backed by Gemini embeddings (``gemini-embedding-2`` …)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            **_api_key(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)
