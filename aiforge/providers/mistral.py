"""
Mistral AI provider: Mistral chat models + Mistral embeddings.

Requires:
    pip install "aiforge-kit[mistral]"
    AIFORGE_MISTRAL_API_KEY (or MISTRAL_API_KEY in the environment)

Mistral caches repeated prompt prefixes and reports the hits in
``usage.prompt_tokens_details.cached_tokens``; ``langchain-mistralai`` drops
that field, so this module restores it as ``cache_read_tokens``. Pass
``prompt_cache_key=`` on a call to group requests that share a prefix.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from aiforge.core.types import ProviderError
from aiforge.providers._errors import wrap_sdk_error
from aiforge.providers._langchain import LangChainEmbeddingProvider, LangChainLLMProvider

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings

_MISSING = 'langchain-mistralai is not installed. Run: pip install "aiforge-kit[mistral]"'


def _api_key(settings: AIForgeSettings) -> dict[str, Any]:
    """Pass the key only when configured, so the SDK's own env lookup still works."""
    return {"api_key": settings.MISTRAL_API_KEY} if settings.MISTRAL_API_KEY else {}


def cached_prompt_tokens(usage: dict[str, Any] | None) -> int:
    """Cached prompt tokens from a Mistral ``usage`` object (0 when absent)."""
    details = (usage or {}).get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0)


@cache
def _chat_class() -> Any:
    """``ChatMistralAI`` subclass that keeps cached-token usage on the message."""
    from langchain_mistralai import ChatMistralAI

    class _ChatMistralAI(ChatMistralAI):
        def _create_chat_result(self, response: dict[str, Any]) -> Any:
            result = super()._create_chat_result(response)
            cached = cached_prompt_tokens(response.get("usage"))
            for gen in result.generations:
                usage = getattr(gen.message, "usage_metadata", None)
                if cached and usage is not None:
                    usage["input_token_details"] = {"cache_read": cached}
            return result

    return _ChatMistralAI


def _offline_tokenizer() -> Any:
    """A local stand-in for the tokenizer ``MistralAIEmbeddings`` downloads.

    ``MistralAIEmbeddings`` fetches a gated Hugging Face tokenizer at
    construction only to size its request batches. Counting every three
    characters as one token over-estimates real token counts, which keeps
    batches safely under the API limit without any network access.
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]{1,3}"), behavior="isolated")
    return tokenizer


class MistralProvider(LangChainLLMProvider):
    """LLM provider backed by Mistral AI (``langchain-mistralai``)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            import langchain_mistralai  # noqa: F401
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._settings = settings
        self._model_name = settings.MODEL
        self._llm = _chat_class()(
            model=settings.MODEL,
            # Mistral accepts temperatures in [0, 1]; AIForge allows up to 2.
            temperature=min(settings.TEMPERATURE, 1.0),
            max_tokens=settings.MAX_TOKENS,
            **_api_key(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)


class MistralEmbeddingProvider(LangChainEmbeddingProvider):
    """Embedding provider backed by Mistral embeddings (``mistral-embed`` …)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_mistralai import MistralAIEmbeddings
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = MistralAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            tokenizer=_offline_tokenizer(),
            **_api_key(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)
