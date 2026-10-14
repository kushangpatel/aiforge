"""
LLMProvider and EmbeddingProvider abstract base classes.

Every concrete provider (OpenAI, Anthropic, Ollama, Azure) implements
both ABCs.  Providers are resolved via the registry in
``aiforge.providers``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from aiforge.core.types import CompletionResult


class LLMProvider(ABC):
    """Abstract provider for chat-completion LLM calls.

    Messages are OpenAI-style dicts; ``content`` may be a string or a list of
    content blocks (see :mod:`aiforge.core.messages`).

    Providers that set :attr:`supports_tools` accept ``tools=`` (and
    ``tool_choice=``) in ``**kwargs`` and report requested calls on
    :attr:`CompletionResult.tool_calls`. Providers that set
    :attr:`supports_structured_output` accept ``response_format=`` (a Pydantic
    model or JSON-schema dict) and fill :attr:`CompletionResult.parsed`.
    Callers check these flags before passing the keywords.
    """

    supports_tools: bool = False
    supports_structured_output: bool = False

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Synchronous chat completion.

        Args:
            messages: OpenAI-style message list, e.g.
                      ``[{"role": "user", "content": "Hello"}]``
            **kwargs: Provider-specific overrides (temperature, max_tokens …),
                      plus ``tools`` / ``response_format`` where supported.

        Returns:
            :class:`CompletionResult` with text, token counts, latency, etc.
        """

    @abstractmethod
    async def complete_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Async chat completion."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Async streaming completion.

        Yields individual string tokens as they arrive.
        """
        # Make this method an async generator
        return
        yield  # pragma: no cover

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier string (e.g. ``"gpt-4o"``)."""


class EmbeddingProvider(ABC):
    """Abstract provider for text embedding calls."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """

    @abstractmethod
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async batch embedding."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate a single query embedding (may use a different prompt)."""

    @abstractmethod
    async def embed_query_async(self, text: str) -> list[float]:
        """Async single query embedding."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The embedding model identifier string."""
