"""
Provider registry: resolve an LLMProvider or EmbeddingProvider from settings.

Custom providers can be registered at runtime::

    from aiforge.providers import register_provider
    register_provider("my_provider", MyLLMProvider, MyEmbeddingProvider)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiforge.core.provider import EmbeddingProvider, LLMProvider
    from aiforge.infra.config import AIForgeSettings

# Registry: provider_name -> (llm_class_factory, embedding_class_factory)
_REGISTRY: dict[str, tuple[Any, Any]] = {}


def _ensure_registered() -> None:
    """Lazily populate the registry on first use to avoid import-time side-effects."""
    if _REGISTRY:
        return

    from aiforge.providers.openai import OpenAIEmbeddingProvider, OpenAIProvider

    _REGISTRY["openai"] = (OpenAIProvider, OpenAIEmbeddingProvider)

    # Anthropic — optional extra
    try:
        from aiforge.providers.anthropic import (
            AnthropicEmbeddingProvider,
            AnthropicProvider,
        )

        _REGISTRY["anthropic"] = (AnthropicProvider, AnthropicEmbeddingProvider)
    except ImportError:
        pass

    # Ollama — optional extra
    try:
        from aiforge.providers.ollama import OllamaEmbeddingProvider, OllamaProvider

        _REGISTRY["ollama"] = (OllamaProvider, OllamaEmbeddingProvider)
    except ImportError:
        pass

    # Azure OpenAI
    try:
        from aiforge.providers.azure import AzureEmbeddingProvider, AzureProvider

        _REGISTRY["azure"] = (AzureProvider, AzureEmbeddingProvider)
    except ImportError:
        pass


def get_provider(settings: "AIForgeSettings") -> "LLMProvider":
    """Return the LLMProvider instance for the configured provider."""
    _ensure_registered()
    name = settings.PROVIDER.lower()
    if name not in _REGISTRY:
        from aiforge.core.types import ConfigError

        available = list(_REGISTRY.keys())
        raise ConfigError(
            f"Provider {name!r} is not available. "
            f"Available: {available}. "
            f"Install the matching extra, e.g. pip install \"aiforge[{name}]\""
        )
    llm_cls, _ = _REGISTRY[name]
    return llm_cls(settings)


def get_embedding_provider(settings: "AIForgeSettings") -> "EmbeddingProvider":
    """Return the EmbeddingProvider instance for the configured provider."""
    _ensure_registered()
    name = settings.PROVIDER.lower()
    if name not in _REGISTRY:
        from aiforge.core.types import ConfigError

        raise ConfigError(f"Provider {name!r} not registered.")
    _, emb_cls = _REGISTRY[name]
    return emb_cls(settings)


def register_provider(
    name: str,
    llm_class: Any,
    embedding_class: Any,
) -> None:
    """Register a custom provider.

    Args:
        name:            Provider key (used in ``AIFORGE_PROVIDER`` config).
        llm_class:       Class (or callable) that accepts ``(settings)`` and
                         returns an :class:`~aiforge.core.provider.LLMProvider`.
        embedding_class: Class (or callable) that accepts ``(settings)`` and
                         returns an :class:`~aiforge.core.provider.EmbeddingProvider`.
    """
    _REGISTRY[name.lower()] = (llm_class, embedding_class)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    _ensure_registered()
    return list(_REGISTRY.keys())
