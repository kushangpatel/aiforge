"""
Shared pytest fixtures for the AIForge test suite.

All fixtures are available to every test file automatically.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiforge.core.types import CompletionResult
from aiforge.infra.cost_tracker import CostTracker
from aiforge.infra.rate_limiter import RateLimiter

# ---------------------------------------------------------------------------
# Environment setup: ensure no real API keys are needed for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Unset real API keys during unit tests and inject dummies."""
    from aiforge.infra.config import get_settings
    from aiforge.infra.tracing import reset_tracing

    monkeypatch.setenv("AIFORGE_OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
    monkeypatch.setenv("AIFORGE_ANTHROPIC_API_KEY", "test-anthropic-fake-key")
    monkeypatch.setenv("AIFORGE_VOYAGE_API_KEY", "pa-test-voyage-key")
    monkeypatch.setenv("AIFORGE_TRACING_ENABLED", "false")
    monkeypatch.setenv("AIFORGE_COST_TRACKING_ENABLED", "true")
    # Clear cached settings so monkeypatched env vars take effect
    get_settings.cache_clear()
    reset_tracing()
    yield
    get_settings.cache_clear()
    reset_tracing()


# ---------------------------------------------------------------------------
# Provider registry — restore state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provider_registry() -> Iterator[None]:
    """Prevent test-registered providers from leaking into other tests."""
    import copy

    import aiforge.providers as _reg

    snapshot = copy.copy(_reg._REGISTRY)
    builtins_loaded = _reg._BUILTINS_LOADED
    yield
    _reg._REGISTRY.clear()
    _reg._REGISTRY.update(snapshot)
    _reg._BUILTINS_LOADED = builtins_loaded


@pytest.fixture(autouse=True)
def reset_plugin_registry() -> Iterator[None]:
    """Prevent test-registered modules/vectorstores from leaking."""
    import copy

    import aiforge.registry as _reg

    mod_snapshot = copy.copy(_reg._module_registry)
    vs_snapshot = copy.copy(_reg._vectorstore_registry)
    yield
    _reg._module_registry.clear()
    _reg._module_registry.update(mod_snapshot)
    _reg._vectorstore_registry.clear()
    _reg._vectorstore_registry.update(vs_snapshot)


# ---------------------------------------------------------------------------
# Cost tracker — reset between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_cost_tracker() -> Iterator[None]:
    """Ensure each test starts with a fresh cost tracker."""
    CostTracker.reset_session()
    yield
    CostTracker.reset_session()


@pytest.fixture()
def cost_tracker() -> CostTracker:
    return CostTracker.get_session()


# ---------------------------------------------------------------------------
# Rate limiter — reset between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    RateLimiter.reset_instance()
    yield
    RateLimiter.reset_instance()


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_completion_result() -> CompletionResult:
    return CompletionResult(
        text="This is a mock LLM response.",
        input_tokens=50,
        output_tokens=20,
        model="gpt-4o",
        finish_reason="stop",
        latency_ms=123.4,
    )


@pytest.fixture()
def mock_llm(mock_completion_result: CompletionResult) -> MagicMock:
    """A mock LLMProvider that returns a predictable CompletionResult."""
    provider = MagicMock()
    provider.complete.return_value = mock_completion_result
    provider.complete_async = AsyncMock(return_value=mock_completion_result)
    provider.model_name = "gpt-4o"
    return provider


# ---------------------------------------------------------------------------
# Mock embedding provider
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embeddings() -> MagicMock:
    """A mock EmbeddingProvider returning fixed 4-dimensional vectors."""
    provider = MagicMock()
    provider.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    provider.embed_async = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    provider.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    provider.embed_query_async = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    provider.model_name = "text-embedding-3-small"
    return provider


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": "doc1",
            "text": "AIForge is a modular AI toolkit.",
            "metadata": {"source": "readme"},
        },
        {
            "id": "doc2",
            "text": "LangChain provides LLM abstractions.",
            "metadata": {"source": "docs"},
        },
        {
            "id": "doc3",
            "text": "ChromaDB stores vector embeddings.",
            "metadata": {"source": "docs"},
        },
    ]


# ---------------------------------------------------------------------------
# Temp vector store
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_vector_store(tmp_path: Any) -> Any:
    """A ChromaVectorStore backed by a temporary directory."""
    from aiforge.core.vectorstore import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings() -> Any:
    """Return an AIForgeSettings instance with test values."""
    import warnings

    from aiforge.infra.config import AIForgeSettings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AIForgeSettings(
            PROVIDER="openai",
            MODEL="gpt-4o",
            EMBEDDING_MODEL="text-embedding-3-small",
            OPENAI_API_KEY="sk-test-fake-key-for-unit-tests",
        )
