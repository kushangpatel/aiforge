"""Unit tests for the AIForge plugin registry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiforge.core.types import BaseModule, ModuleResponse
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.registry import (
    get_module,
    get_vectorstore,
    list_modules,
    list_vectorstores,
    register_module,
    register_vectorstore,
)

# ---------------------------------------------------------------------------
# Helpers — minimal concrete stubs
# ---------------------------------------------------------------------------


class _StubModule(BaseModule):
    """Minimal concrete module for testing registration."""

    def run(self, text: str, **kwargs: Any) -> ModuleResponse:
        return ModuleResponse()

    async def arun(self, text: str, **kwargs: Any) -> ModuleResponse:
        return ModuleResponse()


class _StubVectorStore(VectorStoreAdapter):
    """Minimal concrete vector store for testing registration."""

    def add_documents(self, texts, embeddings, metadatas, ids):
        pass

    def query(self, embedding, top_k=5, filter=None, include_embeddings=False):
        return []

    def delete_collection(self, name):
        pass

    def delete_by_source(self, source):
        return 0

    def list_all_metadata(self):
        return []

    def list_collections(self):
        return []

    def collection_stats(self, name):
        return {}


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------


class TestModuleRegistry:
    def test_register_and_get(self) -> None:
        register_module("stub", _StubModule)
        assert get_module("stub") is _StubModule

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get_module("nonexistent")

    def test_get_unknown_shows_available(self) -> None:
        register_module("alpha", _StubModule)
        with pytest.raises(KeyError, match="alpha"):
            get_module("missing")

    def test_register_non_basemodule_raises(self) -> None:
        with pytest.raises(TypeError, match="BaseModule"):
            register_module("bad", str)  # type: ignore[arg-type]

    def test_register_instance_raises(self) -> None:
        with pytest.raises(TypeError, match="BaseModule"):
            register_module("bad", _StubModule())  # type: ignore[arg-type]

    def test_list_modules_sorted(self) -> None:
        register_module("zeta", _StubModule)
        register_module("alpha", _StubModule)
        names = list_modules()
        assert names == sorted(names)
        assert "alpha" in names
        assert "zeta" in names

    def test_overwrite_existing(self) -> None:
        """Re-registering the same name overwrites silently."""
        register_module("dup", _StubModule)

        class AnotherModule(_StubModule):
            pass

        register_module("dup", AnotherModule)
        assert get_module("dup") is AnotherModule


# ---------------------------------------------------------------------------
# Vector store registry
# ---------------------------------------------------------------------------


class TestVectorStoreRegistry:
    def test_register_and_get(self) -> None:
        register_vectorstore("stub_vs", _StubVectorStore)
        assert get_vectorstore("stub_vs") is _StubVectorStore

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get_vectorstore("mystery")

    def test_get_unknown_shows_available(self) -> None:
        register_vectorstore("pgvector", _StubVectorStore)
        with pytest.raises(KeyError, match="pgvector"):
            get_vectorstore("missing")

    def test_register_non_adapter_raises(self) -> None:
        with pytest.raises(TypeError, match="VectorStoreAdapter"):
            register_vectorstore("bad", dict)  # type: ignore[arg-type]

    def test_register_instance_raises(self) -> None:
        with pytest.raises(TypeError, match="VectorStoreAdapter"):
            register_vectorstore("bad", _StubVectorStore())  # type: ignore[arg-type]

    def test_list_vectorstores_sorted(self) -> None:
        register_vectorstore("weaviate", _StubVectorStore)
        register_vectorstore("pgvector", _StubVectorStore)
        names = list_vectorstores()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Built-in auto-registration (after importing aiforge)
# ---------------------------------------------------------------------------


class TestBuiltinAutoRegistration:
    def test_all_8_modules_registered(self) -> None:
        """After import aiforge, all 8 built-in modules are in the registry."""
        import aiforge  # noqa: F401 — side effect: auto-registration

        expected = {
            "summarize",
            "qa",
            "extract",
            "classify",
            "search",
            "generate",
            "chat",
            "moderate",
        }
        registered = set(list_modules())
        assert expected.issubset(registered), f"Missing: {expected - registered}"

    def test_builtin_modules_are_correct_classes(self) -> None:
        import aiforge

        assert get_module("summarize") is aiforge.Summarizer
        assert get_module("qa") is aiforge.DocQA
        assert get_module("extract") is aiforge.Extractor
        assert get_module("classify") is aiforge.Classifier
        assert get_module("search") is aiforge.SemanticSearch
        assert get_module("generate") is aiforge.Generator
        assert get_module("chat") is aiforge.ChatAgent
        assert get_module("moderate") is aiforge.Moderator

    def test_chroma_vectorstore_registered(self) -> None:
        import aiforge  # noqa: F401
        from aiforge.core.vectorstore import ChromaVectorStore

        assert get_vectorstore("chroma") is ChromaVectorStore


# ---------------------------------------------------------------------------
# Custom module works in Pipeline
# ---------------------------------------------------------------------------


class TestCustomModuleInPipeline:
    def test_registered_module_usable_in_pipeline(self) -> None:
        """A custom registered module can participate in a Pipeline."""
        from aiforge.pipeline.pipeline import Pipeline

        class UpperModule(BaseModule):
            def run(self, text: str, **kwargs: Any) -> ModuleResponse:
                return _TextResponse(text=text.upper())

            async def arun(self, text: str, **kwargs: Any) -> ModuleResponse:
                return _TextResponse(text=text.upper())

        register_module("upper", UpperModule)

        # Resolve from registry
        cls = get_module("upper")
        instance = object.__new__(cls)
        # Manually set required BaseModule attributes (bypass __init__)
        import structlog

        from aiforge.infra.cost_tracker import CostTracker

        instance.settings = MagicMock()
        instance.settings.RATE_LIMIT_RPM = 500
        instance.settings.RATE_LIMIT_TPM = 150_000
        instance.settings.CUSTOM_PROMPTS = {}
        instance.cost_tracker = CostTracker.get_session()
        instance.logger = structlog.get_logger()
        instance._provider = None
        instance._embedding_provider = None

        pipe = Pipeline([("upper", instance)])
        result = pipe.run("hello")
        assert result.steps["upper"].to_text() == "HELLO"


class _TextResponse(ModuleResponse):
    """Simple response that wraps text for pipeline chaining."""

    text_value: str = ""

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text_value = text

    def to_text(self) -> str:
        return self.text_value


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_registry_importable_from_top_level(self) -> None:
        from aiforge import (
            ChromaVectorStore,
            VectorStoreAdapter,
            get_module,
            get_provider,
            get_vectorstore,
            list_modules,
            list_providers,
            list_vectorstores,
            register_module,
            register_provider,
            register_vectorstore,
        )

        assert callable(register_module)
        assert callable(get_module)
        assert callable(list_modules)
        assert callable(register_vectorstore)
        assert callable(get_vectorstore)
        assert callable(list_vectorstores)
        assert callable(register_provider)
        assert callable(get_provider)
        assert callable(list_providers)
        assert VectorStoreAdapter is not None
        assert ChromaVectorStore is not None


# ---------------------------------------------------------------------------
# Provider registry (delegates to aiforge.providers)
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_register_and_list(self) -> None:
        """register_provider() stores a custom provider, list_providers() includes it."""
        from aiforge.registry import list_providers, register_provider

        class _FakeLLM:
            pass

        class _FakeEmbed:
            pass

        register_provider("custom_test", _FakeLLM, _FakeEmbed)
        assert "custom_test" in list_providers()

    def test_list_providers_includes_builtins(self) -> None:
        from aiforge.registry import list_providers

        names = list_providers()
        # OpenAI is always lazily registered
        assert "openai" in names

    def test_register_provider_importable_from_registry(self) -> None:
        """Spec requires register_provider, get_provider, list_providers in registry.py."""
        from aiforge.registry import get_provider, list_providers, register_provider

        assert callable(register_provider)
        assert callable(get_provider)
        assert callable(list_providers)


# ---------------------------------------------------------------------------
# Name normalisation and validation
# ---------------------------------------------------------------------------


class TestNameNormalisation:
    def test_module_name_lowercased(self) -> None:
        register_module("MyModule", _StubModule)
        assert get_module("mymodule") is _StubModule
        assert "mymodule" in list_modules()

    def test_module_name_stripped(self) -> None:
        register_module("  padded  ", _StubModule)
        assert get_module("padded") is _StubModule

    def test_vectorstore_name_lowercased(self) -> None:
        register_vectorstore("PgVector", _StubVectorStore)
        assert get_vectorstore("pgvector") is _StubVectorStore
        assert "pgvector" in list_vectorstores()

    def test_get_module_case_insensitive(self) -> None:
        register_module("alpha", _StubModule)
        assert get_module("Alpha") is _StubModule
        assert get_module("ALPHA") is _StubModule


class TestNameValidation:
    def test_empty_module_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_module("", _StubModule)

    def test_whitespace_module_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_module("   ", _StubModule)

    def test_empty_vectorstore_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_vectorstore("", _StubVectorStore)

    def test_none_module_name_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            register_module(None, _StubModule)  # type: ignore[arg-type]

    def test_empty_name_on_get_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            get_module("")


# ---------------------------------------------------------------------------
# Custom provider end-to-end: register → settings → BaseModule
# ---------------------------------------------------------------------------


class TestCustomProviderEndToEnd:
    """Verify that register_provider() + Summarizer(provider=...) works."""

    def test_config_accepts_registered_provider(self) -> None:
        """AIForgeSettings should accept a custom provider name that is in the registry."""
        from aiforge.infra.config import AIForgeSettings
        from aiforge.registry import register_provider

        class _DummyLLM:
            pass

        class _DummyEmbed:
            pass

        register_provider("test_gateway", _DummyLLM, _DummyEmbed)
        # Must NOT raise ValidationError
        settings = AIForgeSettings(PROVIDER="test_gateway")
        assert settings.PROVIDER == "test_gateway"

    def test_config_rejects_unregistered_provider(self) -> None:
        """An unknown provider name that's not registered should still be rejected."""
        from aiforge.infra.config import AIForgeSettings

        with pytest.raises(Exception, match="register_provider"):
            AIForgeSettings(PROVIDER="totally_unknown_xyz")

    def test_base_module_provider_kwarg(self) -> None:
        """BaseModule(provider='x') should set settings.PROVIDER to 'x'."""
        from aiforge.registry import register_provider

        class _DummyLLM:
            pass

        class _DummyEmbed:
            pass

        register_provider("my_custom", _DummyLLM, _DummyEmbed)

        instance = _StubModule(provider="my_custom")
        assert instance.settings.PROVIDER == "my_custom"

    def test_base_module_provider_kwarg_invalid_raises(self) -> None:
        """BaseModule(provider='nonexistent') should raise during settings creation."""
        with pytest.raises(ValidationError):
            _StubModule(provider="nonexistent_provider_xyz")

    def test_base_module_kwargs_apply_custom_pricing_overrides(self) -> None:
        """Module kwargs path should still register CUSTOM_PRICING overrides."""
        from aiforge.core.pricing import calculate_cost

        model = "unit-test-custom-pricing-model"
        assert calculate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000) == 0.0

        _StubModule(
            MODEL="gpt-4o",
            CUSTOM_PRICING={model: {"input": 1.25, "output": 2.5}},
        )

        assert calculate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(
            3.75
        )
