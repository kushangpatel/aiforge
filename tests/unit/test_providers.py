"""Unit tests for the provider registry and pricing (providers/__init__.py + core/pricing.py)."""

from __future__ import annotations

import pytest

from aiforge.core.pricing import (
    _context_window_overrides,
    _pricing_overrides,
    apply_context_window_overrides,
    apply_pricing_overrides,
    calculate_cost,
    get_context_window,
    get_model_pricing,
)
from aiforge.core.types import APIConnectionError, CompletionResult, ProviderError


class TestPricing:
    def test_known_model_gpt4o(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert pytest.approx(cost) == 5.00

    def test_known_model_output_cost(self) -> None:
        cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert pytest.approx(cost) == 15.00

    def test_embedding_only_has_input_cost(self) -> None:
        cost = calculate_cost("text-embedding-3-small", input_tokens=1_000_000, output_tokens=0)
        assert pytest.approx(cost) == 0.02

    def test_free_local_model(self) -> None:
        cost = calculate_cost("llama3", input_tokens=50_000, output_tokens=10_000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        cost = calculate_cost("totally-unknown-model-xyz", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_proportional_cost(self) -> None:
        cost_half = calculate_cost("gpt-4o", input_tokens=500_000, output_tokens=0)
        cost_full = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert pytest.approx(cost_half) == cost_full / 2


class TestContextWindow:
    def test_gpt4o_context_window(self) -> None:
        assert get_context_window("gpt-4o") == 128_000

    def test_claude_context_window(self) -> None:
        assert get_context_window("claude-3-5-sonnet-20241022") == 200_000

    def test_unknown_model_returns_default(self) -> None:
        assert get_context_window("some-unknown-model") == 8_192


class TestCustomPricingOverrides:
    """Custom pricing via apply_pricing_overrides."""

    def setup_method(self) -> None:
        _pricing_overrides.clear()

    def teardown_method(self) -> None:
        _pricing_overrides.clear()

    def test_override_existing_model(self) -> None:
        apply_pricing_overrides({"gpt-4o": {"input": 1.0, "output": 2.0}})
        cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert pytest.approx(cost) == 3.0

    def test_add_new_model(self) -> None:
        apply_pricing_overrides({"my-custom-model": {"input": 10.0, "output": 20.0}})
        pricing = get_model_pricing("my-custom-model")
        assert pricing == {"input": 10.0, "output": 20.0}
        cost = calculate_cost("my-custom-model", input_tokens=1_000_000, output_tokens=500_000)
        assert pytest.approx(cost) == 20.0  # 10.0 + (20.0 * 0.5)

    def test_override_takes_precedence(self) -> None:
        original = get_model_pricing("gpt-4o")
        assert original is not None
        assert original["input"] == 5.00
        apply_pricing_overrides({"gpt-4o": {"input": 99.0, "output": 99.0}})
        overridden = get_model_pricing("gpt-4o")
        assert overridden is not None
        assert overridden["input"] == 99.0

    def test_override_with_fuzzy_match(self) -> None:
        apply_pricing_overrides({"claude-3-5-sonnet": {"input": 1.0, "output": 5.0}})
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing["input"] == 1.0


class TestCustomContextWindowOverrides:
    """Custom context windows via apply_context_window_overrides."""

    def setup_method(self) -> None:
        _context_window_overrides.clear()

    def teardown_method(self) -> None:
        _context_window_overrides.clear()

    def test_override_existing_model(self) -> None:
        apply_context_window_overrides({"gpt-4o": 256_000})
        assert get_context_window("gpt-4o") == 256_000

    def test_add_new_model(self) -> None:
        apply_context_window_overrides({"my-custom-llm": 32_768})
        assert get_context_window("my-custom-llm") == 32_768

    def test_override_takes_precedence(self) -> None:
        assert get_context_window("gpt-4o") == 128_000  # built-in
        apply_context_window_overrides({"gpt-4o": 64_000})
        assert get_context_window("gpt-4o") == 64_000

    def test_unknown_still_returns_default(self) -> None:
        assert get_context_window("totally-unknown") == 8_192


class TestProviderRegistry:
    def test_openai_provider_registered(self) -> None:
        from aiforge.providers import list_providers

        assert "openai" in list_providers()

    def test_register_custom_provider(self) -> None:
        from aiforge.core.provider import EmbeddingProvider, LLMProvider
        from aiforge.providers import list_providers, register_provider

        class FakeLLM(LLMProvider):
            @property
            def model_name(self) -> str:
                return "fake"

            def complete(self, messages, **kwargs) -> CompletionResult:
                return CompletionResult(text="fake", input_tokens=0, output_tokens=0, model="fake")

            async def complete_async(self, messages, **kwargs) -> CompletionResult:
                return self.complete(messages)

            async def stream(self, messages, **kwargs):
                yield "fake"

        class FakeEmbedding(EmbeddingProvider):
            @property
            def model_name(self) -> str:
                return "fake-embed"

            def embed(self, texts):
                return [[0.0] * 4 for _ in texts]

            async def embed_async(self, texts):
                return self.embed(texts)

            def embed_query(self, text):
                return [0.0] * 4

            async def embed_query_async(self, text):
                return self.embed_query(text)

        register_provider("fake_provider", FakeLLM, FakeEmbedding)
        assert "fake_provider" in list_providers()

    def test_unknown_provider_raises_config_error(self, test_settings) -> None:
        from aiforge.core.types import ConfigError
        from aiforge.providers import get_provider

        test_settings.PROVIDER = "nonexistent_xyz"
        # Temporarily remove from registry check
        with pytest.raises(ConfigError, match="not available"):
            get_provider(test_settings)


class TestCompletionResult:
    def test_total_tokens_property(self) -> None:
        result = CompletionResult(
            text="hello",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
        )
        assert result.total_tokens == 150

    def test_defaults(self) -> None:
        result = CompletionResult(
            text="hi",
            input_tokens=0,
            output_tokens=0,
            model="gpt-4o",
        )
        assert result.finish_reason == "stop"
        assert result.latency_ms == 0.0
        assert result.raw_response is None


class TestOllamaErrorWrapper:
    """Verify _wrap_ollama_error is a callable module-level function."""

    def test_connection_error_returns_api_connection_error(self) -> None:
        from aiforge.providers.ollama import _wrap_ollama_error

        err = _wrap_ollama_error(ConnectionError("Connection refused"))
        assert isinstance(err, APIConnectionError)
        assert "Connection refused" in str(err)

    def test_generic_error_returns_provider_error(self) -> None:
        from aiforge.providers.ollama import _wrap_ollama_error

        err = _wrap_ollama_error(RuntimeError("model not found"))
        assert isinstance(err, ProviderError)
        assert "model not found" in str(err)
