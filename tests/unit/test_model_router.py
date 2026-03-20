"""Unit tests for the Model Router."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from aiforge.core.model_router import (
    MODEL_CATALOG,
    ModelRouter,
    ModelSpec,
    _TIER_RANK,
)


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_frozen(self) -> None:
        spec = ModelSpec("m", "openai", "high", 1.0, 2.0, 128_000)
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        spec = ModelSpec("gpt-4o", "openai", "high", 5.0, 15.0, 128_000)
        assert spec.name == "gpt-4o"
        assert spec.provider == "openai"
        assert spec.tier == "high"
        assert spec.input_cost_per_m == 5.0
        assert spec.output_cost_per_m == 15.0
        assert spec.context_window == 128_000


# ---------------------------------------------------------------------------
# ModelRouter.select()
# ---------------------------------------------------------------------------


class TestSelect:
    def test_medium_returns_cheapest(self) -> None:
        router = ModelRouter()
        model = router.select("medium")
        # Ollama llama3.1 is medium tier with 0.0 cost — cheapest
        assert model == "llama3.1"

    def test_high_returns_high_tier(self) -> None:
        router = ModelRouter()
        model = router.select("high")
        # Only high-tier models; cheapest is claude-3-5-sonnet at $3.00
        spec = next(m for m in MODEL_CATALOG if m.name == model)
        assert spec.tier == "high"

    def test_low_includes_all(self) -> None:
        router = ModelRouter()
        model = router.select("low")
        # Low tier means any model qualifies; cheapest wins
        assert model is not None

    def test_provider_filter(self) -> None:
        router = ModelRouter()
        model = router.select("medium", provider="anthropic")
        assert model == "claude-3-haiku-20240307"

    def test_prefer_local(self) -> None:
        router = ModelRouter()
        model = router.select("medium", prefer_local=True)
        spec = next(m for m in MODEL_CATALOG if m.name == model)
        assert spec.provider == "ollama"

    def test_min_context_filters(self) -> None:
        router = ModelRouter()
        model = router.select("medium", min_context=200_000)
        spec = next(m for m in MODEL_CATALOG if m.name == model)
        assert spec.context_window >= 200_000

    def test_no_match_raises(self) -> None:
        router = ModelRouter()
        with pytest.raises(ValueError, match="No model found"):
            router.select("high", provider="ollama")

    def test_cost_ordering(self) -> None:
        """When multiple candidates exist, cheapest input cost is selected."""
        catalog = [
            ModelSpec("expensive", "openai", "medium", 10.0, 30.0, 128_000),
            ModelSpec("cheap", "openai", "medium", 0.1, 0.3, 128_000),
            ModelSpec("mid", "openai", "medium", 1.0, 3.0, 128_000),
        ]
        router = ModelRouter(catalog=catalog)
        assert router.select("medium") == "cheap"

    def test_available_providers_filter(self) -> None:
        router = ModelRouter(available_providers=["openai"])
        model = router.select("medium")
        spec = next(m for m in MODEL_CATALOG if m.name == model)
        assert spec.provider == "openai"

    def test_provider_and_quality_combined(self) -> None:
        router = ModelRouter()
        model = router.select("high", provider="openai")
        assert model == "gpt-4o"


# ---------------------------------------------------------------------------
# ModelRouter.estimate_cost()
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_known_model(self) -> None:
        router = ModelRouter()
        cost = router.estimate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
        # 5.00 * 1 + 15.00 * 0.5 = 12.50
        assert cost == pytest.approx(12.50)

    def test_unknown_model(self) -> None:
        router = ModelRouter()
        assert router.estimate_cost("nonexistent-model", input_tokens=1000) == 0.0

    def test_zero_cost_ollama(self) -> None:
        router = ModelRouter()
        cost = router.estimate_cost("llama3.1", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 0.0

    def test_input_only(self) -> None:
        router = ModelRouter()
        cost = router.estimate_cost("gpt-4o-mini", input_tokens=2_000_000)
        # 0.15 * 2 = 0.30
        assert cost == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# ModelRouter.list_models()
# ---------------------------------------------------------------------------


class TestListModels:
    def test_no_filter(self) -> None:
        router = ModelRouter()
        models = router.list_models()
        assert len(models) == len(MODEL_CATALOG)

    def test_quality_filter(self) -> None:
        router = ModelRouter()
        models = router.list_models(quality="high")
        assert all(_TIER_RANK[m["tier"]] >= _TIER_RANK["high"] for m in models)
        assert len(models) < len(MODEL_CATALOG)

    def test_provider_filter(self) -> None:
        router = ModelRouter()
        models = router.list_models(provider="anthropic")
        assert all(m["provider"] == "anthropic" for m in models)

    def test_dict_structure(self) -> None:
        router = ModelRouter()
        models = router.list_models()
        for m in models:
            assert "name" in m
            assert "provider" in m
            assert "tier" in m
            assert "input_cost_per_m" in m
            assert "output_cost_per_m" in m
            assert "context_window" in m


# ---------------------------------------------------------------------------
# Custom catalog
# ---------------------------------------------------------------------------


class TestCustomCatalog:
    def test_custom_catalog_used(self) -> None:
        custom = [
            ModelSpec("my-model", "openai", "high", 1.0, 2.0, 64_000),
        ]
        router = ModelRouter(catalog=custom)
        assert router.select("high") == "my-model"
        assert len(router.list_models()) == 1

    def test_empty_catalog_raises(self) -> None:
        router = ModelRouter(catalog=[])
        with pytest.raises(ValueError, match="No model found"):
            router.select("medium")


# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------


class TestConfigKeys:
    def test_default_values(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}, clear=False):
            from aiforge.infra.config import AIForgeSettings
            s = AIForgeSettings()
            assert s.MODEL_ROUTER_ENABLED is True
            assert s.DEFAULT_QUALITY == "medium"
            assert s.PREFER_LOCAL is False

    def test_custom_values(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}, clear=False):
            from aiforge.infra.config import AIForgeSettings
            s = AIForgeSettings(
                MODEL_ROUTER_ENABLED=False,
                DEFAULT_QUALITY="high",
                PREFER_LOCAL=True,
            )
            assert s.MODEL_ROUTER_ENABLED is False
            assert s.DEFAULT_QUALITY == "high"
            assert s.PREFER_LOCAL is True


# ---------------------------------------------------------------------------
# BaseModule integration — quality= wiring
# ---------------------------------------------------------------------------


class TestBaseModuleRouterIntegration:
    """Verify that quality= in BaseModule.__init__ routes via ModelRouter."""

    def test_quality_routes_model(self) -> None:
        """quality='medium' + PROVIDER='openai' → gpt-4o-mini."""
        from aiforge.modules.classify import Classifier

        clf = Classifier(labels=["a", "b"], quality="medium")
        assert clf.settings.MODEL == "gpt-4o-mini"

    def test_quality_high_routes_model(self) -> None:
        """quality='high' + PROVIDER='openai' → gpt-4o."""
        from aiforge.modules.classify import Classifier

        clf = Classifier(labels=["a", "b"], quality="high")
        assert clf.settings.MODEL == "gpt-4o"

    def test_quality_low_routes_model(self) -> None:
        """quality='low' + PROVIDER='openai' → cheapest low-or-above model."""
        from aiforge.modules.classify import Classifier

        clf = Classifier(labels=["a", "b"], quality="low")
        # gpt-4o-mini ($0.15) is cheapest openai model at low-or-above tier
        assert clf.settings.MODEL == "gpt-4o-mini"

    def test_explicit_model_overrides_quality(self) -> None:
        """Explicit MODEL= takes priority over quality=."""
        from aiforge.modules.classify import Classifier

        clf = Classifier(labels=["a", "b"], quality="low", MODEL="gpt-4o")
        assert clf.settings.MODEL == "gpt-4o"

    def test_router_disabled_ignores_quality(self) -> None:
        """MODEL_ROUTER_ENABLED=False → quality= is ignored."""
        from aiforge.infra.config import AIForgeSettings
        from aiforge.modules.classify import Classifier

        default_model = AIForgeSettings().MODEL
        clf = Classifier(
            labels=["a", "b"],
            quality="low",
            MODEL_ROUTER_ENABLED=False,
        )
        # Keeps the default model since router is disabled
        assert clf.settings.MODEL == default_model

    def test_no_quality_keeps_default_model(self) -> None:
        """No quality= → settings.MODEL is unchanged."""
        from aiforge.infra.config import get_settings
        from aiforge.modules.classify import Classifier

        clf = Classifier(labels=["a", "b"])
        assert clf.settings.MODEL == get_settings().MODEL

    def test_quality_with_anthropic_provider(self) -> None:
        """quality='medium' + PROVIDER='anthropic' → claude-3-haiku."""
        from aiforge.modules.classify import Classifier

        clf = Classifier(
            labels=["a", "b"],
            quality="medium",
            PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-fake",
            EMBEDDING_MODEL="voyage-3",
        )
        assert clf.settings.MODEL == "claude-3-haiku-20240307"

    def test_quality_does_not_mutate_singleton(self) -> None:
        """Routing creates a private settings copy, not mutating the global."""
        from aiforge.infra.config import get_settings
        from aiforge.modules.classify import Classifier

        original_model = get_settings().MODEL
        clf = Classifier(labels=["a", "b"], quality="low")
        assert get_settings().MODEL == original_model

    def test_no_matching_model_falls_back(self) -> None:
        """If router can't match, keeps default model."""
        from aiforge.infra.config import AIForgeSettings
        from aiforge.modules.classify import Classifier

        default_model = AIForgeSettings().MODEL
        with patch(
            "aiforge.core.model_router.ModelRouter.select",
            side_effect=ValueError("No model found"),
        ):
            clf = Classifier(labels=["a", "b"], quality="high")
            assert clf.settings.MODEL == default_model
