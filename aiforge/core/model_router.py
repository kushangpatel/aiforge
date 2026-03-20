"""Model Router — auto-select the cheapest model that meets a quality threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import structlog

from aiforge.core.pricing import MODEL_CONTEXT_WINDOWS, PRICING

logger = structlog.get_logger()

QualityTier = Literal["high", "medium", "low"]

# Tier ordering: high > medium > low
_TIER_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a single model."""

    name: str
    provider: str  # "openai" | "anthropic" | "ollama"
    tier: str  # "high" | "medium" | "low"
    input_cost_per_m: float  # USD per 1M input tokens
    output_cost_per_m: float  # USD per 1M output tokens
    context_window: int  # max tokens


# Built-in catalog — extendable via config or constructor
MODEL_CATALOG: list[ModelSpec] = [
    # OpenAI
    ModelSpec("gpt-4o", "openai", "high", 5.00, 15.00, 128_000),
    ModelSpec("gpt-4o-mini", "openai", "medium", 0.15, 0.60, 128_000),
    ModelSpec("gpt-3.5-turbo", "openai", "low", 0.50, 1.50, 16_385),
    # Anthropic
    ModelSpec(
        "claude-3-5-sonnet-20241022", "anthropic", "high", 3.00, 15.00, 200_000
    ),
    ModelSpec(
        "claude-3-haiku-20240307", "anthropic", "medium", 0.25, 1.25, 200_000
    ),
    # Ollama (local — zero cost)
    ModelSpec("llama3.1", "ollama", "medium", 0.0, 0.0, 131_072),
    ModelSpec("llama3.2", "ollama", "low", 0.0, 0.0, 131_072),
]


class ModelRouter:
    """Select the cheapest model that meets a minimum quality tier.

    Usage::

        router = ModelRouter()
        model = router.select(quality="medium")
        # → "gpt-4o-mini" (cheapest medium-or-above model)

        model = router.select(quality="medium", provider="anthropic")
        # → "claude-3-haiku-20240307"
    """

    def __init__(
        self,
        catalog: list[ModelSpec] | None = None,
        available_providers: list[str] | None = None,
    ) -> None:
        self._catalog = list(MODEL_CATALOG) if catalog is None else catalog
        self._available_providers = available_providers

    def select(
        self,
        quality: QualityTier = "medium",
        *,
        provider: str | None = None,
        prefer_local: bool = False,
        min_context: int = 0,
    ) -> str:
        """Return the cheapest model name meeting the constraints.

        Args:
            quality:       Minimum quality tier.
            provider:      Restrict to a specific provider.
            prefer_local:  Prefer Ollama models (zero cost) over cloud models
                           at the same or higher tier.
            min_context:   Minimum context window size required.

        Returns:
            Model name string (e.g., ``"gpt-4o-mini"``).

        Raises:
            ValueError: if no model satisfies the constraints.
        """
        min_rank = _TIER_RANK.get(quality, 2)
        candidates = [
            m
            for m in self._catalog
            if _TIER_RANK.get(m.tier, 0) >= min_rank
            and (provider is None or m.provider == provider)
            and (
                self._available_providers is None
                or m.provider in self._available_providers
            )
            and m.context_window >= min_context
        ]

        if not candidates:
            raise ValueError(
                f"No model found for quality={quality!r}, "
                f"provider={provider!r}, min_context={min_context}"
            )

        # Sort: prefer_local puts ollama first, then by input cost ascending
        def _sort_key(m: ModelSpec) -> tuple[int, float]:
            local_priority = (
                0 if (prefer_local and m.provider == "ollama") else 1
            )
            return (local_priority, m.input_cost_per_m)

        candidates.sort(key=_sort_key)
        selected = candidates[0]

        logger.debug(
            "model_router.selected",
            model=selected.name,
            provider=selected.provider,
            tier=selected.tier,
            quality_requested=quality,
            input_cost_per_m=selected.input_cost_per_m,
        )

        return selected.name

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Estimate cost in USD for a given model and token count."""
        spec = next((m for m in self._catalog if m.name == model), None)
        if spec is None:
            return 0.0
        return (
            spec.input_cost_per_m * input_tokens / 1_000_000
            + spec.output_cost_per_m * output_tokens / 1_000_000
        )

    def list_models(
        self,
        quality: QualityTier | None = None,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return catalog entries matching optional filters."""
        results = []
        for m in self._catalog:
            if quality and _TIER_RANK.get(m.tier, 0) < _TIER_RANK.get(
                quality, 0
            ):
                continue
            if provider and m.provider != provider:
                continue
            results.append(
                {
                    "name": m.name,
                    "provider": m.provider,
                    "tier": m.tier,
                    "input_cost_per_m": m.input_cost_per_m,
                    "output_cost_per_m": m.output_cost_per_m,
                    "context_window": m.context_window,
                }
            )
        return results
