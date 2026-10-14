"""
Token pricing tables and cost calculation helpers.

Prices are in USD per 1,000,000 tokens (1M tokens).
Config overrides via ``aiforge.yaml`` ``pricing:`` section.

Unknown models emit a WARNING and return cost=0.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pricing  (USD / 1M tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # OpenAI GPT-4 family (gpt-4-turbo, gpt-3.5-turbo, o1 and o3-mini shut down
    # 2026-10-23; gpt-4o / gpt-4o-mini snapshots on 2027-01-14)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.5": {"input": 75.00, "output": 150.00},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # OpenAI o-series reasoning models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 10.00, "output": 40.00},
    # OpenAI embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    # Anthropic Claude — current models
    "claude-fable-5-1": {"input": 10.00, "output": 50.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    # Haiku 5.5 rate for prompts up to 100K tokens ($0.50 / $2.50 beyond)
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Anthropic Claude 4.0 (deprecated)
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    # Anthropic Claude 3.x (retired — kept for historical cost reports)
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    # Voyage AI embeddings (used with Anthropic provider)
    "voyage-3": {"input": 0.06, "output": 0.0},
    "voyage-3-lite": {"input": 0.02, "output": 0.0},
    "voyage-large-2": {"input": 0.12, "output": 0.0},
    "voyage-code-2": {"input": 0.12, "output": 0.0},
    # Ollama / local (no cost)
    "llama3": {"input": 0.0, "output": 0.0},
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "llama3.3": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "mistral-nemo": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "phi-3": {"input": 0.0, "output": 0.0},
    "phi-4": {"input": 0.0, "output": 0.0},
    "gemma2": {"input": 0.0, "output": 0.0},
    "qwen2.5": {"input": 0.0, "output": 0.0},
    "nomic-embed-text": {"input": 0.0, "output": 0.0},
    "mxbai-embed-large": {"input": 0.0, "output": 0.0},
    "all-minilm": {"input": 0.0, "output": 0.0},
}

# ---------------------------------------------------------------------------
# Context window registry  (max tokens per model)
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.5": 128_000,
    "gpt-4.5-preview": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "claude-fable-5-1": 1_000_000,
    "claude-opus-5-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-5-5": 1_000_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-7-sonnet-20250219": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-opus": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "llama3": 8_192,
    "llama3.1": 131_072,
    "llama3.2": 131_072,
    "llama3.3": 131_072,
    "mistral": 32_768,
    "mistral-nemo": 128_000,
    "codellama": 16_384,
    "phi-3": 4_096,
    "phi-4": 16_384,
    "gemma2": 8_192,
    "qwen2.5": 128_000,
}

_DEFAULT_CONTEXT_WINDOW = 8_192

# Runtime override tables (populated from config at startup)
_pricing_overrides: dict[str, dict[str, float]] = {}
_context_window_overrides: dict[str, int] = {}


def apply_pricing_overrides(overrides: dict[str, Any]) -> None:
    """Merge runtime pricing overrides (from ``custom_pricing`` in config)."""
    _pricing_overrides.update(overrides)


def apply_context_window_overrides(overrides: dict[str, int]) -> None:
    """Merge runtime context-window overrides (from ``custom_context_windows`` in config)."""
    _context_window_overrides.update(overrides)


def get_model_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for *model*, checking overrides first.

    Returns ``None`` and logs a WARNING if the model is unknown.
    """
    # Normalise model name (strip date-suffixes for lookup flexibility)
    for table in (_pricing_overrides, PRICING):
        if model in table:
            return table[model]
        # Fuzzy match: strip trailing version suffix e.g. "-20241022"
        base = model.rsplit("-", 1)[0] if "-" in model else model
        if base in table:
            return table[base]

    _logger.warning("Unknown model %r — cost will be reported as $0.00", model)
    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate USD cost for a completed LLM or embedding call."""
    pricing = get_model_pricing(model)
    if pricing is None:
        return 0.0
    input_cost = pricing.get("input", 0.0) * input_tokens / 1_000_000
    output_cost = pricing.get("output", 0.0) * output_tokens / 1_000_000
    return input_cost + output_cost


def get_context_window(model: str) -> int:
    """Return the context window (max tokens) for *model*, checking overrides first."""
    for table in (_context_window_overrides, MODEL_CONTEXT_WINDOWS):
        if model in table:
            return table[model]
        base = model.rsplit("-", 1)[0] if "-" in model else model
        if base in table:
            return table[base]
    return _DEFAULT_CONTEXT_WINDOW
