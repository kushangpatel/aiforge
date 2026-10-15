"""
Token pricing tables and cost calculation helpers.

Prices are in USD per 1,000,000 tokens (1M tokens).
Config overrides via ``aiforge.yaml`` ``pricing:`` section.

Unknown models emit a WARNING and return cost=0.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default pricing  (USD / 1M tokens)
# ---------------------------------------------------------------------------


def _cached(input: float, output: float, cache_read: float, cache_write: float) -> dict[str, float]:
    """A pricing entry with explicit prompt-cache read and write prices."""
    return {"input": input, "output": output, "cache_read": cache_read, "cache_write": cache_write}


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
    # Google Gemini (Gemini Developer API, Standard tier, checked 2026-10-15:
    # https://ai.google.dev/gemini-api/docs/pricing). Pro models are listed at
    # the <=200K-token prompt rate; longer prompts cost more.
    # gemini-4-argon rises to $4 / $20 (cache $0.20) on 2026-12-01, and
    # gemini-3.8-flash / 3.6-flash to $1.50 / $7.50 (cache $0.15) on 2027-01-01.
    "gemini-4-argon": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75, "cache_read": 0.075},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cache_read": 0.03},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cache_read": 0.025},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00, "cache_read": 0.20},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cache_read": 0.125},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cache_read": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cache_read": 0.01},
    "gemini-embedding-2": {"input": 0.20, "output": 0.0},
    # Amazon Bedrock, US East (N. Virginia) on-demand, checked 2026-10-15 via the
    # AWS Price List API. Keys are Bedrock model IDs without the version suffix;
    # geo profiles (us., eu., …) and in-region IDs share the plain key, and
    # "global." profiles have their own (cheaper) key. See _bedrock_candidates().
    "global.anthropic.claude-fable-5-1": _cached(10.00, 50.00, 0.25, 12.50),
    "anthropic.claude-fable-5-1": _cached(11.00, 55.00, 0.275, 13.75),
    "global.anthropic.claude-opus-5-5": _cached(4.00, 20.00, 0.20, 5.00),
    "anthropic.claude-opus-5-5": _cached(4.40, 22.00, 0.22, 5.50),
    "global.anthropic.claude-sonnet-5-5": _cached(2.00, 10.00, 0.10, 2.50),
    "anthropic.claude-sonnet-5-5": _cached(2.20, 11.00, 0.11, 2.75),
    # Haiku 5.5: rate for prompts up to 100K tokens (5x beyond).
    "global.anthropic.claude-haiku-5-5": _cached(0.10, 0.50, 0.01, 0.125),
    "anthropic.claude-haiku-5-5": _cached(0.11, 0.55, 0.011, 0.1375),
    "global.anthropic.claude-sonnet-4-6": _cached(3.00, 15.00, 0.30, 3.75),
    "anthropic.claude-sonnet-4-6": _cached(3.30, 16.50, 0.33, 4.125),
    "global.anthropic.claude-haiku-4-5": _cached(1.00, 5.00, 0.10, 1.25),
    "anthropic.claude-haiku-4-5": _cached(1.10, 5.50, 0.11, 1.375),
    # Amazon Nova: cache writes are not charged.
    "global.amazon.nova-2-lite-v1": _cached(0.30, 2.50, 0.075, 0.0),
    "amazon.nova-2-lite-v1": _cached(0.33, 2.75, 0.0825, 0.0),
    "amazon.nova-pro-v1": _cached(0.80, 3.20, 0.20, 0.0),
    "amazon.nova-lite-v1": _cached(0.06, 0.24, 0.015, 0.0),
    "amazon.nova-micro-v1": _cached(0.035, 0.14, 0.00875, 0.0),
    "amazon.titan-embed-text-v2": {"input": 0.02, "output": 0.0},
    "amazon.titan-embed-text-v1": {"input": 0.10, "output": 0.0},
    # Mistral AI (https://docs.mistral.ai/inference/pricing, checked 2026-10-15).
    # Cached input is billed at 10% of the input price.
    # Mistral Large 4 (public preview) at its sale price; list is $1.36 / $4.18.
    "mistral-large-4": {"input": 0.68, "output": 2.09, "cache_read": 0.07},
    "mistral-large-2512": {"input": 0.50, "output": 1.50, "cache_read": 0.05},
    "mistral-medium-3-5": {"input": 1.50, "output": 7.50, "cache_read": 0.15},
    "mistral-small-2603": {"input": 0.15, "output": 0.60, "cache_read": 0.015},
    "codestral-2508": {"input": 0.30, "output": 0.90, "cache_read": 0.03},
    "mistral-embed": {"input": 0.10, "output": 0.0},
    "mistral-embed-2312": {"input": 0.10, "output": 0.0},
    "codestral-embed": {"input": 0.15, "output": 0.0},
    "codestral-embed-2505": {"input": 0.15, "output": 0.0},
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
    "gemini-4-argon": 1_048_576,
    "gemini-3.8-flash": 1_048_576,
    "gemini-3.6-flash": 1_048_576,
    "gemini-3.5-flash-lite": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "anthropic.claude-fable-5-1": 1_000_000,
    "anthropic.claude-opus-5-5": 1_000_000,
    "anthropic.claude-sonnet-5-5": 1_000_000,
    "anthropic.claude-haiku-5-5": 1_000_000,
    "anthropic.claude-sonnet-4-6": 1_000_000,
    "anthropic.claude-haiku-4-5": 200_000,
    "amazon.nova-2-lite-v1": 1_000_000,
    "amazon.nova-pro-v1": 300_000,
    "amazon.nova-lite-v1": 300_000,
    "amazon.nova-micro-v1": 128_000,
    "mistral-large-4": 1_000_000,
    "mistral-large-2512": 256_000,
    "mistral-medium-3-5": 256_000,
    "mistral-small-2603": 256_000,
    "codestral-2508": 128_000,
    "mistral-embed": 8_192,
    "mistral-embed-2312": 8_192,
    "codestral-embed": 8_192,
    "codestral-embed-2505": 8_192,
    "gemini-embedding-2": 8_192,
    "gemini-embedding-001": 2_048,
    "amazon.titan-embed-text-v2": 8_192,
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


_BEDROCK_GEO_PREFIXES = frozenset({"us", "us-gov", "eu", "apac", "jp", "au", "ca", "in"})


def _bedrock_candidates(model: str) -> list[str]:
    """Lookup keys for an Amazon Bedrock model or inference-profile ID.

    ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` → ``anthropic.claude-haiku-4-5``;
    ``global.amazon.nova-2-lite-v1:0`` → ``global.amazon.nova-2-lite-v1`` then
    ``amazon.nova-2-lite-v1``. Non-Bedrock names come back unchanged.
    """
    name = re.sub(r":\d+$", "", model)  # ":0" version suffix
    name = re.sub(r"-\d{8}-v\d+$", "", name)  # "-20251001-v1" snapshot suffix
    prefix, dot, rest = name.partition(".")
    if dot and prefix in _BEDROCK_GEO_PREFIXES:
        return [rest]
    if dot and prefix == "global":
        return [name, rest]
    return [name]


def _lookup(model: str, tables: tuple[dict[str, Any], ...]) -> Any:
    candidates = [model, *_bedrock_candidates(model)]
    for table in tables:
        for name in candidates:
            if name in table:
                return table[name]
        # Fuzzy match: strip trailing version suffix e.g. "-20241022"
        base = model.rsplit("-", 1)[0] if "-" in model else model
        if base in table:
            return table[base]
    return None


def get_model_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for *model*, checking overrides first.

    Returns ``None`` and logs a WARNING if the model is unknown.
    """
    found: dict[str, float] | None = _lookup(model, (_pricing_overrides, PRICING))
    if found is not None:
        return found

    _logger.warning("Unknown model %r — cost will be reported as $0.00", model)
    return None


def _cache_multipliers(model: str) -> tuple[float, float]:
    """Default (read, write) price multipliers for prompt-cached input tokens.

    Anthropic bills cache reads at 0.1x and 5-minute cache writes at 1.25x the
    input price. OpenAI-style automatic caching bills reads at 0.5x and adds
    no write surcharge. A model's pricing entry can override either with
    explicit ``cache_read`` / ``cache_write`` prices.
    """
    if model.lower().split("/")[-1].split("anthropic.", 1)[-1].startswith("claude"):
        return 0.1, 1.25
    return 0.5, 1.0


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Calculate USD cost for a completed LLM or embedding call.

    *input_tokens* is the total prompt size; the cached portions given by
    *cache_read_tokens* and *cache_write_tokens* are priced at their own rates.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return 0.0
    input_price = pricing.get("input", 0.0)
    read_mult, write_mult = _cache_multipliers(model)
    read_price = pricing.get("cache_read", input_price * read_mult)
    write_price = pricing.get("cache_write", input_price * write_mult)
    uncached = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    input_cost = (
        input_price * uncached + read_price * cache_read_tokens + write_price * cache_write_tokens
    ) / 1_000_000
    output_cost = pricing.get("output", 0.0) * output_tokens / 1_000_000
    return input_cost + output_cost


def get_context_window(model: str) -> int:
    """Return the context window (max tokens) for *model*, checking overrides first."""
    found: int | None = _lookup(model, (_context_window_overrides, MODEL_CONTEXT_WINDOWS))
    return found if found is not None else _DEFAULT_CONTEXT_WINDOW
