"""
Session-level cost tracker — thread-safe singleton per process.

Usage::

    from aiforge import CostTracker

    tracker = CostTracker.get_session()
    print(tracker.total_cost)
    print(tracker.by_module)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import structlog

_logger = structlog.get_logger(__name__)

# Module-level singleton
_session: CostTracker | None = None
_lock = threading.Lock()


class CostTracker:
    """Accumulates token usage and USD cost across all module calls.

    All public methods are thread-safe.
    """

    def __init__(self, warn_threshold: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._warn_threshold = warn_threshold
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._by_module: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_session(cls, warn_threshold: float = 1.0) -> "CostTracker":
        """Return (or create) the process-level session singleton."""
        global _session
        if _session is None:
            with _lock:
                if _session is None:
                    _session = cls(warn_threshold=warn_threshold)
        return _session

    @classmethod
    def reset_session(cls) -> None:
        """Reset the session singleton (primarily for tests)."""
        global _session
        with _lock:
            _session = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        module: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Record a completed LLM/embedding call."""
        with self._lock:
            self._total_cost += cost
            self._total_tokens += input_tokens + output_tokens
            self._by_module[module] += cost
            self._by_model[model] += cost
            self._call_count += 1

        if cost > self._warn_threshold:
            _logger.warning(
                "High-cost LLM call",
                cost_usd=f"${cost:.4f}",
                threshold_usd=f"${self._warn_threshold:.4f}",
                module=module,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def by_module(self) -> dict[str, float]:
        with self._lock:
            return dict(self._by_module)

    @property
    def by_model(self) -> dict[str, float]:
        with self._lock:
            return dict(self._by_model)

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of all tracked metrics."""
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_tokens": self._total_tokens,
                "call_count": self._call_count,
                "by_module": dict(self._by_module),
                "by_model": dict(self._by_model),
            }

    def reset(self) -> None:
        """Reset all counters (resets this instance, not the singleton)."""
        with self._lock:
            self._total_cost = 0.0
            self._total_tokens = 0
            self._call_count = 0
            self._by_module.clear()
            self._by_model.clear()
