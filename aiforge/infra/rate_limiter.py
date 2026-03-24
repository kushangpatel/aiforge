"""
Token-bucket rate limiter — thread-safe singleton per process.

Limits both:
  * RPM — requests per minute
  * TPM — tokens per minute

When a limit is hit the caller **blocks** (queues) rather than failing,
matching the behaviour described in NFR-014.

Configuration via env vars::

    AIFORGE_RATE_LIMIT_RPM=500   (default)
    AIFORGE_RATE_LIMIT_TPM=150000  (default)
"""

from __future__ import annotations

import threading
import time
from typing import ClassVar

import structlog

_logger = structlog.get_logger(__name__)

# Module-level singleton
_instance: "RateLimiter | None" = None
_instance_lock = threading.Lock()


class _TokenBucket:
    """A single token bucket (fixed-window, replenished every minute)."""

    def __init__(self, capacity: int, label: str) -> None:
        self._capacity = capacity
        self._label = label
        self._tokens = capacity
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: int = 1) -> None:
        """Consume *n* tokens, blocking until tokens are available."""
        if n <= 0:
            return
        # Clamp to capacity to prevent infinite blocking when a single
        # request exceeds the bucket size (e.g. 200K tokens vs 150K TPM).
        if n > self._capacity:
            _logger.warning(
                "Rate limit: request exceeds bucket capacity, clamping",
                bucket=self._label,
                requested=n,
                capacity=self._capacity,
            )
            n = self._capacity
        while True:
            with self._lock:
                self._replenish()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                # Not enough tokens — calculate sleep duration
                elapsed = time.monotonic() - self._window_start
                sleep_for = max(0.0, 60.0 - elapsed) + 0.05

            _logger.debug(
                "Rate limit: sleeping",
                bucket=self._label,
                needed=n,
                available=self._tokens,
                sleep_seconds=round(sleep_for, 2),
            )
            time.sleep(sleep_for)

    def _replenish(self) -> None:
        """Refill bucket if the current window has expired (every 60 s)."""
        now = time.monotonic()
        if now - self._window_start >= 60.0:
            self._tokens = self._capacity
            self._window_start = now

    def update_capacity(self, capacity: int) -> None:
        with self._lock:
            self._capacity = capacity


class RateLimiter:
    """Process-level rate limiter shared across all threads and module instances.

    Wraps two token buckets: one for request count (RPM) and one for
    token count (TPM).
    """

    _instances: ClassVar[dict[str, "RateLimiter"]] = {}
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, rpm: int = 500, tpm: int = 150_000) -> None:
        self._rpm_bucket = _TokenBucket(capacity=rpm, label="RPM")
        self._tpm_bucket = _TokenBucket(capacity=tpm, label="TPM")

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, rpm: int = 500, tpm: int = 150_000) -> "RateLimiter":
        """Return (or create) the process-level rate limiter singleton."""
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = cls(rpm=rpm, tpm=tpm)
        return _instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (tests only)."""
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, tokens: int = 0) -> None:
        """Acquire one request slot and optionally *tokens* token slots.

        Blocks until both quota buckets have capacity.
        """
        self._rpm_bucket.consume(1)
        if tokens > 0:
            self._tpm_bucket.consume(tokens)

    def record_tokens(self, tokens: int) -> None:
        """Post-hoc TPM accounting — consume tokens after the actual count is known.

        Call this *after* a provider response to enforce TPM limits on
        subsequent requests.
        """
        if tokens > 0:
            self._tpm_bucket.consume(tokens)

    def update_limits(self, rpm: int | None = None, tpm: int | None = None) -> None:
        """Update bucket capacities at runtime (takes effect next window)."""
        if rpm is not None:
            self._rpm_bucket.update_capacity(rpm)
        if tpm is not None:
            self._tpm_bucket.update_capacity(tpm)
