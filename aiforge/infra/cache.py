"""Request Cache — cache identical LLM requests to reduce cost and latency.

A SHA-256 hash of the full request fingerprint (model, messages, temperature,
etc.) serves as the cache key.  Two backends are shipped:

* **MemoryBackend** — in-process ``dict`` with TTL expiration and LRU eviction.
* **RedisBackend** — shared cache via Redis (requires the ``redis`` package).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class CacheBackend(ABC):
    """Abstract backend for cache storage."""

    @abstractmethod
    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached value or ``None``."""

    @abstractmethod
    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """Store a value with a TTL."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a key."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def size(self) -> int:
        """Return number of cached entries."""


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class MemoryBackend(CacheBackend):
    """In-process dictionary cache with TTL expiration and max-entry eviction."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: dict[str, tuple[dict[str, Any], float]] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        with self._lock:
            # Evict oldest entry when at capacity (FIFO / insertion-order).
            if len(self._store) >= self._max_entries and key not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            expires_at = time.monotonic() + ttl_seconds
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for _, (_, exp) in self._store.items() if now <= exp)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


class RedisBackend(CacheBackend):
    """Redis-backed cache.  Requires the ``redis`` package.

    Only instantiated when ``AIFORGE_CACHE_BACKEND=redis``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "aiforge:cache:",
    ) -> None:
        import redis  # deferred — only needed when Redis is selected

        self._client = redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def get(self, key: str) -> dict[str, Any] | None:
        raw = self._client.get(self._prefix + key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._client.setex(self._prefix + key, ttl_seconds, json.dumps(value))

    def delete(self, key: str) -> None:
        self._client.delete(self._prefix + key)

    def clear(self) -> None:
        keys = self._client.keys(self._prefix + "*")
        if keys:
            self._client.delete(*keys)

    def size(self) -> int:
        return len(self._client.keys(self._prefix + "*"))


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


def _make_cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    module: str,
    provider: str = "",
) -> str:
    """Produce a deterministic SHA-256 cache key from request parameters."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "module": module,
            "provider": provider,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RequestCache
# ---------------------------------------------------------------------------


class RequestCache:
    """Cache for LLM request/response pairs.

    Usage::

        cache = RequestCache()
        key = cache.build_key(model, messages, temperature, max_tokens, module)
        cached = cache.get(key)
        if cached is not None:
            return cached          # cache hit — skip LLM call

        result = provider.complete(messages, ...)
        cache.put(key, result_dict)
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ) -> None:
        self._backend = backend or MemoryBackend()
        self._ttl = ttl_seconds
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        self._counter_lock = threading.Lock()

    # -- Properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    # -- Public API --------------------------------------------------------

    def build_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        module: str,
        provider: str = "",
    ) -> str:
        """Build a deterministic cache key from request parameters."""
        return _make_cache_key(model, messages, temperature, max_tokens, module, provider)

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached response.  Returns ``None`` on miss or if disabled."""
        if not self._enabled:
            return None
        result = self._backend.get(key)
        if result is not None:
            with self._counter_lock:
                self._hits += 1
            logger.debug("request_cache.hit", key=key[:16])
        else:
            with self._counter_lock:
                self._misses += 1
            logger.debug("request_cache.miss", key=key[:16])
        return result

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a response in the cache."""
        if not self._enabled:
            return
        self._backend.set(key, value, self._ttl)
        logger.debug("request_cache.stored", key=key[:16])

    def invalidate(self, key: str) -> None:
        """Remove a specific entry."""
        self._backend.delete(key)

    def clear(self) -> None:
        """Clear all cached entries and reset hit/miss counters."""
        self._backend.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "enabled": self._enabled,
            "size": self._backend.size(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl_seconds": self._ttl,
        }
