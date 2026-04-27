"""Unit tests for aiforge.infra.cache — Request Cache."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiforge.infra.cache import (
    CacheBackend,
    MemoryBackend,
    RequestCache,
    _make_cache_key,
)

# ======================================================================
# Cache key determinism
# ======================================================================


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        msgs = [{"role": "user", "content": "hi"}]
        k1 = _make_cache_key("gpt-4o", msgs, 0.0, None, "classify")
        k2 = _make_cache_key("gpt-4o", msgs, 0.0, None, "classify")
        assert k1 == k2

    def test_varies_on_model(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = _make_cache_key("gpt-4o", msgs, 0.0, None, "classify")
        k2 = _make_cache_key("gpt-4o-mini", msgs, 0.0, None, "classify")
        assert k1 != k2

    def test_varies_on_messages(self) -> None:
        k1 = _make_cache_key(
            "gpt-4o", [{"role": "user", "content": "hello"}], 0.0, None, "classify"
        )
        k2 = _make_cache_key("gpt-4o", [{"role": "user", "content": "bye"}], 0.0, None, "classify")
        assert k1 != k2

    def test_varies_on_temperature(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = _make_cache_key("gpt-4o", msgs, 0.0, None, "classify")
        k2 = _make_cache_key("gpt-4o", msgs, 0.7, None, "classify")
        assert k1 != k2

    def test_varies_on_max_tokens(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = _make_cache_key("gpt-4o", msgs, 0.0, 100, "classify")
        k2 = _make_cache_key("gpt-4o", msgs, 0.0, 200, "classify")
        assert k1 != k2

    def test_varies_on_module(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = _make_cache_key("gpt-4o", msgs, 0.0, None, "classify")
        k2 = _make_cache_key("gpt-4o", msgs, 0.0, None, "summarize")
        assert k1 != k2

    def test_key_is_hex_sha256(self) -> None:
        key = _make_cache_key("gpt-4o", [{"role": "user", "content": "hi"}], 0.0, None, "x")
        assert len(key) == 64
        int(key, 16)  # must be valid hex


# ======================================================================
# MemoryBackend
# ======================================================================


class TestMemoryBackend:
    def test_put_and_get(self) -> None:
        b = MemoryBackend()
        b.set("k", {"a": 1}, ttl_seconds=60)
        assert b.get("k") == {"a": 1}

    def test_get_miss(self) -> None:
        b = MemoryBackend()
        assert b.get("missing") is None

    def test_ttl_expiration(self) -> None:
        b = MemoryBackend()
        b.set("k", {"a": 1}, ttl_seconds=0)
        # TTL of 0 = already expired on next monotonic tick
        time.sleep(0.01)
        assert b.get("k") is None

    def test_delete(self) -> None:
        b = MemoryBackend()
        b.set("k", {"a": 1}, ttl_seconds=60)
        b.delete("k")
        assert b.get("k") is None

    def test_delete_missing_key_is_noop(self) -> None:
        b = MemoryBackend()
        b.delete("nope")  # should not raise

    def test_clear(self) -> None:
        b = MemoryBackend()
        b.set("k1", {"a": 1}, ttl_seconds=60)
        b.set("k2", {"a": 2}, ttl_seconds=60)
        b.clear()
        assert b.size() == 0

    def test_size(self) -> None:
        b = MemoryBackend()
        assert b.size() == 0
        b.set("k1", {}, ttl_seconds=60)
        assert b.size() == 1
        b.set("k2", {}, ttl_seconds=60)
        assert b.size() == 2

    def test_max_entries_eviction(self) -> None:
        b = MemoryBackend(max_entries=2)
        b.set("k1", {"v": 1}, ttl_seconds=60)
        b.set("k2", {"v": 2}, ttl_seconds=60)
        b.set("k3", {"v": 3}, ttl_seconds=60)  # should evict k1
        assert b.size() == 2
        assert b.get("k1") is None
        assert b.get("k2") == {"v": 2}
        assert b.get("k3") == {"v": 3}

    def test_update_existing_key_no_eviction(self) -> None:
        """Updating an existing key should not trigger eviction."""
        b = MemoryBackend(max_entries=2)
        b.set("k1", {"v": 1}, ttl_seconds=60)
        b.set("k2", {"v": 2}, ttl_seconds=60)
        b.set("k1", {"v": 99}, ttl_seconds=60)  # update, not new
        assert b.size() == 2
        assert b.get("k1") == {"v": 99}
        assert b.get("k2") == {"v": 2}

    def test_concurrent_access_thread_safe(self) -> None:
        """MemoryBackend should be safe under concurrent access."""
        import threading

        b = MemoryBackend(max_entries=500)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    b.set(f"k{i}", {"v": i}, ttl_seconds=60)
            except Exception as exc:
                errors.append(exc)

        def reader(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    b.get(f"k{i}")
                    b.size()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader, args=(0,)),
            threading.Thread(target=reader, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ======================================================================
# CacheBackend ABC
# ======================================================================


class TestCacheBackendABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            CacheBackend()  # type: ignore[abstract]


# ======================================================================
# RequestCache
# ======================================================================


class TestRequestCache:
    def test_put_and_get(self) -> None:
        cache = RequestCache()
        key = cache.build_key("m", [{"role": "user", "content": "hi"}], 0.0, None, "mod")
        cache.put(key, {"text": "hello"})
        assert cache.get(key) == {"text": "hello"}

    def test_get_miss_returns_none(self) -> None:
        cache = RequestCache()
        assert cache.get("nonexistent") is None

    def test_disabled_cache_always_misses(self) -> None:
        cache = RequestCache(enabled=False)
        key = "abc"
        cache.put(key, {"x": 1})
        assert cache.get(key) is None

    def test_disabled_cache_skips_put(self) -> None:
        backend = MemoryBackend()
        cache = RequestCache(backend=backend, enabled=False)
        cache.put("k", {"x": 1})
        assert backend.size() == 0

    def test_hit_miss_counters(self) -> None:
        cache = RequestCache()
        key = cache.build_key("m", [{"role": "user", "content": "hi"}], 0.0, None, "mod")

        cache.get(key)  # miss
        assert cache.hits == 0
        assert cache.misses == 1

        cache.put(key, {"text": "x"})
        cache.get(key)  # hit
        assert cache.hits == 1
        assert cache.misses == 1

    def test_hit_rate(self) -> None:
        cache = RequestCache()
        assert cache.hit_rate == 0.0  # no calls yet

        key = "k"
        cache.put(key, {"x": 1})
        cache.get(key)  # hit
        cache.get("nope")  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_stats_snapshot(self) -> None:
        cache = RequestCache(ttl_seconds=1800)
        key = "k"
        cache.put(key, {"x": 1})
        cache.get(key)
        cache.get("miss_key")
        s = cache.stats()
        assert s["enabled"] is True
        assert s["size"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["ttl_seconds"] == 1800

    def test_invalidate_removes_entry(self) -> None:
        cache = RequestCache()
        key = "k"
        cache.put(key, {"x": 1})
        cache.invalidate(key)
        assert cache.get(key) is None

    def test_clear_resets_everything(self) -> None:
        cache = RequestCache()
        cache.put("k1", {"a": 1})
        cache.put("k2", {"b": 2})
        cache.get("k1")  # hit
        cache.clear()
        s = cache.stats()
        assert s["size"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0

    def test_ttl_expiration(self) -> None:
        cache = RequestCache(ttl_seconds=0)
        cache.put("k", {"x": 1})
        time.sleep(0.01)
        assert cache.get("k") is None


# ======================================================================
# BaseModule integration
# ======================================================================


class TestBaseModuleCacheIntegration:
    """Test that _call_llm / _acall_llm use the cache correctly."""

    def setup_method(self) -> None:
        """Reset the class-level cache before each test."""
        from aiforge.core.types import BaseModule

        BaseModule._cache = None

    def teardown_method(self) -> None:
        from aiforge.core.types import BaseModule

        BaseModule._cache = None

    def _make_module(self, cache_enabled: bool = True) -> Any:
        """Create a minimal BaseModule subclass instance with a mock provider."""
        from aiforge.core.types import BaseModule, CompletionResult, ModuleResponse

        class _Stub(BaseModule):
            def run(self, text: str, **kw: Any) -> ModuleResponse:
                return ModuleResponse()

            async def arun(self, text: str, **kw: Any) -> ModuleResponse:
                return ModuleResponse()

        mock_provider = MagicMock()
        mock_provider.complete.return_value = CompletionResult(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
        )

        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            instance = _Stub(CACHE_ENABLED=cache_enabled)
        instance._provider = mock_provider
        return instance

    def test_cache_hit_skips_provider(self) -> None:
        mod = self._make_module()
        msgs = [{"role": "user", "content": "hello"}]

        # First call — cache miss → calls provider
        r1 = mod._call_llm(msgs, temperature=0.0)
        assert mod.provider.complete.call_count == 1

        # Second call — cache hit → skips provider
        r2 = mod._call_llm(msgs, temperature=0.0)
        assert mod.provider.complete.call_count == 1  # still 1
        assert r2.text == r1.text

    def test_use_cache_false_skips_cache(self) -> None:
        mod = self._make_module()
        msgs = [{"role": "user", "content": "hello"}]

        mod._call_llm(msgs, temperature=0.0)
        mod._call_llm(msgs, temperature=0.0, use_cache=False)
        assert mod.provider.complete.call_count == 2

    def test_cache_disabled_skips_cache(self) -> None:
        mod = self._make_module(cache_enabled=False)
        msgs = [{"role": "user", "content": "hello"}]

        mod._call_llm(msgs, temperature=0.0)
        mod._call_llm(msgs, temperature=0.0)
        assert mod.provider.complete.call_count == 2

    def test_set_cache_class_method(self) -> None:
        from aiforge.core.types import BaseModule

        custom_cache = RequestCache(ttl_seconds=999)
        BaseModule.set_cache(custom_cache)

        mod = self._make_module()
        assert mod._get_cache() is custom_cache

    def test_cache_stats_track_hits(self) -> None:
        mod = self._make_module()
        msgs = [{"role": "user", "content": "stats test"}]

        mod._call_llm(msgs, temperature=0.0)
        mod._call_llm(msgs, temperature=0.0)

        cache = mod._get_cache()
        assert cache is not None
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    @pytest.mark.asyncio
    async def test_acall_llm_cache_hit(self) -> None:
        from aiforge.core.types import CompletionResult

        mod = self._make_module()
        mock_result = CompletionResult(
            text="async-hello",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
        )
        mod.provider.complete_async = MagicMock(return_value=mock_result)

        # Make complete_async actually awaitable

        async def _async_complete(msgs: Any, **kw: Any) -> CompletionResult:
            return mock_result

        mod.provider.complete_async = _async_complete

        msgs = [{"role": "user", "content": "async test"}]
        r1 = await mod._acall_llm(msgs, temperature=0.0)
        r2 = await mod._acall_llm(msgs, temperature=0.0)

        # Second call should have been served from cache
        assert r2.text == r1.text
        cache = mod._get_cache()
        assert cache is not None
        assert cache.hits == 1

    def test_disabled_cache_ignores_existing_singleton(self) -> None:
        """If CACHE_ENABLED=False, _get_cache() returns None even when a
        previous module already initialised the singleton cache (BUG-2 fix)."""
        from aiforge.core.types import BaseModule

        # Module A enables cache → auto-initialises singleton
        mod_a = self._make_module(cache_enabled=True)
        cache_a = mod_a._get_cache()
        assert cache_a is not None
        assert BaseModule._cache is not None  # singleton is set

        # Module B disables cache → must get None
        mod_b = self._make_module(cache_enabled=False)
        assert mod_b._get_cache() is None

    def test_redis_backend_selected_from_settings(self) -> None:
        """When CACHE_BACKEND='redis', _get_cache() should create a
        RedisBackend (BUG-1 fix).  We mock Redis to avoid a real connection."""
        from aiforge.core.types import BaseModule, ModuleResponse
        from aiforge.infra.cache import RedisBackend

        class _Stub(BaseModule):
            def run(self, text: str, **kw: Any) -> ModuleResponse:
                return ModuleResponse()

            async def arun(self, text: str, **kw: Any) -> ModuleResponse:
                return ModuleResponse()

        with (
            patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}),
            patch("aiforge.infra.cache.RedisBackend.__init__", return_value=None),
        ):
            instance = _Stub(CACHE_BACKEND="redis", CACHE_REDIS_URL="redis://fake:6379/0")
            cache = instance._get_cache()
            assert cache is not None
            assert isinstance(cache._backend, RedisBackend)


class TestCacheBackendValidator:
    """CACHE_BACKEND field validator rejects invalid values (BUG-3 fix)."""

    def test_valid_memory(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            s = AIForgeSettings(CACHE_BACKEND="memory")
            assert s.CACHE_BACKEND == "memory"

    def test_valid_redis(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            s = AIForgeSettings(CACHE_BACKEND="redis")
            assert s.CACHE_BACKEND == "redis"

    def test_invalid_backend_rejected(self) -> None:
        import pydantic

        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            with pytest.raises(pydantic.ValidationError, match="AIFORGE_CACHE_BACKEND"):
                AIForgeSettings(CACHE_BACKEND="memcached")
