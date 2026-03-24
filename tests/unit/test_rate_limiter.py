"""Unit tests for RateLimiter and retry decorators."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aiforge.core.types import APIConnectionError, RateLimitError
from aiforge.infra.rate_limiter import RateLimiter
from aiforge.infra.retry import retry_embedding_call, retry_llm_call


class TestRateLimiter:
    def test_singleton_returns_same_instance(self) -> None:
        a = RateLimiter.get_instance()
        b = RateLimiter.get_instance()
        assert a is b

    def test_reset_gives_new_instance(self) -> None:
        a = RateLimiter.get_instance()
        RateLimiter.reset_instance()
        b = RateLimiter.get_instance()
        assert a is not b

    def test_acquire_succeeds_within_capacity(self) -> None:
        limiter = RateLimiter(rpm=100, tpm=10_000)
        # Should not block for first call
        start = time.monotonic()
        limiter.acquire(tokens=10)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # well under 1 second

    def test_acquire_request_slot_depletes_rpm(self) -> None:
        limiter = RateLimiter(rpm=2, tpm=1_000_000)
        limiter.acquire(tokens=0)
        limiter.acquire(tokens=0)
        # 3rd call should block — but we don't wait; just check tokens depleted
        assert limiter._rpm_bucket._tokens == 0

    def test_update_limits(self) -> None:
        limiter = RateLimiter(rpm=100, tpm=1_000)
        limiter.update_limits(rpm=200, tpm=2_000)
        assert limiter._rpm_bucket._capacity == 200
        assert limiter._tpm_bucket._capacity == 2_000

    def test_concurrent_acquire_thread_safe(self) -> None:
        """Multiple threads can call acquire() without deadlock or assertion errors."""
        limiter = RateLimiter(rpm=50, tpm=100_000)
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                limiter.acquire(tokens=100)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"

    def test_consume_zero_tokens_is_noop(self) -> None:
        """Consuming 0 tokens should return immediately without blocking."""
        limiter = RateLimiter(rpm=100, tpm=100)
        start = time.monotonic()
        limiter._tpm_bucket.consume(0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_consume_over_capacity_clamps_and_succeeds(self) -> None:
        """Requesting more tokens than capacity should clamp, not deadlock."""
        limiter = RateLimiter(rpm=100, tpm=10)
        start = time.monotonic()
        # Request 1000 tokens when capacity is 10 — should clamp to 10
        limiter._tpm_bucket.consume(1000)
        elapsed = time.monotonic() - start
        # Must complete without deadlock (< 2s to allow for a window reset)
        assert elapsed < 2.0
        assert limiter._tpm_bucket._tokens == 0

    def test_record_tokens_consumes_tpm(self) -> None:
        """record_tokens should decrement the TPM bucket."""
        limiter = RateLimiter(rpm=100, tpm=1000)
        initial = limiter._tpm_bucket._tokens
        limiter.record_tokens(200)
        assert limiter._tpm_bucket._tokens == initial - 200

    def test_record_tokens_zero_is_noop(self) -> None:
        """record_tokens(0) should not change the bucket."""
        limiter = RateLimiter(rpm=100, tpm=1000)
        initial = limiter._tpm_bucket._tokens
        limiter.record_tokens(0)
        assert limiter._tpm_bucket._tokens == initial


class TestRetryDecorators:
    def test_retry_llm_call_succeeds_on_first_try(self) -> None:
        call_count = 0

        @retry_llm_call(max_attempts=3)
        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = _fn()
        assert result == "ok"
        assert call_count == 1

    def test_retry_llm_call_retries_on_rate_limit(self) -> None:
        call_count = 0

        @retry_llm_call(max_attempts=3)
        def _fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("rate limited")
            return "success"

        with patch("time.sleep"):  # skip actual sleep
            result = _fn()

        assert result == "success"
        assert call_count == 3

    def test_retry_llm_call_raises_after_max_attempts(self) -> None:
        @retry_llm_call(max_attempts=2)
        def _fn() -> str:
            raise APIConnectionError("cannot connect")

        with patch("time.sleep"), pytest.raises(APIConnectionError):
            _fn()

    def test_retry_embedding_call_succeeds_immediately(self) -> None:
        @retry_embedding_call()
        def _embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 4 for _ in texts]

        result = _embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_non_retryable_exception_propagates_immediately(self) -> None:
        call_count = 0

        @retry_llm_call(max_attempts=3)
        def _fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("not a retryable error")

        with pytest.raises(ValueError):
            _fn()

        assert call_count == 1  # did not retry
