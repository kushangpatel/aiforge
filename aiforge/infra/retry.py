"""
Tenacity-based retry decorators for LLM and embedding calls.

Usage:
    from aiforge.infra.retry import retry_llm_call, retry_embedding_call

    @retry_llm_call()
    def my_llm_call(...):
        ...
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from aiforge.core.types import APIConnectionError, APITimeoutError, RateLimitError

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_LLM_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


def retry_llm_call(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[F], F]:
    """Decorator factory — wraps a function with LLM-aware retry logic.

    Retries on :exc:`RateLimitError`, :exc:`APITimeoutError`, and
    :exc:`APIConnectionError` with exponential backoff.
    """
    return retry(  # type: ignore[return-value]
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait, jitter=2),
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )


def retry_embedding_call(
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 8.0,
) -> Callable[[F], F]:
    """Decorator factory — wraps an embedding call with retry logic."""
    return retry(  # type: ignore[return-value]
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait, jitter=2),
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
