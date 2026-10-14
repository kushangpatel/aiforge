"""
Tenacity-based retry decorators for LLM and embedding calls.

Usage:
    from aiforge.infra.retry import retry_llm_call, retry_embedding_call

    @retry_llm_call()
    def my_llm_call(...):
        ...
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

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

# tenacity 9.2 renamed ``initial`` to ``multiplier`` and warns on the old name.
_JITTER_BASE_ARG = (
    "multiplier"
    if "multiplier" in inspect.signature(wait_exponential_jitter.__init__).parameters
    else "initial"
)


def _backoff(min_wait: float, max_wait: float) -> wait_exponential_jitter:
    return wait_exponential_jitter(**{_JITTER_BASE_ARG: min_wait}, max=max_wait, jitter=2)


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
        wait=_backoff(min_wait, max_wait),
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
        wait=_backoff(min_wait, max_wait),
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
