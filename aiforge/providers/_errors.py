"""
Map SDK exceptions from any provider to the AIForge error hierarchy.

Provider SDKs signal the same few conditions in different ways: a class name
(``RateLimitError``, ``ThrottlingException``, ``ReadTimeout``), an HTTP status
code (``exc.status_code``, ``exc.code`` or ``exc.response.status_code``) or,
for botocore, an error code in ``exc.response["Error"]["Code"]``. This module
looks at all of them, across the exception's ``__cause__`` chain, so wrapped
errors (LangChain re-raising a google-genai ``ClientError``, say) still map to
the retryable type.
"""

from __future__ import annotations

from aiforge.core.types import (
    APIConnectionError,
    APITimeoutError,
    ProviderError,
    RateLimitError,
)

_RATE_LIMIT_NAMES = ("ratelimit", "throttl", "toomanyrequests", "resourceexhausted")
_TIMEOUT_NAMES = ("timeout", "deadlineexceeded")
_CONNECTION_NAMES = ("connect", "networkerror", "remotedisconnected", "endpointresolution")

#: botocore / AWS service error codes.
_RATE_LIMIT_CODES = {
    "throttlingexception",
    "toomanyrequestsexception",
    "servicequotaexceededexception",
    "throttling",
    "requestlimitexceeded",
}
_TIMEOUT_CODES = {"modeltimeoutexception", "requesttimeout", "requesttimeoutexception"}


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code of an SDK exception."""
    for value in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if isinstance(response, dict):  # botocore ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _aws_error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", "")).lower()
    return ""


def _classify(exc: BaseException) -> type[ProviderError] | None:
    names = [cls.__name__.lower() for cls in type(exc).__mro__]
    code = _aws_error_code(exc)
    status = _status_code(exc)
    if (
        code in _RATE_LIMIT_CODES
        or status == 429
        or any(k in n for n in names for k in _RATE_LIMIT_NAMES)
    ):
        return RateLimitError
    if (
        code in _TIMEOUT_CODES
        or status in (408, 504)
        or any(k in n for n in names for k in _TIMEOUT_NAMES)
    ):
        return APITimeoutError
    if any(k in n for n in names for k in _CONNECTION_NAMES):
        return APIConnectionError
    return None


def wrap_sdk_error(exc: Exception) -> ProviderError:
    """Return the AIForge error for *exc*, following its ``__cause__`` chain.

    Rate limits (HTTP 429, AWS throttling) become :class:`RateLimitError`,
    timeouts (HTTP 408/504, client read/connect timeouts) become
    :class:`APITimeoutError` and network failures become
    :class:`APIConnectionError`; these three are retried. Anything else is a
    plain :class:`ProviderError`.
    """
    if isinstance(exc, ProviderError):
        return exc
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        kind = _classify(current)
        if kind is not None:
            return kind(str(exc))
        current = current.__cause__
    return ProviderError(str(exc))
