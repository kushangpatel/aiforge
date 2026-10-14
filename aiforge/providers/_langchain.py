"""
Shared implementation for providers backed by a LangChain chat model.

Subclasses construct ``self._llm`` and ``self._model_name`` and supply an
error mapper; this base handles message conversion, native tool calling,
native structured output, prompt-cache accounting and streaming the same
way for every provider.

Per-call keyword arguments understood here (all optional):

* ``tools`` — LangChain tools, functions, Pydantic models or JSON-schema
  tool dicts to offer the model. Requested calls come back on
  :attr:`CompletionResult.tool_calls`.
* ``tool_choice`` — passed to ``bind_tools`` (``"auto"``, ``"none"``, or a
  tool name). Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced choices.
* ``response_format`` — a Pydantic model class or JSON-schema dict. The model
  is constrained to that schema natively; the validated object is on
  :attr:`CompletionResult.parsed` and its JSON on ``.text``.

Anything else is forwarded to the LangChain model (``max_tokens``, ``stop`` …).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from pydantic import BaseModel

from aiforge.core.provider import LLMProvider
from aiforge.core.types import (
    APIConnectionError,
    APITimeoutError,
    CompletionResult,
    ProviderError,
    RateLimitError,
    ToolCall,
)
from aiforge.infra.retry import retry_llm_call
from aiforge.providers._utils import drop_unsupported_sampling, to_lc_messages

logger = structlog.get_logger(__name__)

# Errors that must surface (and be retried) rather than trigger the
# structured-output fallback.
_TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError)


def wrap_error_by_class_name(exc: Exception) -> ProviderError:
    """Map an SDK exception to the AIForge error hierarchy by its class name."""
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name:
        return RateLimitError(str(exc))
    if "Timeout" in cls_name:
        return APITimeoutError(str(exc))
    if "Connection" in cls_name:
        return APIConnectionError(str(exc))
    return ProviderError(str(exc))


def _named_schema(schema: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
    """JSON-schema dicts need a ``title`` for OpenAI's ``json_schema`` response format."""
    if isinstance(schema, dict) and "title" not in schema:
        return {"title": "output", **schema}
    return schema


def _finish_reason(response: Any) -> str:
    meta = getattr(response, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or meta.get("done_reason")
    if getattr(response, "tool_calls", None):
        return "tool_calls"
    return str(reason or "stop")


class LangChainLLMProvider(LLMProvider):
    """Base class for the built-in providers. See the module docstring."""

    supports_tools = True
    supports_structured_output = True

    _llm: Any
    _model_name: str

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception to the AIForge error hierarchy."""
        return wrap_error_by_class_name(exc)

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Provider-specific request options (e.g. prompt caching). Mutates nothing."""
        return drop_unsupported_sampling(self._model_name, kwargs)

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message rewriting (e.g. cache breakpoints)."""
        return messages

    def _usage(self, response: Any, messages: list[dict[str, Any]], text: str) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _runnable(self, kwargs: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        """Return ``(runnable, response_format, call_kwargs)`` for one call."""
        kwargs = dict(kwargs)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)
        if tools and response_format is not None:
            raise ValueError("Pass either tools= or response_format=, not both")
        call_kwargs = self._request_kwargs(kwargs)

        llm = self._llm
        if tools:
            bind_kwargs: dict[str, Any] = {}
            if tool_choice is not None:
                bind_kwargs["tool_choice"] = tool_choice
            llm = llm.bind_tools(tools, **bind_kwargs)
        if response_format is not None:
            llm = llm.with_structured_output(
                _named_schema(response_format), method="json_schema", include_raw=True
            )
        return llm, response_format, call_kwargs

    def _result(
        self,
        output: Any,
        messages: list[dict[str, Any]],
        response_format: Any,
        start: float,
    ) -> CompletionResult:
        parsed: dict[str, Any] | None = None
        if response_format is not None:
            # with_structured_output(include_raw=True) → {"raw", "parsed", "parsing_error"}
            response = output["raw"]
            value = output.get("parsed")
            if isinstance(value, BaseModel):
                parsed = value.model_dump(mode="json")
            elif isinstance(value, dict):
                parsed = value
        else:
            response = output

        text = json.dumps(parsed) if parsed is not None else response.text
        input_tokens, output_tokens = self._usage(response, messages, text)
        details = (getattr(response, "usage_metadata", None) or {}).get("input_token_details", {})
        cache_write = details.get("cache_creation") or 0
        cache_write += sum(
            details.get(k) or 0 for k in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
        )
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model_name,
            finish_reason=_finish_reason(response),
            tool_calls=[
                ToolCall(id=tc.get("id") or "", name=tc["name"], arguments=tc.get("args") or {})
                for tc in getattr(response, "tool_calls", None) or []
            ],
            parsed=parsed,
            cache_read_tokens=details.get("cache_read") or 0,
            cache_write_tokens=cache_write,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def _structured_fallback(self, exc: ProviderError, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop ``response_format`` after a non-transient failure of the native path.

        Some models or schemas aren't accepted by a provider's native
        structured-output mode; the caller's prompt still describes the
        schema, so a plain call followed by its own validation still works.
        """
        if isinstance(exc, _TRANSIENT) or "response_format" not in kwargs:
            raise exc
        logger.warning("structured_output_fallback", model=self._model_name, error=str(exc)[:200])
        return {k: v for k, v in kwargs.items() if k != "response_format"}

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def _invoke(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> CompletionResult:
        runnable, response_format, call_kwargs = self._runnable(kwargs)
        start = time.monotonic()
        try:
            output = runnable.invoke(
                to_lc_messages(self._prepare_messages(messages)), **call_kwargs
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return self._result(output, messages, response_format, start)

    async def _ainvoke(
        self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> CompletionResult:
        runnable, response_format, call_kwargs = self._runnable(kwargs)
        start = time.monotonic()
        try:
            output = await runnable.ainvoke(
                to_lc_messages(self._prepare_messages(messages)), **call_kwargs
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc
        return self._result(output, messages, response_format, start)

    @retry_llm_call()
    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        try:
            return self._invoke(messages, kwargs)
        except ProviderError as exc:
            return self._invoke(messages, self._structured_fallback(exc, kwargs))

    @retry_llm_call()
    async def complete_async(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> CompletionResult:
        try:
            return await self._ainvoke(messages, kwargs)
        except ProviderError as exc:
            return await self._ainvoke(messages, self._structured_fallback(exc, kwargs))

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._llm.astream(
            to_lc_messages(self._prepare_messages(messages)), **self._request_kwargs(kwargs)
        ):
            if chunk.text:
                yield chunk.text
