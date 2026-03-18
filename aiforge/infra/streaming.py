"""
Async streaming helpers.

These utilities wrap LangChain's streaming interface into a clean
``AsyncGenerator[str, None]`` that works with ``async for`` loops
and callback notifications.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any


async def stream_llm_response(
    chain: Any,  # noqa: ANN401  — LangChain Runnable
    input_data: Any,  # noqa: ANN401
    *,
    on_token: Callable[[str], None] | None = None,
) -> AsyncGenerator[str, None]:
    """Yield individual string tokens from a LangChain streaming Runnable.

    Args:
        chain:      Any LangChain :class:`Runnable` that supports
                    ``astream()``.
        input_data: Input passed directly to ``chain.astream()``.
        on_token:   Optional synchronous callback invoked for each token
                    (useful for WebSocket / SSE push scenarios).

    Yields:
        Decoded string chunks as they arrive from the provider.
    """
    async for chunk in chain.astream(input_data):
        # Chunks may be AIMessage objects or raw strings depending on
        # which runnable produced them.
        if hasattr(chunk, "content"):
            token: str = chunk.content
        else:
            token = str(chunk)

        if on_token is not None:
            on_token(token)

        yield token


async def collect_stream(
    gen: AsyncGenerator[str, None],
) -> str:
    """Consume an async generator and return the concatenated string."""
    parts: list[str] = []
    async for token in gen:
        parts.append(token)
    return "".join(parts)


async def stream_with_accumulator(
    chain: Any,  # noqa: ANN401
    input_data: Any,  # noqa: ANN401
    *,
    on_token: Callable[[str], None] | None = None,
) -> tuple[AsyncGenerator[str, None], list[str]]:
    """Return a generator and a shared list that accumulates all tokens.

    This lets callers stream to the user while also capturing the full
    response for cost-tracking after streaming completes.

    Note: The accumulator list is populated lazily as the generator is
    consumed — it will be empty until iteration starts.
    """
    buffer: list[str] = []

    async def _gen() -> AsyncGenerator[str, None]:
        async for chunk in chain.astream(input_data):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            buffer.append(token)
            if on_token is not None:
                on_token(token)
            yield token

    return _gen(), buffer
