"""Sync/async bridge for MCP tool calls.

The MCP SDK's ``session.call_tool()`` is async, but LangGraph's
``execute_tool_node`` calls ``.invoke()`` synchronously.  This module
provides a persistent event loop running in a background thread so that
MCP sessions and their transports survive across sync calls.

The same pattern is used by ``ChatAgent._make_llm_fn()`` where LangGraph
runs sync node functions in a thread pool.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class _LoopThread:
    """A background daemon thread running a persistent asyncio event loop.

    All MCP async operations (connect, call_tool, disconnect) are
    submitted to this loop so that sessions and transports remain
    bound to a single, long-lived event loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="aiforge-mcp-loop",
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Schedule *coro* on the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        """Stop the background loop and join the thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop


def make_sync_tool_caller(
    session: Any,
    tool_name: str,
    timeout: float,
    loop_thread: "_LoopThread",
) -> Callable[..., str]:
    """Return a sync callable that invokes an MCP tool via ``session.call_tool``.

    The returned function accepts ``**kwargs`` matching the tool's input
    schema and returns a stringified result.  The async call is submitted
    to the *same* event loop that owns the session.
    """

    def _call(**kwargs: Any) -> str:
        async def _invoke() -> str:
            # Strip None values — MCP servers may reject null for optional params
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=filtered),
                timeout=timeout,
            )
            # MCP CallToolResult has .content (list of content blocks)
            parts: list[str] = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts) if parts else str(result)

        return loop_thread.run(_invoke())

    return _call
