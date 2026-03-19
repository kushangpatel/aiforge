"""
ChatAgent module — multi-turn conversational agent with memory and tool use.

Powered by LangGraph. Users can register tools, maintain multi-turn
conversations with configurable memory modes, and get streaming responses.

Usage::

    from aiforge import ChatAgent

    agent = ChatAgent(
        system_prompt="You are a helpful coding assistant.",
        tools=[calculator],
        memory="buffer",
        memory_k=20,
    )

    response = agent.chat("What is 15% off $249.99?")
    print(response.message)
    print(response.tool_calls)

    # Multi-turn (memory maintained)
    response = agent.chat("And what about 20%?")

    # Reset conversation
    agent.reset()
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator, Iterator

from aiforge.core.pricing import calculate_cost
from aiforge.core.types import BaseModule
from aiforge.modules.chat.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    SUMMARIZE_MEMORY_PROMPT,
    TOOL_USE_SYSTEM,
)
from aiforge.modules.chat.schemas import ChatResponse, ToolCallRecord


class ChatAgent(BaseModule):
    """Multi-turn conversational agent with memory and optional tool use.

    Args:
        system_prompt:       Custom system prompt. Defaults to a generic helper.
        tools:               LangChain-compatible tools (each needs ``.name``,
                             ``.description``, ``.invoke()``).
        memory:              Memory mode: ``"buffer"`` | ``"summary"`` | ``"none"``.
        memory_k:            Number of messages to keep in the buffer.
        max_tool_iterations: Safety cap on tool-use loops.
        **kwargs:            Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        memory: str = "buffer",
        memory_k: int = 20,
        max_tool_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if memory not in ("buffer", "summary", "none"):
            raise ValueError(
                f"memory must be 'buffer', 'summary', or 'none', got {memory!r}"
            )
        if memory_k < 1:
            raise ValueError("memory_k must be >= 1")
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be >= 1")

        self.tools: list[Any] = tools or []
        self.memory_mode: str = memory
        self.memory_k: int = memory_k
        self.max_tool_iterations: int = max_tool_iterations
        self.conversation_id: str = uuid.uuid4().hex[:16]

        # Build the effective system prompt
        base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if tools:
            tool_descs = "\n".join(
                f"- {t.name}: {t.description}" for t in self.tools
            )
            tool_instructions = TOOL_USE_SYSTEM.format(
                tool_descriptions=tool_descs,
            )
            # Combine the user's custom prompt with tool-use instructions
            self.system_prompt: str = f"{base_prompt}\n\n{tool_instructions}"
        else:
            self.system_prompt = base_prompt

        # Conversation history — list of ``{"role": ..., "content": ...}`` dicts
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str) -> ChatResponse:
        """Send a message and get a response.

        Maintains conversation state across calls. Tools are invoked
        automatically when the LLM requests them.

        Args:
            message: The user's message.

        Returns:
            :class:`ChatResponse` with the assistant's reply.

        Raises:
            ValueError: If *message* is empty or whitespace-only.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty or whitespace-only")

        from aiforge.modules.chat.agent import build_chat_graph

        t0 = time.monotonic()

        self._messages.append({"role": "user", "content": message})

        graph = build_chat_graph(
            llm_fn=self._make_llm_fn(),
            tools=self.tools,
        )
        final = graph.invoke(self._initial_state(message))

        # The last message should be the assistant's response
        response_text = self._extract_response(final)

        # Append only the final assistant reply to our clean history.
        # Do NOT overwrite self._messages with graph's internal messages
        # (which include intermediate tool-call markers and synthetic
        # "Tool results" entries that would pollute multi-turn context).
        self._messages.append({"role": "assistant", "content": response_text})

        # Memory management
        self._manage_memory()

        return self._build_response(final, response_text, t0)

    async def achat(self, message: str) -> ChatResponse:
        """Async version of :meth:`chat`."""
        if not message or not message.strip():
            raise ValueError("message must not be empty or whitespace-only")

        from aiforge.modules.chat.agent import build_chat_graph

        t0 = time.monotonic()

        self._messages.append({"role": "user", "content": message})

        graph = build_chat_graph(
            llm_fn=self._make_async_llm_fn(),
            tools=self.tools,
        )
        final = await graph.ainvoke(self._initial_state(message))

        response_text = self._extract_response(final)
        self._messages.append({"role": "assistant", "content": response_text})
        self._manage_memory()

        return self._build_response(final, response_text, t0)

    async def chat_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Stream the assistant's response token-by-token.

        Yields text chunks as they become available.
        After the stream completes, conversation history is updated.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty or whitespace-only")

        from aiforge.modules.chat.agent import build_chat_graph

        self._messages.append({"role": "user", "content": message})

        graph = build_chat_graph(
            llm_fn=self._make_async_llm_fn(),
            tools=self.tools,
        )

        full_response = ""
        async for event in graph.astream(self._initial_state(message)):
            # LangGraph astream yields dicts keyed by node name
            for _node_name, node_state in event.items():
                msgs = node_state.get("messages", [])
                if msgs:
                    last = msgs[-1] if isinstance(msgs, list) else msgs
                    if isinstance(last, dict) and last.get("role") == "assistant":
                        content = last["content"]
                        new_text = content[len(full_response):]
                        if new_text:
                            full_response = content
                            yield new_text

        if full_response:
            self._messages.append(
                {"role": "assistant", "content": full_response}
            )
            self._manage_memory()

    async def achat_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Alias for :meth:`chat_stream`."""
        async for chunk in self.chat_stream(message):
            yield chunk

    def reset(self) -> None:
        """Clear conversation history. Keeps the system prompt."""
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.conversation_id = uuid.uuid4().hex[:16]

    # ------------------------------------------------------------------
    # BaseModule interface
    # ------------------------------------------------------------------

    def run(self, text: str, **kwargs: Any) -> ChatResponse:
        """BaseModule interface — delegates to :meth:`chat`."""
        return self.chat(text)

    async def arun(self, text: str, **kwargs: Any) -> ChatResponse:
        """Async BaseModule interface — delegates to :meth:`achat`."""
        return await self.achat(text)

    def run_stream(self, text: str, **kwargs: Any) -> Iterator[str]:
        """Sync streaming — wraps async stream with event loop."""
        import asyncio

        async def _collect() -> list[str]:
            chunks: list[str] = []
            async for chunk in self.chat_stream(text):
                chunks.append(chunk)
            return chunks

        return iter(asyncio.get_event_loop().run_until_complete(_collect()))

    async def arun_stream(self, text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Async streaming — delegates to :meth:`chat_stream`."""
        async for chunk in self.chat_stream(text):
            yield chunk

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _manage_memory(self) -> None:
        """Apply memory strategy after each turn."""
        if self.memory_mode == "none":
            # Keep only system + last exchange
            if len(self._messages) >= 3:
                self._messages = [
                    self._messages[0],       # system prompt
                    self._messages[-2],      # last user message
                    self._messages[-1],      # last AI response
                ]
        elif self.memory_mode == "buffer":
            # Keep last K messages + system prompt
            if len(self._messages) > self.memory_k + 1:
                self._messages = (
                    [self._messages[0]] + self._messages[-self.memory_k:]
                )
        elif self.memory_mode == "summary":
            # When buffer exceeds 2×K, summarize older messages
            if len(self._messages) > self.memory_k * 2:
                old_messages = self._messages[1:-self.memory_k]
                recent_messages = self._messages[-self.memory_k:]

                convo_text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in old_messages
                )
                prompt = SUMMARIZE_MEMORY_PROMPT.format(conversation=convo_text)
                summary_result = self._call_llm(
                    [{"role": "user", "content": prompt}]
                )
                summary_msg = {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary_result.text}",
                }
                self._messages = (
                    [self._messages[0], summary_msg] + recent_messages
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_llm_fn(self):
        """Return a closure wrapping ``_call_llm`` for the graph."""

        def _fn(messages: list[dict[str, str]]) -> Any:
            return self._call_llm(messages)

        return _fn

    def _make_async_llm_fn(self):
        """Return an async closure wrapping ``_acall_llm`` for the graph.

        Note: Since build_chat_graph expects a sync callable, the async
        variant is still invoked synchronously in the graph nodes. True
        async streaming uses ``graph.astream()``.
        """

        def _fn(messages: list[dict[str, str]]) -> Any:
            return self._call_llm(messages)

        return _fn

    def _initial_state(self, message: str) -> dict[str, Any]:
        """Build the initial LangGraph state dictionary."""
        return {
            "messages": self._messages.copy(),
            "current_input": message,
            "tool_calls": [],
            "tool_results": [],
            "iteration": 0,
            "max_tool_iterations": self.max_tool_iterations,
            "trace": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "model": "",
        }

    @staticmethod
    def _extract_response(final: dict[str, Any]) -> str:
        """Pull the assistant's reply from the final graph state."""
        messages = final.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg["content"]
        return ""

    def _build_response(
        self,
        final: dict[str, Any],
        response_text: str,
        t0: float,
    ) -> ChatResponse:
        """Construct a :class:`ChatResponse` from graph output."""
        model = final.get("model", "") or ""
        cost = calculate_cost(
            model,
            final["total_input_tokens"],
            final["total_output_tokens"],
        )
        tool_records = [
            ToolCallRecord(
                tool=tr["tool"],
                input=tr.get("input", {}),
                output=tr.get("output"),
            )
            for tr in final.get("tool_results", [])
        ]
        return ChatResponse(
            message=response_text,
            conversation_id=self.conversation_id,
            tool_calls=tool_records,
            iterations=final.get("iteration", 0),
            memory_mode=self.memory_mode,
            input_tokens=final["total_input_tokens"],
            output_tokens=final["total_output_tokens"],
            cost=cost,
            model=model,
            trace_id=uuid.uuid4().hex[:16],
            latency_ms=(time.monotonic() - t0) * 1000,
        )
