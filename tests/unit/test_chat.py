"""Unit tests for the ChatAgent module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.modules.chat.agent import (
    ChatAgentState,
    build_chat_graph,
    _parse_tool_calls,
)
from aiforge.modules.chat.module import ChatAgent
from aiforge.modules.chat.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    SUMMARIZE_MEMORY_PROMPT,
    TOOL_USE_SYSTEM,
)
from aiforge.modules.chat.schemas import ChatResponse, ToolCallRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    memory: str = "buffer",
    memory_k: int = 20,
    max_tool_iterations: int = 5,
) -> ChatAgent:
    """Build a ChatAgent bypassing BaseModule provider init."""
    agent = object.__new__(ChatAgent)
    agent.tools = tools or []
    agent.memory_mode = memory
    agent.memory_k = memory_k
    agent.max_tool_iterations = max_tool_iterations
    agent.conversation_id = "test-conv-id"

    base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    if tools:
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        tool_instructions = TOOL_USE_SYSTEM.format(tool_descriptions=tool_descs)
        agent.system_prompt = f"{base_prompt}\n\n{tool_instructions}"
    else:
        agent.system_prompt = base_prompt

    agent._messages = [{"role": "system", "content": agent.system_prompt}]

    import structlog

    agent.logger = structlog.get_logger()
    agent._provider = None
    agent._embedding_provider = MagicMock()
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.MODEL = "gpt-4o-mini"
    agent.settings = settings
    from aiforge.infra.cost_tracker import CostTracker

    agent.cost_tracker = CostTracker.get_session()
    return agent


def _fake_llm(
    text: str = "test response", inp: int = 50, out: int = 30,
) -> MagicMock:
    """Create a fake CompletionResult."""
    result = MagicMock()
    result.text = text
    result.input_tokens = inp
    result.output_tokens = out
    result.model = "gpt-4o-mini"
    result.finish_reason = "stop"
    return result


def _initial_state(
    message: str = "Hello",
    max_tool_iterations: int = 5,
) -> ChatAgentState:
    """Build a minimal initial state for graph tests."""
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": message},
        ],
        "current_input": message,
        "tool_calls": [],
        "tool_results": [],
        "iteration": 0,
        "max_tool_iterations": max_tool_iterations,
        "trace": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "model": "",
    }


def _mock_tool(name: str = "calculator", description: str = "Math tool") -> MagicMock:
    """Create a mock LangChain-compatible tool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.invoke = MagicMock(return_value="42")
    return tool


# ===========================================================================
# Schema tests
# ===========================================================================


class TestSchemas:
    def test_chat_response_fields(self) -> None:
        resp = ChatResponse(
            message="Hello!",
            conversation_id="abc",
            tool_calls=[],
            iterations=0,
            memory_mode="buffer",
        )
        assert resp.message == "Hello!"
        assert resp.conversation_id == "abc"
        assert resp.tool_calls == []
        assert resp.iterations == 0

    def test_chat_response_to_text(self) -> None:
        resp = ChatResponse(message="Hi there", conversation_id="x")
        assert resp.to_text() == "Hi there"

    def test_tool_call_record(self) -> None:
        tc = ToolCallRecord(
            tool="calculator",
            input={"expression": "2+2"},
            output="4",
        )
        assert tc.tool == "calculator"
        assert tc.output == "4"
        assert "calculator" in tc.to_text()

    def test_chat_response_defaults(self) -> None:
        resp = ChatResponse()
        assert resp.message == ""
        assert resp.conversation_id == ""
        assert resp.tool_calls == []
        assert resp.cost == 0.0


# ===========================================================================
# Prompt tests
# ===========================================================================


class TestPrompts:
    def test_default_system_prompt_exists(self) -> None:
        assert "helpful" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_tool_use_system_format(self) -> None:
        result = TOOL_USE_SYSTEM.format(tool_descriptions="- calc: math")
        assert "calc" in result
        assert "tool" in result.lower()

    def test_summarize_memory_prompt_format(self) -> None:
        result = SUMMARIZE_MEMORY_PROMPT.format(conversation="user: hi\nassistant: hello")
        assert "hi" in result
        assert "hello" in result


# ===========================================================================
# Tool call parsing tests
# ===========================================================================


class TestParseToolCalls:
    def test_valid_json_tool_call(self) -> None:
        text = '{"tool": "calculator", "input": {"expression": "2+2"}}'
        result = _parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["tool"] == "calculator"

    def test_no_tool_call_in_plain_text(self) -> None:
        text = "I'll just answer directly without any tools."
        assert _parse_tool_calls(text) == []

    def test_tool_call_in_multiline(self) -> None:
        text = 'Let me calculate that.\n{"tool": "calculator", "input": {"expression": "5*3"}}'
        result = _parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["tool"] == "calculator"

    def test_invalid_json_returns_empty(self) -> None:
        text = '{"tool": "calc", "input": {malformed}'
        assert _parse_tool_calls(text) == []

    def test_json_without_tool_key_ignored(self) -> None:
        text = '{"name": "not a tool call"}'
        assert _parse_tool_calls(text) == []

    def test_multiple_tool_calls_on_separate_lines(self) -> None:
        text = (
            '{"tool": "calculator", "input": {"expression": "1+1"}}\n'
            '{"tool": "current_time", "input": {}}'
        )
        result = _parse_tool_calls(text)
        assert len(result) == 2
        assert result[0]["tool"] == "calculator"
        assert result[1]["tool"] == "current_time"


# ===========================================================================
# Agent graph tests — direct answer (no tools)
# ===========================================================================


class TestAgentGraphDirect:
    """Graph returns a direct answer when no tool calls are detected."""

    def test_direct_answer(self) -> None:
        graph = build_chat_graph(
            llm_fn=lambda m: _fake_llm("Hello! How can I help?"),
        )
        result = graph.invoke(_initial_state())
        # Last message should be the assistant's response
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "Hello! How can I help?"

    def test_direct_answer_trace(self) -> None:
        graph = build_chat_graph(
            llm_fn=lambda m: _fake_llm("Hi"),
        )
        result = graph.invoke(_initial_state())
        assert any(t["step"] == "route" for t in result["trace"])
        assert result["trace"][0]["decision"] == "direct_answer"

    def test_direct_answer_tokens_accumulated(self) -> None:
        graph = build_chat_graph(
            llm_fn=lambda m: _fake_llm("answer", inp=100, out=20),
        )
        result = graph.invoke(_initial_state())
        assert result["total_input_tokens"] == 100
        assert result["total_output_tokens"] == 20


# ===========================================================================
# Agent graph tests — tool use
# ===========================================================================


class TestAgentGraphToolUse:
    """Graph invokes tools when the LLM returns a tool call JSON."""

    def test_single_tool_call(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "calculator", "input": {"expression": "2+2"}}'),
            _fake_llm("The answer is 4."),
        ])
        tool = _mock_tool("calculator")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state())
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "calculator"
        assert result["messages"][-1]["content"] == "The answer is 4."

    def test_tool_call_trace(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "calculator", "input": {}}'),
            _fake_llm("Done."),
        ])
        tool = _mock_tool("calculator")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state())
        steps = [t["step"] for t in result["trace"]]
        assert "route" in steps
        assert "execute_tool" in steps
        assert "process_result" in steps

    def test_unknown_tool_returns_error(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "nonexistent_tool", "input": {}}'),
            _fake_llm("Sorry, I couldn't use that tool."),
        ])
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[],
        )
        result = graph.invoke(_initial_state())
        assert "Error" in result["tool_results"][0]["output"]
        assert "Unknown tool" in result["tool_results"][0]["output"]

    def test_tool_exception_captured(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "broken", "input": {}}'),
            _fake_llm("The tool failed."),
        ])
        broken_tool = _mock_tool("broken")
        broken_tool.invoke = MagicMock(side_effect=RuntimeError("Tool broke"))
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[broken_tool],
        )
        result = graph.invoke(_initial_state())
        assert "Error" in result["tool_results"][0]["output"]
        assert "Tool broke" in result["tool_results"][0]["output"]

    def test_iteration_increments(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "calc", "input": {}}'),
            _fake_llm("Done."),
        ])
        tool = _mock_tool("calc")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state())
        assert result["iteration"] == 1


# ===========================================================================
# Agent graph — multi-step tool use
# ===========================================================================


class TestAgentGraphMultiTool:
    """Graph can chain multiple tool calls (tool → process → tool → respond)."""

    def test_two_tool_iterations(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "calc", "input": {"expression": "1+1"}}'),
            _fake_llm('{"tool": "calc", "input": {"expression": "2+3"}}'),
            _fake_llm("Results: 2 and 5."),
        ])
        tool = _mock_tool("calc")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state(max_tool_iterations=5))
        assert result["iteration"] == 2
        assert len(result["tool_results"]) == 2

    def test_max_tool_iterations_respected(self) -> None:
        """Even if LLM keeps requesting tools, cap at max_tool_iterations."""
        calls = iter([
            _fake_llm('{"tool": "calc", "input": {}}'),  # route
            _fake_llm('{"tool": "calc", "input": {}}'),  # process 1
            _fake_llm('{"tool": "calc", "input": {}}'),  # process 2 — at cap
            _fake_llm("Final answer."),  # shouldn't reach this
        ])
        tool = _mock_tool("calc")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state(max_tool_iterations=2))
        assert result["iteration"] <= 2

    def test_tokens_accumulate_across_iterations(self) -> None:
        calls = iter([
            _fake_llm('{"tool": "calc", "input": {}}', inp=100, out=10),
            _fake_llm("Done.", inp=80, out=20),
        ])
        tool = _mock_tool("calc")
        graph = build_chat_graph(
            llm_fn=lambda m: next(calls),
            tools=[tool],
        )
        result = graph.invoke(_initial_state())
        assert result["total_input_tokens"] == 180
        assert result["total_output_tokens"] == 30


# ===========================================================================
# ChatAgent (module.py) tests — basic chat
# ===========================================================================


class TestChatAgentBasic:
    """Basic ChatAgent.chat() functionality."""

    def test_basic_chat(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("Hello!"))
        result = agent.chat("Hi")
        assert result.message == "Hello!"
        assert result.conversation_id == "test-conv-id"
        assert isinstance(result, ChatResponse)

    def test_chat_adds_to_history(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("I'm fine"))
        agent.chat("How are you?")
        # History should have: system + user + assistant
        roles = [m["role"] for m in agent._messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_chat_returns_cost(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("ok", inp=100, out=50))
        result = agent.chat("Hi")
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms > 0

    def test_chat_empty_message_raises(self) -> None:
        agent = _agent()
        with pytest.raises(ValueError, match="message must not be empty"):
            agent.chat("")

    def test_chat_whitespace_message_raises(self) -> None:
        agent = _agent()
        with pytest.raises(ValueError, match="message must not be empty"):
            agent.chat("   ")

    def test_chat_trace_id_generated(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        result = agent.chat("Hi")
        assert result.trace_id
        assert len(result.trace_id) == 16


# ===========================================================================
# ChatAgent — multi-turn memory
# ===========================================================================


class TestChatAgentMultiTurn:
    """Multi-turn conversation with memory retention."""

    def test_multi_turn_retains_history(self) -> None:
        agent = _agent(memory="buffer", memory_k=20)
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        agent.chat("First message")
        agent.chat("Second message")
        # Should have system + user/assistant pairs
        assert len(agent._messages) >= 5

    def test_conversation_id_stable(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        r1 = agent.chat("Hi")
        r2 = agent.chat("Again")
        assert r1.conversation_id == r2.conversation_id


# ===========================================================================
# ChatAgent — tool use
# ===========================================================================


class TestChatAgentTools:
    """ChatAgent with tool registration."""

    def test_chat_with_tool(self) -> None:
        tool = _mock_tool("calculator", "Math tool")
        agent = _agent(tools=[tool])

        calls = iter([
            _fake_llm('{"tool": "calculator", "input": {"expression": "2+2"}}'),
            _fake_llm("The answer is 42."),
        ])
        agent._call_llm = MagicMock(side_effect=lambda m, **kw: next(calls))

        result = agent.chat("What is 2+2?")
        assert result.message == "The answer is 42."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "calculator"

    def test_system_prompt_includes_tool_descriptions(self) -> None:
        tool = _mock_tool("calculator", "Math operations")
        agent = _agent(tools=[tool])
        assert "calculator" in agent.system_prompt
        assert "Math operations" in agent.system_prompt

    def test_no_tools_uses_default_prompt(self) -> None:
        agent = _agent()
        assert agent.system_prompt == DEFAULT_SYSTEM_PROMPT


# ===========================================================================
# ChatAgent — reset
# ===========================================================================


class TestChatAgentReset:
    """Conversation reset."""

    def test_reset_clears_history(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        agent.chat("Hello")
        agent.reset()
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "system"

    def test_reset_new_conversation_id(self) -> None:
        agent = _agent()
        old_id = agent.conversation_id
        agent.reset()
        assert agent.conversation_id != old_id

    def test_reset_keeps_system_prompt(self) -> None:
        agent = _agent(system_prompt="You are a pirate.")
        agent._call_llm = MagicMock(return_value=_fake_llm("Arrr!"))
        agent.chat("Hello")
        agent.reset()
        assert agent._messages[0]["content"] == "You are a pirate."


# ===========================================================================
# ChatAgent — memory modes
# ===========================================================================


class TestMemoryBuffer:
    """Buffer memory mode — keep last K messages."""

    def test_buffer_limits_messages(self) -> None:
        agent = _agent(memory="buffer", memory_k=4)
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        for i in range(10):
            agent.chat(f"Message {i}")
        # system + last 4 messages
        assert len(agent._messages) <= 5

    def test_buffer_keeps_system_prompt(self) -> None:
        agent = _agent(memory="buffer", memory_k=2)
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        for i in range(5):
            agent.chat(f"Msg {i}")
        assert agent._messages[0]["role"] == "system"


class TestMemoryNone:
    """None memory mode — only system + last exchange."""

    def test_none_mode_minimal_history(self) -> None:
        agent = _agent(memory="none")
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        agent.chat("First")
        agent.chat("Second")
        # system + last user + last assistant
        assert len(agent._messages) == 3
        assert agent._messages[0]["role"] == "system"

    def test_none_mode_last_exchange_only(self) -> None:
        agent = _agent(memory="none")
        agent._call_llm = MagicMock(return_value=_fake_llm("response"))
        agent.chat("old message")
        agent.chat("new message")
        user_msgs = [m for m in agent._messages if m["role"] == "user"]
        assert len(user_msgs) == 1


class TestMemorySummary:
    """Summary memory mode — summarizes old messages when buffer exceeds 2×K."""

    def test_summary_mode_compresses(self) -> None:
        agent = _agent(memory="summary", memory_k=3)
        call_count = [0]

        def fake_llm(messages, **kw):
            call_count[0] += 1
            # The summary call will request a condensation
            return _fake_llm("Summary of conversation." if call_count[0] > 10 else "ok")

        agent._call_llm = MagicMock(side_effect=fake_llm)
        for i in range(10):
            agent.chat(f"Message {i}")
        # History should be much smaller than 20+ messages
        assert len(agent._messages) < 15

    def test_summary_preserves_system_prompt(self) -> None:
        agent = _agent(memory="summary", memory_k=3)
        agent._call_llm = MagicMock(return_value=_fake_llm("ok"))
        for i in range(10):
            agent.chat(f"Msg {i}")
        assert agent._messages[0]["role"] == "system"


# ===========================================================================
# ChatAgent — custom system prompt
# ===========================================================================


class TestCustomSystemPrompt:
    def test_custom_prompt(self) -> None:
        agent = _agent(system_prompt="You are a pirate.")
        assert agent.system_prompt == "You are a pirate."
        assert agent._messages[0]["content"] == "You are a pirate."

    def test_default_prompt_when_none(self) -> None:
        agent = _agent()
        assert agent.system_prompt == DEFAULT_SYSTEM_PROMPT


# ===========================================================================
# ChatAgent — constructor validation
# ===========================================================================


class TestChatAgentValidation:
    def test_invalid_memory_mode(self) -> None:
        with pytest.raises(ValueError, match="memory must be"):
            ChatAgent(memory="invalid_mode")

    def test_memory_k_zero(self) -> None:
        with pytest.raises(ValueError, match="memory_k must be"):
            ChatAgent(memory_k=0)

    def test_max_tool_iterations_zero(self) -> None:
        with pytest.raises(ValueError, match="max_tool_iterations must be"):
            ChatAgent(max_tool_iterations=0)


# ===========================================================================
# ChatAgent — async
# ===========================================================================


class TestChatAgentAsync:
    @pytest.mark.asyncio
    async def test_achat_basic(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("Async hello!"))
        result = await agent.achat("Hi")
        assert result.message == "Async hello!"
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_achat_empty_raises(self) -> None:
        agent = _agent()
        with pytest.raises(ValueError, match="message must not be empty"):
            await agent.achat("")


# ===========================================================================
# ChatAgent — run / arun (BaseModule interface)
# ===========================================================================


class TestBaseModuleInterface:
    def test_run_delegates_to_chat(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("via run"))
        result = agent.run("Hello")
        assert result.message == "via run"

    @pytest.mark.asyncio
    async def test_arun_delegates_to_achat(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("via arun"))
        result = await agent.arun("Hello")
        assert result.message == "via arun"


# ===========================================================================
# ChatAgent — batch
# ===========================================================================


class TestChatAgentBatch:
    def test_run_batch(self) -> None:
        agent = _agent()
        agent._call_llm = MagicMock(return_value=_fake_llm("batch ok"))
        agent._acall_llm = AsyncMock(return_value=_fake_llm("batch ok"))
        result = agent.run_batch(["q1", "q2"])
        assert len(result.results) + len(result.failed) == 2


# ===========================================================================
# Route tests
# ===========================================================================


class TestRoutes:
    """FastAPI route tests."""

    def test_chat_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from aiforge.server.routes.chat import router, _conversations
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        _conversations.clear()

        with patch("aiforge.modules.chat.module.ChatAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.conversation_id = "test-id"
            mock_response = ChatResponse(
                message="Hello!",
                conversation_id="test-id",
                tool_calls=[],
                iterations=0,
                memory_mode="buffer",
            )
            mock_instance.chat.return_value = mock_response
            MockAgent.return_value = mock_instance

            client = TestClient(app)
            resp = client.post("/chat", json={"message": "Hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["message"] == "Hello!"
            assert data["conversation_id"] == "test-id"

    def test_chat_validation_error(self) -> None:
        from fastapi.testclient import TestClient
        from aiforge.server.routes.chat import router, _conversations
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        _conversations.clear()

        client = TestClient(app)
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_reset_not_found(self) -> None:
        from fastapi.testclient import TestClient
        from aiforge.server.routes.chat import router, _conversations
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        _conversations.clear()

        client = TestClient(app)
        resp = client.post(
            "/chat/reset",
            json={"conversation_id": "nonexistent"},
        )
        assert resp.status_code == 404

    def test_reset_existing_conversation(self) -> None:
        from fastapi.testclient import TestClient
        from aiforge.server.routes.chat import router, _conversations
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        _conversations.clear()

        mock_agent = MagicMock()
        mock_agent.conversation_id = "existing-id"

        def _do_reset():
            mock_agent.conversation_id = "new-id-after-reset"

        mock_agent.reset = MagicMock(side_effect=_do_reset)
        _conversations["existing-id"] = mock_agent

        client = TestClient(app)
        resp = client.post(
            "/chat/reset",
            json={"conversation_id": "existing-id"},
        )
        assert resp.status_code == 200
        mock_agent.reset.assert_called_once()
        # Old key must be removed, new key must be registered
        assert "existing-id" not in _conversations
        assert "new-id-after-reset" in _conversations

    def test_conversations_list(self) -> None:
        from fastapi.testclient import TestClient
        from aiforge.server.routes.chat import router, _conversations
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        _conversations.clear()
        _conversations["id1"] = MagicMock()
        _conversations["id2"] = MagicMock()

        client = TestClient(app)
        resp = client.get("/chat/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2


# ===========================================================================
# Widget — safe calculator
# ===========================================================================


class TestSafeCalculator:
    def test_basic_arithmetic(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("2 + 3") == "5"
        assert _safe_calculator("10 * 5") == "50"
        assert _safe_calculator("2 ** 3") == "8"

    def test_negative_numbers(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("-5 + 3") == "-2"

    def test_float_arithmetic(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        result = float(_safe_calculator("249.99 * 0.85"))
        assert abs(result - 212.4915) < 0.01

    def test_division(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("10 / 3") == str(10 / 3)

    def test_rejects_function_calls_to_unsafe_builtins(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises(ValueError, match="Unsafe function"):
            _safe_calculator("__import__('os')")

    def test_rejects_attribute_access(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises(ValueError, match="Unsafe"):
            _safe_calculator("().__class__.__bases__")

    def test_rejects_list_comprehension(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises((ValueError, SyntaxError)):
            _safe_calculator("[x for x in range(10)]")

    # --- Advanced math functions ---

    def test_sqrt(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("sqrt(16)") == "4.0"
        assert _safe_calculator("sqrt(2)") == str(__import__("math").sqrt(2))

    def test_trig_functions(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("sin(0)")) - 0.0) < 1e-10
        assert abs(float(_safe_calculator("cos(0)")) - 1.0) < 1e-10
        assert abs(float(_safe_calculator("tan(0)")) - 0.0) < 1e-10
        # sin(pi/2) ≈ 1.0
        assert abs(float(_safe_calculator("sin(pi / 2)")) - 1.0) < 1e-10

    def test_inverse_trig(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("asin(1)")) - m.asin(1)) < 1e-10
        assert abs(float(_safe_calculator("acos(0)")) - m.acos(0)) < 1e-10
        assert abs(float(_safe_calculator("atan(1)")) - m.atan(1)) < 1e-10

    def test_log_functions(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("log(e)")) - 1.0) < 1e-10
        assert abs(float(_safe_calculator("log2(8)")) - 3.0) < 1e-10
        assert abs(float(_safe_calculator("log10(1000)")) - 3.0) < 1e-10
        # log with base: log(100, 10) = 2
        assert abs(float(_safe_calculator("log(100, 10)")) - 2.0) < 1e-10

    def test_exp(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("exp(1)")) - m.e) < 1e-10

    def test_ceil_floor(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("ceil(3.2)") == "4"
        assert _safe_calculator("floor(3.9)") == "3"

    def test_factorial(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("factorial(5)") == "120"
        assert _safe_calculator("factorial(0)") == "1"

    def test_gcd(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("gcd(12, 8)") == "4"

    def test_constants(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("pi")) - m.pi) < 1e-10
        assert abs(float(_safe_calculator("e")) - m.e) < 1e-10
        assert abs(float(_safe_calculator("tau")) - m.tau) < 1e-10

    def test_combined_expression(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        # sqrt(2) * pi
        expected = m.sqrt(2) * m.pi
        assert abs(float(_safe_calculator("sqrt(2) * pi")) - expected) < 1e-10

    def test_radians_degrees(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator
        import math as m

        assert abs(float(_safe_calculator("radians(180)")) - m.pi) < 1e-10
        assert abs(float(_safe_calculator("degrees(pi)")) - 180.0) < 1e-10

    def test_abs_round_min_max(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("abs(-42)") == "42"
        assert _safe_calculator("round(3.7)") == "4"
        assert _safe_calculator("min(3, 7)") == "3"
        assert _safe_calculator("max(3, 7)") == "7"

    def test_hypot(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        assert _safe_calculator("hypot(3, 4)") == "5.0"

    def test_rejects_exec(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises((ValueError, SyntaxError)):
            _safe_calculator("exec('print(1)')")

    def test_rejects_unknown_name(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises(ValueError, match="Unknown variable"):
            _safe_calculator("x + 1")


class TestCurrentTime:
    def test_returns_iso_format(self) -> None:
        from aiforge.modules.chat.widget import _current_time

        result = _current_time()
        assert "T" in result  # ISO format has 'T' separator
        assert "+" in result or "Z" in result  # timezone info


class TestGetBuiltinTools:
    def test_returns_four_tools(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"calculator", "current_time", "research", "unit_converter"}

    def test_tools_are_invokable(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "3 + 4"})
        assert result == "7"

    def test_calculator_tool_handles_advanced(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "sqrt(16)"})
        assert result == "4.0"

    def test_research_tool_graceful_fallback(self) -> None:
        """Research tool returns a message when search module can't connect."""
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        research = next(t for t in tools if t.name == "research")
        # Will fail gracefully since no vector store is configured
        result = research.invoke({"query": "test query"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unit_converter_tool(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        conv = next(t for t in tools if t.name == "unit_converter")
        result = conv.invoke({"expression": "100 celsius to fahrenheit"})
        assert "212" in result


class TestResearchTool:
    def test_returns_string(self) -> None:
        from aiforge.modules.chat.widget import _research

        result = _research("test query")
        assert isinstance(result, str)

    def test_handles_import_error(self) -> None:
        from aiforge.modules.chat.widget import _research

        with patch("aiforge.modules.chat.widget._research.__module__"):
            result = _research("anything")
            assert isinstance(result, str)


class TestUnitConverter:
    def test_celsius_to_fahrenheit(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("100 celsius to fahrenheit")
        assert "212" in result

    def test_fahrenheit_to_celsius(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("32 fahrenheit to celsius")
        assert "0.0000" in result

    def test_miles_to_km(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("1 miles to km")
        assert "1.6093" in result

    def test_kg_to_lbs(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("1 kg to lbs")
        assert "2.2046" in result

    def test_degrees_to_radians(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter
        import math

        result = _unit_converter("180 degrees to radians")
        assert str(round(math.pi, 4)) in result

    def test_same_unit(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("5 kg to kg")
        assert "same unit" in result

    def test_bad_format_raises(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        with pytest.raises(ValueError, match="Format"):
            _unit_converter("convert this")

    def test_unknown_units_raises(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        with pytest.raises(ValueError, match="Unknown conversion"):
            _unit_converter("5 parsecs to lightyears")

    def test_negative_values(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("-40 celsius to fahrenheit")
        assert "-40" in result

    def test_liters_to_gallons(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("3.78541 liters to gallons")
        assert "1.0000" in result

    def test_celsius_to_kelvin(self) -> None:
        from aiforge.modules.chat.widget import _unit_converter

        result = _unit_converter("0 celsius to kelvin")
        assert "273.15" in result


# ===========================================================================
# Bug regression tests
# ===========================================================================


class TestCustomPromptWithTools:
    """Custom system prompt must be preserved when tools are also provided."""

    def test_custom_prompt_combined_with_tool_instructions(self) -> None:
        tool = _mock_tool("calculator", "Math tool")
        agent = _agent(system_prompt="You are a pirate.", tools=[tool])
        # Both the custom prompt and the tool descriptions must be present
        assert "You are a pirate." in agent.system_prompt
        assert "calculator" in agent.system_prompt
        assert "Math tool" in agent.system_prompt

    def test_default_prompt_combined_with_tool_instructions(self) -> None:
        tool = _mock_tool("calculator", "Math tool")
        agent = _agent(tools=[tool])
        # Default prompt + tool instructions
        assert DEFAULT_SYSTEM_PROMPT in agent.system_prompt
        assert "calculator" in agent.system_prompt


class TestHistoryDoesNotLeakToolIntermediates:
    """Graph's internal tool messages must not pollute conversation history."""

    def test_history_clean_after_tool_use(self) -> None:
        """After a tool-using turn, _messages should only contain
        system, user, assistant — no intermediate tool messages."""
        tool = _mock_tool("calculator", "Math tool")
        agent = _agent(tools=[tool])

        calls = iter([
            _fake_llm('{"tool": "calculator", "input": {"expression": "2+2"}}'),
            _fake_llm("The answer is 4."),
        ])
        agent._call_llm = MagicMock(side_effect=lambda m, **kw: next(calls))
        agent.chat("What is 2+2?")

        roles = [m["role"] for m in agent._messages]
        # Should be: system, user, assistant — no synthetic tool messages
        assert roles == ["system", "user", "assistant"]
        # No "[Tool calls:" markers in the messages
        for msg in agent._messages:
            assert not msg["content"].startswith("[Tool calls:")
            assert not msg["content"].startswith("Tool results:")

    def test_none_mode_preserves_real_user_message_after_tools(self) -> None:
        """In 'none' mode with tools, the user's actual message must be kept,
        not a synthetic 'Tool results:' message."""
        tool = _mock_tool("calc", "Math")
        agent = _agent(tools=[tool], memory="none")

        calls = iter([
            _fake_llm('{"tool": "calc", "input": {}}'),
            _fake_llm("Done."),
        ])
        agent._call_llm = MagicMock(side_effect=lambda m, **kw: next(calls))
        agent.chat("Please calculate something")

        assert len(agent._messages) == 3
        assert agent._messages[1]["role"] == "user"
        assert agent._messages[1]["content"] == "Please calculate something"
        assert agent._messages[2]["role"] == "assistant"
        assert agent._messages[2]["content"] == "Done."

    def test_buffer_mode_not_inflated_by_tool_messages(self) -> None:
        """Tool intermediate messages should not consume memory_k slots."""
        tool = _mock_tool("calc", "Math")
        agent = _agent(tools=[tool], memory="buffer", memory_k=4)

        calls = iter([
            _fake_llm('{"tool": "calc", "input": {}}'),
            _fake_llm("r1"),
            _fake_llm('{"tool": "calc", "input": {}}'),
            _fake_llm("r2"),
            _fake_llm('{"tool": "calc", "input": {}}'),
            _fake_llm("r3"),
        ])
        agent._call_llm = MagicMock(side_effect=lambda m, **kw: next(calls))

        agent.chat("q1")
        agent.chat("q2")
        agent.chat("q3")

        # With K=4, after 3 turns (6 user/assistant msgs + 1 system = 7 total)
        # buffer trims to system + last 4 = 5 messages
        assert len(agent._messages) <= 5
        # All messages should be real roles (no tool intermediates)
        for msg in agent._messages:
            assert msg["role"] in ("system", "user", "assistant")
