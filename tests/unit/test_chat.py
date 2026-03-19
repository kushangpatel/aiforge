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

    if tools:
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        agent.system_prompt = TOOL_USE_SYSTEM.format(tool_descriptions=tool_descs)
    else:
        agent.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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
        _conversations["existing-id"] = mock_agent

        client = TestClient(app)
        resp = client.post(
            "/chat/reset",
            json={"conversation_id": "existing-id"},
        )
        assert resp.status_code == 200
        mock_agent.reset.assert_called_once()

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

    def test_rejects_function_calls(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises(ValueError, match="Unsafe"):
            _safe_calculator("__import__('os').system('ls')")

    def test_rejects_attribute_access(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises(ValueError, match="Unsafe"):
            _safe_calculator("().__class__.__bases__")

    def test_rejects_list_comprehension(self) -> None:
        from aiforge.modules.chat.widget import _safe_calculator

        with pytest.raises((ValueError, SyntaxError)):
            _safe_calculator("[x for x in range(10)]")


class TestCurrentTime:
    def test_returns_iso_format(self) -> None:
        from aiforge.modules.chat.widget import _current_time

        result = _current_time()
        assert "T" in result  # ISO format has 'T' separator
        assert "+" in result or "Z" in result  # timezone info


class TestGetBuiltinTools:
    def test_returns_two_tools(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "calculator" in names
        assert "current_time" in names

    def test_tools_are_invokable(self) -> None:
        from aiforge.modules.chat.widget import _get_builtin_tools

        tools = _get_builtin_tools()
        calc = next(t for t in tools if t.name == "calculator")
        result = calc.invoke({"expression": "3 + 4"})
        assert result == "7"
