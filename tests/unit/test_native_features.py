"""Tests for native tool calling, structured output, rich messages and prompt caching."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from aiforge.core import messages as m
from aiforge.core.pricing import calculate_cost
from aiforge.core.types import CompletionResult, ProviderError, RateLimitError, ToolCall
from aiforge.infra.config import AIForgeSettings
from aiforge.providers._utils import to_lc_messages

langchain_core = pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic(prompt_cache: str = "system") -> Any:
    pytest.importorskip("langchain_anthropic")
    from aiforge.providers.anthropic import AnthropicProvider

    return AnthropicProvider(
        AIForgeSettings(
            PROVIDER="anthropic",
            MODEL="claude-opus-5-5",
            EMBEDDING_MODEL="voyage-3",
            ANTHROPIC_API_KEY="test",
            PROMPT_CACHE=prompt_cache,  # type: ignore[arg-type]
        )
    )


def _openai() -> Any:
    pytest.importorskip("langchain_openai")
    from aiforge.providers.openai import OpenAIProvider

    return OpenAIProvider(AIForgeSettings(MODEL="gpt-4o", OPENAI_API_KEY="test"))


def _anthropic_payload(provider: Any, msgs: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Build the HTTP request body ChatAnthropic would send — no network."""
    lc = to_lc_messages(provider._prepare_messages(msgs))
    return provider._llm._get_request_payload(lc, **provider._request_kwargs(kwargs))


class Weather(BaseModel):
    city: str
    celsius: float


# ---------------------------------------------------------------------------
# Message helpers and conversion
# ---------------------------------------------------------------------------


class TestMessageHelpers:
    def test_user_with_text_only_stays_a_string(self) -> None:
        assert m.user("hi") == {"role": "user", "content": "hi"}

    def test_user_with_image_builds_blocks(self) -> None:
        msg = m.user("What is this?", m.image(url="https://example.com/a.png"))
        assert msg["content"] == [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "url": "https://example.com/a.png"},
        ]

    def test_image_from_path_is_base64(self, tmp_path: Any) -> None:
        p = tmp_path / "pic.png"
        p.write_bytes(b"\x89PNG")
        block = m.image(path=p)
        assert block == {"type": "image", "base64": "iVBORw==", "mime_type": "image/png"}

    def test_file_requires_mime_type_for_raw_bytes(self) -> None:
        with pytest.raises(ValueError, match="mime_type"):
            m.file(data=b"%PDF")

    def test_exactly_one_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            m.image(url="https://x", data=b"y", mime_type="image/png")

    def test_system_cache_marks_last_block(self) -> None:
        msg = m.system("rules", cache=True)
        assert msg["content"] == [
            {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
        ]

    def test_assistant_and_tool_result(self) -> None:
        call = ToolCall(id="c1", name="add", arguments={"a": 1})
        assert m.assistant("", [call])["tool_calls"] == [
            {"id": "c1", "name": "add", "arguments": {"a": 1}}
        ]
        assert m.tool_result("c1", "boom", is_error=True) == {
            "role": "tool",
            "content": "boom",
            "tool_call_id": "c1",
            "is_error": True,
        }

    def test_to_lc_messages_tool_round_trip(self) -> None:
        call = ToolCall(id="c1", name="add", arguments={"a": 1})
        lc = to_lc_messages([m.assistant("thinking", [call]), m.tool_result("c1", "2")])
        assert isinstance(lc[0], AIMessage)
        assert lc[0].tool_calls[0]["id"] == "c1"
        assert lc[0].tool_calls[0]["args"] == {"a": 1}
        assert isinstance(lc[1], ToolMessage)
        assert lc[1].tool_call_id == "c1"
        assert lc[1].status == "success"


# ---------------------------------------------------------------------------
# Request payloads (real LangChain formatting, no network)
# ---------------------------------------------------------------------------


class TestAnthropicPayload:
    def test_system_prompt_gets_cache_breakpoint(self) -> None:
        payload = _anthropic_payload(_anthropic(), [m.system("Be brief."), m.user("hi")])
        assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in payload

    def test_auto_mode_uses_top_level_cache_control(self) -> None:
        payload = _anthropic_payload(_anthropic("auto"), [m.system("Be brief."), m.user("hi")])
        assert payload["cache_control"] == {"type": "ephemeral"}

    def test_off_mode_sends_no_breakpoints(self) -> None:
        payload = _anthropic_payload(_anthropic("off"), [m.system("Be brief."), m.user("hi")])
        assert "cache_control" not in payload
        assert "cache_control" not in str(payload["system"])

    def test_tool_turns_and_images_are_formatted(self) -> None:
        call = ToolCall(id="toolu_1", name="add", arguments={"a": 1, "b": 2})
        payload = _anthropic_payload(
            _anthropic("off"),
            [
                m.user("Add these", m.image(data=b"img", mime_type="image/png")),
                m.assistant("", [call]),
                m.tool_result("toolu_1", "3"),
            ],
        )
        user, assistant, result = payload["messages"]
        assert user["content"][1]["type"] == "image"
        assert assistant["content"][-1] == {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "add",
            "input": {"a": 1, "b": 2},
        }
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "toolu_1"


class TestOpenAIPayload:
    def test_tool_turns_and_images_are_formatted(self) -> None:
        provider = _openai()
        call = ToolCall(id="call_1", name="add", arguments={"a": 1})
        lc = to_lc_messages(
            [
                m.user("Look", m.image(url="https://example.com/a.png")),
                m.assistant("", [call]),
                m.tool_result("call_1", "2"),
            ]
        )
        payload = provider._llm._get_request_payload(lc)
        user, assistant, tool = payload["messages"]
        assert user["content"][1]["type"] == "image_url"
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert tool == {"role": "tool", "content": "2", "tool_call_id": "call_1"}


# ---------------------------------------------------------------------------
# Provider results
# ---------------------------------------------------------------------------


class TestProviderResult:
    def test_text_tool_calls_and_cache_usage(self) -> None:
        provider = _anthropic()
        response = AIMessage(
            content=[
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Let me add."},
            ],
            tool_calls=[{"id": "toolu_1", "name": "add", "args": {"a": 1}, "type": "tool_call"}],
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 20,
                "total_tokens": 1020,
                "input_token_details": {"cache_read": 800, "cache_creation": 0},
            },
            response_metadata={"stop_reason": "tool_use"},
        )
        provider._llm = MagicMock()
        provider._llm.bind_tools.return_value.invoke.return_value = response

        result = provider.complete([m.user("1+1?")], tools=["tool-sentinel"])

        provider._llm.bind_tools.assert_called_once_with(["tool-sentinel"])
        assert result.text == "Let me add."
        assert result.tool_calls == [ToolCall(id="toolu_1", name="add", arguments={"a": 1})]
        assert result.finish_reason == "tool_calls"
        assert result.cache_read_tokens == 800
        assert result.cost == pytest.approx(
            calculate_cost("claude-opus-5-5", 1000, 20, cache_read_tokens=800)
        )

    def test_structured_output_uses_native_json_schema(self) -> None:
        provider = _anthropic()
        provider._llm = MagicMock()
        structured = provider._llm.with_structured_output.return_value
        structured.invoke.return_value = {
            "raw": AIMessage(content='{"city": "Oslo", "celsius": 4}'),
            "parsed": Weather(city="Oslo", celsius=4),
            "parsing_error": None,
        }

        result = provider.complete([m.user("Weather?")], response_format=Weather)

        provider._llm.with_structured_output.assert_called_once_with(
            Weather, method="json_schema", include_raw=True
        )
        assert result.parsed == {"city": "Oslo", "celsius": 4.0}
        assert result.text == '{"city": "Oslo", "celsius": 4.0}'

    def test_dict_schema_gets_a_title(self) -> None:
        provider = _openai()
        provider._llm = MagicMock()
        provider._llm.with_structured_output.return_value.invoke.return_value = {
            "raw": AIMessage(content="{}"),
            "parsed": {"x": 1},
            "parsing_error": None,
        }
        provider.complete([m.user("x")], response_format={"type": "object"})
        schema = provider._llm.with_structured_output.call_args.args[0]
        assert schema == {"title": "output", "type": "object"}

    def test_structured_output_falls_back_to_plain_call(self) -> None:
        provider = _anthropic()
        provider._llm = MagicMock()
        provider._llm.with_structured_output.return_value.invoke.side_effect = Exception(
            "schema not supported"
        )
        provider._llm.invoke.return_value = AIMessage(content='{"city": "Oslo", "celsius": 4}')

        result = provider.complete([m.user("Weather?")], response_format=Weather)

        assert result.text == '{"city": "Oslo", "celsius": 4}'
        assert result.parsed is None

    def test_transient_errors_do_not_trigger_fallback(self) -> None:
        provider = _anthropic()
        with pytest.raises(RateLimitError):
            provider._structured_fallback(RateLimitError("429"), {"response_format": Weather})

    def test_plain_errors_without_response_format_propagate(self) -> None:
        provider = _anthropic()
        with pytest.raises(ProviderError):
            provider._structured_fallback(ProviderError("400"), {})


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestCachePricing:
    def test_claude_cache_reads_and_writes(self) -> None:
        # Opus 5.5: $4/M input → reads $0.40/M, writes $5/M.
        cost = calculate_cost(
            "claude-opus-5-5",
            1_000_000,
            0,
            cache_read_tokens=500_000,
            cache_write_tokens=250_000,
        )
        assert cost == pytest.approx(0.25 * 4 + 0.5 * 0.4 + 0.25 * 5)

    def test_openai_cached_input_is_half_price(self) -> None:
        cost = calculate_cost("gpt-4o", 1_000_000, 0, cache_read_tokens=1_000_000)
        assert cost == pytest.approx(1.25)

    def test_no_cache_tokens_is_unchanged(self) -> None:
        assert calculate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)


# ---------------------------------------------------------------------------
# Capability discovery and module wiring
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_builtin_providers_declare_native_features(self) -> None:
        from aiforge.providers import provider_supports

        settings = AIForgeSettings(MODEL="gpt-4o", OPENAI_API_KEY="test")
        assert provider_supports(settings, "supports_tools")
        assert provider_supports(settings, "supports_structured_output")

    def test_factory_registered_provider_reports_false(self) -> None:
        from aiforge.providers import _REGISTRY, provider_supports, register_provider

        register_provider("factory_only", lambda s: MagicMock(), lambda s: MagicMock())
        try:
            settings = AIForgeSettings(PROVIDER="factory_only", MODEL="x")
            assert not provider_supports(settings, "supports_tools")
        finally:
            _REGISTRY.pop("factory_only", None)

    def test_extractor_requests_native_structured_output(self) -> None:
        from aiforge.modules.extract.module import Extractor

        extractor = Extractor(output_schema=Weather, OPENAI_API_KEY="test")
        fake = CompletionResult(
            text='{"city": "Oslo", "celsius": 4}', input_tokens=1, output_tokens=1, model="gpt-4o"
        )
        with patch.object(extractor, "_call_llm", return_value=fake) as call:
            result = extractor.extract("It is 4C in Oslo")
        assert call.call_args.kwargs["response_format"] is Weather
        assert result.data == {"city": "Oslo", "celsius": 4.0}

    def test_classifier_schema_restricts_labels(self) -> None:
        from aiforge.modules.classify.module import Classifier

        clf = Classifier(labels=["spam", "ham"], OPENAI_API_KEY="test")
        schema = clf._output_schema()
        assert schema["properties"]["label"]["enum"] == ["spam", "ham"]


# ---------------------------------------------------------------------------
# ChatAgent native tool loop
# ---------------------------------------------------------------------------


class TestNativeToolLoop:
    @staticmethod
    def _add_tool() -> Any:
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        return add

    def test_tool_call_then_answer(self) -> None:
        from aiforge.modules.chat.agent import build_chat_graph

        replies = iter(
            [
                CompletionResult(
                    text="",
                    input_tokens=10,
                    output_tokens=5,
                    model="gpt-4o",
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
                ),
                CompletionResult(
                    text="2 + 3 = 5", input_tokens=20, output_tokens=5, model="gpt-4o"
                ),
            ]
        )
        seen: list[list[dict[str, Any]]] = []

        def llm_fn(msgs: list[dict[str, Any]]) -> CompletionResult:
            seen.append(list(msgs))
            return next(replies)

        graph = build_chat_graph(llm_fn, tools=[self._add_tool()], native_tools=True)
        final = graph.invoke(
            {
                "messages": [m.user("2+3?")],
                "current_input": "2+3?",
                "tool_calls": [],
                "tool_results": [],
                "iteration": 0,
                "max_tool_iterations": 3,
                "trace": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "model": "",
            }
        )

        # Second LLM call sees the assistant tool call and its tool result.
        assert seen[1][-2]["tool_calls"] == [
            {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}}
        ]
        assert seen[1][-1] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "5",
            "is_error": False,
        }
        assert final["messages"][-1] == {"role": "assistant", "content": "2 + 3 = 5"}
        assert final["tool_results"][0]["output"] == "5"
        assert final["total_cost"] > 0

    def test_unknown_tool_is_reported_as_error(self) -> None:
        from aiforge.modules.chat.agent import build_chat_graph

        replies = iter(
            [
                CompletionResult(
                    text="",
                    input_tokens=1,
                    output_tokens=1,
                    model="gpt-4o",
                    tool_calls=[ToolCall(id="c1", name="nope", arguments={})],
                ),
                CompletionResult(text="sorry", input_tokens=1, output_tokens=1, model="gpt-4o"),
            ]
        )
        seen: list[list[dict[str, Any]]] = []

        def llm_fn(msgs: list[dict[str, Any]]) -> CompletionResult:
            seen.append(list(msgs))
            return next(replies)

        graph = build_chat_graph(llm_fn, tools=[self._add_tool()], native_tools=True)
        graph.invoke(
            {
                "messages": [m.user("go")],
                "current_input": "go",
                "tool_calls": [],
                "tool_results": [],
                "iteration": 0,
                "max_tool_iterations": 3,
                "trace": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "model": "",
            }
        )
        assert seen[1][-1]["is_error"] is True

    def test_chat_agent_offers_tools_to_provider(self) -> None:
        from aiforge.modules.chat.module import ChatAgent

        add = self._add_tool()
        agent = ChatAgent(tools=[add], OPENAI_API_KEY="test")
        assert agent._native_tools
        assert "To use a tool, respond with ONLY a JSON object" not in agent.system_prompt

        fake = CompletionResult(text="hi", input_tokens=1, output_tokens=1, model="gpt-4o")
        with patch.object(agent, "_call_llm", return_value=fake) as call:
            response = agent.chat("hello")
        assert call.call_args.kwargs["tools"] == [add]
        assert call.call_args.kwargs["use_cache"] is False
        assert response.message == "hi"
