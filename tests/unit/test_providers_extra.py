"""Offline tests for the Gemini, Bedrock and Mistral providers.

Each test drives the real LangChain chat model, so request building (messages,
tools, structured output, cache points) and response parsing are LangChain's
own; only the network call at the bottom is replaced:

* Gemini: ``client.models.generate_content`` / ``embed_content`` (google-genai)
* Bedrock: ``client.converse`` / ``invoke_model`` (boto3)
* Mistral: the ``httpx`` client, through an ``httpx.MockTransport``
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from aiforge.core import messages as m
from aiforge.core.pricing import calculate_cost, get_context_window, get_model_pricing
from aiforge.core.types import (
    APIConnectionError,
    APITimeoutError,
    ProviderError,
    RateLimitError,
    ToolCall,
)
from aiforge.infra.config import AIForgeSettings
from aiforge.providers._errors import wrap_sdk_error

pytest.importorskip("langchain_core")


class Weather(BaseModel):
    city: str
    celsius: float


def _add_tool() -> Any:
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return add


def _tool_turns(call_id: str) -> list[dict[str, Any]]:
    """A user question, an assistant tool call and its result."""
    call = ToolCall(id=call_id, name="add", arguments={"a": 1, "b": 2})
    return [
        m.system("Be brief."),
        m.user("Add 1 and 2", m.image(data=b"\x89PNG", mime_type="image/png")),
        m.assistant("", [call]),
        m.tool_result(call_id, "3"),
    ]


# ===========================================================================
# Gemini
# ===========================================================================


def _gemini(**overrides: Any) -> Any:
    pytest.importorskip("langchain_google_genai")
    from aiforge.providers.gemini import GeminiProvider

    return GeminiProvider(AIForgeSettings(PROVIDER="gemini", GOOGLE_API_KEY="test", **overrides))


def _gemini_response(*parts: Any, cached: int = 0, finish: str = "STOP") -> Any:
    from google.genai import types

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=20,
            total_token_count=1020,
            cached_content_token_count=cached,
        ),
    )


def _gemini_call(provider: Any, response: Any, msgs: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Run ``provider.complete`` and return ``(result, request kwargs sent to the SDK)``."""
    with patch.object(
        provider._llm.client.models, "generate_content", return_value=response
    ) as generate:
        result = provider.complete(msgs, **kwargs)
    return result, generate.call_args.kwargs


class TestGemini:
    def test_construction_from_settings(self) -> None:
        provider = _gemini(TEMPERATURE=0.3, MAX_TOKENS=256)
        assert provider.model_name == "gemini-3.8-flash"
        assert provider._llm.model == "gemini-3.8-flash"
        assert provider._llm.temperature == 0.3
        assert provider._llm.max_output_tokens == 256
        assert provider._llm.google_api_key.get_secret_value() == "test"

    def test_defaults_follow_provider(self) -> None:
        settings = AIForgeSettings(PROVIDER="gemini", GOOGLE_API_KEY="test")
        assert (settings.MODEL, settings.EMBEDDING_MODEL) == (
            "gemini-3.8-flash",
            "gemini-embedding-2",
        )
        explicit = AIForgeSettings(PROVIDER="gemini", MODEL="gemini-2.5-pro", GOOGLE_API_KEY="test")
        assert explicit.MODEL == "gemini-2.5-pro"

    def test_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.providers.gemini import GeminiEmbeddingProvider, GeminiProvider

        monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
        settings = AIForgeSettings(PROVIDER="gemini", GOOGLE_API_KEY="test")
        with pytest.raises(ImportError, match=r"aiforge-kit\[gemini\]"):
            GeminiProvider(settings)
        with pytest.raises(ImportError, match=r"aiforge-kit\[gemini\]"):
            GeminiEmbeddingProvider(settings)

    def test_messages_and_tool_turns(self) -> None:
        from google.genai import types

        provider = _gemini()
        result, request = _gemini_call(
            provider, _gemini_response(types.Part(text="3")), _tool_turns("call_1")
        )
        assert result.text == "3"
        config = request["config"]
        assert config.system_instruction.parts[0].text == "Be brief."
        user, model, tool = request["contents"]
        assert user.role == "user"
        assert user.parts[1].inline_data.mime_type == "image/png"
        assert model.role == "model"
        assert model.parts[0].function_call.name == "add"
        assert model.parts[0].function_call.args == {"a": 1, "b": 2}
        assert tool.parts[0].function_response.name == "add"

    def test_tool_calls_usage_and_cache_tokens(self) -> None:
        from google.genai import types

        provider = _gemini()
        response = _gemini_response(
            types.Part(text="Let me add."),
            types.Part(function_call=types.FunctionCall(id="fc1", name="add", args={"a": 1})),
            cached=800,
        )
        result, request = _gemini_call(
            provider, response, [m.user("1+1?")], tools=[_add_tool()], tool_choice="add"
        )

        declaration = request["config"].tools[0].function_declarations[0]
        assert declaration.name == "add"
        assert request["config"].tool_config.function_calling_config.allowed_function_names == [
            "add"
        ]
        assert result.text == "Let me add."
        assert [(c.name, c.arguments) for c in result.tool_calls] == [("add", {"a": 1})]
        assert result.finish_reason == "tool_calls"
        assert (result.input_tokens, result.output_tokens) == (1000, 20)
        assert result.cache_read_tokens == 800
        assert result.cost == pytest.approx(
            calculate_cost("gemini-3.8-flash", 1000, 20, cache_read_tokens=800)
        )
        # Cached input is billed at the explicit cached rate ($0.075/M), not 0.5x.
        assert result.cost == pytest.approx((200 * 0.75 + 800 * 0.075 + 20 * 3.75) / 1e6)

    def test_structured_output_is_native_json_schema(self) -> None:
        from google.genai import types

        provider = _gemini()
        response = _gemini_response(types.Part(text='{"city": "Oslo", "celsius": 4}'))
        result, request = _gemini_call(
            provider, response, [m.user("Weather?")], response_format=Weather
        )

        assert request["config"].response_mime_type == "application/json"
        assert request["config"].response_json_schema["properties"].keys() == {"city", "celsius"}
        assert result.parsed == {"city": "Oslo", "celsius": 4.0}

    def test_rate_limit_is_wrapped(self) -> None:
        from google.genai.errors import ClientError

        provider = _gemini()
        error = ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        with (
            patch.object(provider._llm.client.models, "generate_content", side_effect=error),
            pytest.raises(RateLimitError),
        ):
            provider._invoke([m.user("hi")], {})

    def test_bad_request_is_provider_error(self) -> None:
        from google.genai.errors import ClientError

        provider = _gemini()
        error = ClientError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}})
        with (
            patch.object(provider._llm.client.models, "generate_content", side_effect=error),
            pytest.raises(ProviderError) as info,
        ):
            provider._invoke([m.user("hi")], {})
        assert type(info.value) is ProviderError

    def test_embeddings(self) -> None:
        pytest.importorskip("langchain_google_genai")
        from google.genai import types

        from aiforge.providers.gemini import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(
            AIForgeSettings(PROVIDER="gemini", GOOGLE_API_KEY="test")
        )
        response = types.EmbedContentResponse(
            embeddings=[
                types.ContentEmbedding(values=[0.1, 0.2]),
                types.ContentEmbedding(values=[0.3, 0.4]),
            ]
        )
        client = provider._embeddings.client
        with patch.object(client.models, "embed_content", return_value=response) as embed:
            vectors = provider.embed(["a", "b"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert embed.call_args.kwargs["model"] == "gemini-embedding-2"
        assert provider.model_name == "gemini-embedding-2"


# ===========================================================================
# Bedrock
# ===========================================================================


def _bedrock(**overrides: Any) -> Any:
    pytest.importorskip("langchain_aws")
    from aiforge.providers.bedrock import BedrockProvider

    return BedrockProvider(AIForgeSettings(PROVIDER="bedrock", AWS_REGION="us-east-1", **overrides))


def _converse_response(*content: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": list(content)}},
        "stopReason": stop,
        "usage": {
            "inputTokens": 150,
            "outputTokens": 20,
            "totalTokens": 1020,
            "cacheReadInputTokens": 800,
            "cacheWriteInputTokens": 50,
        },
        "metrics": {"latencyMs": 5},
    }


def _bedrock_call(
    provider: Any, response: dict[str, Any], msgs: list[dict[str, Any]], **kw: Any
) -> Any:
    with patch.object(provider._llm.client, "converse", return_value=response) as converse:
        result = provider.complete(msgs, **kw)
    return result, converse.call_args.kwargs


class TestBedrock:
    def test_construction_from_settings(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A named profile must exist in the AWS config; credentials stay with AWS.
        config = tmp_path / "config"
        config.write_text("[profile dev]\nregion = us-east-1\n")
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
        provider = _bedrock(AWS_PROFILE="dev", MAX_TOKENS=512)
        llm = provider._llm
        assert provider.model_name == "global.anthropic.claude-sonnet-5-5"
        assert llm.model_id == "global.anthropic.claude-sonnet-5-5"
        assert llm.region_name == "us-east-1"
        assert llm.credentials_profile_name == "dev"
        assert llm.max_tokens == 512
        # Claude 5.x rejects sampling parameters, also on Bedrock.
        assert llm.temperature is None

    def test_nova_keeps_temperature(self) -> None:
        provider = _bedrock(MODEL="us.amazon.nova-pro-v1:0", TEMPERATURE=0.2)
        assert provider._llm.temperature == 0.2

    def test_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.providers.bedrock import BedrockEmbeddingProvider, BedrockProvider

        monkeypatch.setitem(sys.modules, "langchain_aws", None)
        settings = AIForgeSettings(PROVIDER="bedrock", AWS_REGION="us-east-1")
        with pytest.raises(ImportError, match=r"aiforge-kit\[bedrock\]"):
            BedrockProvider(settings)
        with pytest.raises(ImportError, match=r"aiforge-kit\[bedrock\]"):
            BedrockEmbeddingProvider(settings)

    def test_messages_tool_turns_and_system_cache_point(self) -> None:
        provider = _bedrock()
        _, request = _bedrock_call(
            provider,
            _converse_response({"text": "3"}),
            _tool_turns("tooluse_1"),
            tools=[_add_tool()],
        )
        assert request["modelId"] == "global.anthropic.claude-sonnet-5-5"
        assert request["system"] == [{"text": "Be brief."}, {"cachePoint": {"type": "default"}}]
        user, assistant, result = request["messages"]
        assert user["content"][1]["image"]["format"] == "png"
        assert assistant["content"][-1]["toolUse"] == {
            "toolUseId": "tooluse_1",
            "name": "add",
            "input": {"a": 1, "b": 2},
        }
        assert result["content"][0]["toolResult"]["toolUseId"] == "tooluse_1"
        assert request["toolConfig"]["tools"][0]["toolSpec"]["name"] == "add"

    def test_prompt_cache_modes(self) -> None:
        msgs = [m.system("Be brief."), m.user("hi")]
        _, off = _bedrock_call(_bedrock(PROMPT_CACHE="off"), _converse_response(), msgs)
        assert "cachePoint" not in json.dumps(off)

        _, auto = _bedrock_call(_bedrock(PROMPT_CACHE="auto"), _converse_response(), msgs)
        assert auto["system"][-1] == {"cachePoint": {"type": "default"}}
        assert auto["messages"][-1]["content"][-1] == {"cachePoint": {"type": "default"}}

        # Explicit markers become cache points, and no duplicate is added.
        _, marked = _bedrock_call(
            _bedrock(), _converse_response(), [m.system("Rules", cache=True), m.user("hi")]
        )
        assert marked["system"] == [{"text": "Rules"}, {"cachePoint": {"type": "default"}}]

    def test_models_without_caching_get_no_cache_points(self) -> None:
        provider = _bedrock(MODEL="meta.llama3-70b-instruct-v1:0")
        _, request = _bedrock_call(
            provider, _converse_response(), [m.system("Rules", cache=True), m.user("hi")]
        )
        assert request["system"] == [{"text": "Rules"}]

    def test_tool_calls_usage_and_cache_tokens(self) -> None:
        provider = _bedrock()
        response = _converse_response(
            {"text": "Let me add."},
            {"toolUse": {"toolUseId": "t1", "name": "add", "input": {"a": 1}}},
            stop="tool_use",
        )
        result, _ = _bedrock_call(provider, response, [m.user("1+1?")], tools=[_add_tool()])
        assert result.text == "Let me add."
        assert result.tool_calls == [ToolCall(id="t1", name="add", arguments={"a": 1})]
        assert result.finish_reason == "tool_calls"
        # Bedrock reports uncached input separately; LangChain sums all three.
        assert (result.input_tokens, result.output_tokens) == (1000, 20)
        assert (result.cache_read_tokens, result.cache_write_tokens) == (800, 50)
        assert result.cost == pytest.approx((150 * 2.00 + 800 * 0.10 + 50 * 2.50 + 20 * 10) / 1e6)

    def test_structured_output_uses_output_config(self) -> None:
        provider = _bedrock()
        result, request = _bedrock_call(
            provider,
            _converse_response({"text": '{"city": "Oslo", "celsius": 4}'}),
            [m.user("Weather?")],
            response_format=Weather,
        )
        schema = request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
        assert json.loads(schema["schema"])["properties"].keys() == {"city", "celsius"}
        assert result.parsed == {"city": "Oslo", "celsius": 4.0}

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("ThrottlingException", RateLimitError),
            ("ModelTimeoutException", APITimeoutError),
            ("AccessDeniedException", ProviderError),
        ],
    )
    def test_client_errors_are_wrapped(self, code: str, expected: type[Exception]) -> None:
        from botocore.exceptions import ClientError

        provider = _bedrock()
        error = ClientError({"Error": {"Code": code, "Message": "nope"}}, "Converse")
        with (
            patch.object(provider._llm.client, "converse", side_effect=error),
            pytest.raises(ProviderError) as info,
        ):
            provider._invoke([m.user("hi")], {})
        assert type(info.value) is expected

    def test_connection_errors_are_wrapped(self) -> None:
        from botocore.exceptions import EndpointConnectionError, ReadTimeoutError

        assert isinstance(
            wrap_sdk_error(EndpointConnectionError(endpoint_url="https://x")), APIConnectionError
        )
        assert isinstance(
            wrap_sdk_error(ReadTimeoutError(endpoint_url="https://x")), APITimeoutError
        )

    def test_embeddings(self) -> None:
        pytest.importorskip("langchain_aws")
        from aiforge.providers.bedrock import BedrockEmbeddingProvider

        provider = BedrockEmbeddingProvider(
            AIForgeSettings(PROVIDER="bedrock", AWS_REGION="us-east-1")
        )
        body = json.dumps({"embedding": [0.5, 0.25], "inputTextTokenCount": 2}).encode()
        client = provider._embeddings.client
        with patch.object(
            client, "invoke_model", side_effect=lambda **kw: {"body": io.BytesIO(body)}
        ) as invoke:
            vectors = provider.embed(["hello", "world"])
            query = provider.embed_query("hello")
        assert vectors == [[0.5, 0.25], [0.5, 0.25]]
        assert query == [0.5, 0.25]
        assert invoke.call_args.kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
        assert json.loads(invoke.call_args.kwargs["body"])["inputText"] == "hello"

    def test_embedding_errors_are_wrapped(self) -> None:
        pytest.importorskip("langchain_aws")
        from botocore.exceptions import ClientError

        from aiforge.providers.bedrock import BedrockEmbeddingProvider

        provider = BedrockEmbeddingProvider(
            AIForgeSettings(PROVIDER="bedrock", AWS_REGION="us-east-1")
        )
        error = ClientError({"Error": {"Code": "ValidationException", "Message": "x"}}, "Invoke")
        with (
            patch.object(provider._embeddings.client, "invoke_model", side_effect=error),
            pytest.raises(ProviderError),
        ):
            provider.embed_query("hello")


# ===========================================================================
# Mistral
# ===========================================================================


class _MistralServer:
    """An ``httpx.MockTransport`` handler that records requests and replies with canned JSON."""

    def __init__(self, reply: dict[str, Any], status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        import httpx

        self.requests.append(json.loads(request.content))
        return httpx.Response(self.status, json=self.reply)


def _mistral(server: _MistralServer, **overrides: Any) -> Any:
    pytest.importorskip("langchain_mistralai")
    import httpx

    from aiforge.providers.mistral import MistralProvider

    provider = MistralProvider(
        AIForgeSettings(PROVIDER="mistral", MISTRAL_API_KEY="test", **overrides)
    )
    transport = httpx.MockTransport(server)
    base = "https://api.mistral.ai/v1"
    provider._llm.client = httpx.Client(base_url=base, transport=transport)
    provider._llm.async_client = httpx.AsyncClient(base_url=base, transport=transport)
    return provider


def _mistral_reply(
    message: dict[str, Any], finish: str = "stop", cached: int = 0
) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "mistral-medium-3-5",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


class TestMistral:
    def test_construction_from_settings(self) -> None:
        pytest.importorskip("langchain_mistralai")
        from aiforge.providers.mistral import MistralProvider

        provider = MistralProvider(
            AIForgeSettings(PROVIDER="mistral", MISTRAL_API_KEY="test", TEMPERATURE=1.5)
        )
        assert provider.model_name == "mistral-medium-3-5"
        assert provider._llm.model == "mistral-medium-3-5"
        # Mistral only accepts temperatures up to 1.0.
        assert provider._llm.temperature == 1.0
        assert provider._llm.mistral_api_key.get_secret_value() == "test"

    def test_missing_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.providers.mistral import MistralEmbeddingProvider, MistralProvider

        monkeypatch.setitem(sys.modules, "langchain_mistralai", None)
        settings = AIForgeSettings(PROVIDER="mistral", MISTRAL_API_KEY="test")
        with pytest.raises(ImportError, match=r"aiforge-kit\[mistral\]"):
            MistralProvider(settings)
        with pytest.raises(ImportError, match=r"aiforge-kit\[mistral\]"):
            MistralEmbeddingProvider(settings)

    def test_messages_and_tool_turns(self) -> None:
        server = _MistralServer(_mistral_reply({"role": "assistant", "content": "3"}))
        provider = _mistral(server)
        result = provider.complete(_tool_turns("abcDEF123"), tools=[_add_tool()])

        body = server.requests[0]
        assert body["model"] == "mistral-medium-3-5"
        system, user, assistant, tool = body["messages"]
        assert system == {"role": "system", "content": "Be brief."}
        assert user["content"][1]["type"] == "image_url"
        assert assistant["tool_calls"][0]["id"] == "abcDEF123"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}
        assert tool["role"] == "tool" and tool["tool_call_id"] == "abcDEF123"
        assert body["tools"][0]["function"]["name"] == "add"
        assert result.text == "3"

    def test_tool_calls_usage_and_cache_tokens(self) -> None:
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abcDEF123",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1}'},
                }
            ],
        }
        server = _MistralServer(_mistral_reply(message, finish="tool_calls", cached=800))
        result = _mistral(server).complete([m.user("1+1?")], tools=[_add_tool()], tool_choice="any")

        assert server.requests[0]["tool_choice"] == "any"
        assert result.tool_calls == [ToolCall(id="abcDEF123", name="add", arguments={"a": 1})]
        assert result.finish_reason == "tool_calls"
        assert (result.input_tokens, result.output_tokens) == (1000, 20)
        assert result.cache_read_tokens == 800
        assert result.cost == pytest.approx((200 * 1.50 + 800 * 0.15 + 20 * 7.50) / 1e6)

    def test_structured_output_is_native_json_schema(self) -> None:
        reply = _mistral_reply({"role": "assistant", "content": '{"city": "Oslo", "celsius": 4}'})
        server = _MistralServer(reply)
        result = _mistral(server).complete([m.user("Weather?")], response_format=Weather)

        response_format = server.requests[0]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"]["properties"].keys() == {"city", "celsius"}
        assert result.parsed == {"city": "Oslo", "celsius": 4.0}

    async def test_async_completion(self) -> None:
        server = _MistralServer(_mistral_reply({"role": "assistant", "content": "hi"}, cached=0))
        result = await _mistral(server).complete_async([m.user("hello")])
        assert result.text == "hi"
        assert result.cache_read_tokens == 0

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(429, RateLimitError), (504, APITimeoutError), (401, ProviderError)],
    )
    def test_http_errors_are_wrapped(self, status: int, expected: type[Exception]) -> None:
        server = _MistralServer({"message": "error"}, status=status)
        with pytest.raises(ProviderError) as info:
            _mistral(server)._invoke([m.user("hi")], {})
        assert type(info.value) is expected

    def test_network_errors_are_wrapped(self) -> None:
        import httpx

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        provider = _mistral(_MistralServer({}))
        provider._llm.client = httpx.Client(transport=httpx.MockTransport(refuse))
        provider._llm.max_retries = 1
        with pytest.raises(APIConnectionError):
            provider._invoke([m.user("hi")], {})

    def test_embeddings(self) -> None:
        pytest.importorskip("langchain_mistralai")
        import httpx

        from aiforge.providers.mistral import MistralEmbeddingProvider

        provider = MistralEmbeddingProvider(
            AIForgeSettings(PROVIDER="mistral", MISTRAL_API_KEY="test")
        )
        server = _MistralServer(
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}], "model": "x"}
        )
        provider._embeddings.client = httpx.Client(
            base_url="https://api.mistral.ai/v1/", transport=httpx.MockTransport(server)
        )
        assert provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
        assert server.requests[0] == {"model": "mistral-embed", "input": ["a", "b"]}

    def test_embeddings_need_no_tokenizer_download(self) -> None:
        pytest.importorskip("langchain_mistralai")
        from aiforge.providers.mistral import _offline_tokenizer

        tokenizer = _offline_tokenizer()
        # Three characters per token: an over-estimate that keeps batches small.
        assert [len(e) for e in tokenizer.encode_batch(["hello world", "abc"])] == [4, 1]


# ===========================================================================
# Shared: registry, settings, pricing, error mapping
# ===========================================================================


class TestRegistry:
    @pytest.mark.parametrize("name", ["gemini", "bedrock", "mistral"])
    def test_registered_and_declares_native_features(self, name: str) -> None:
        from aiforge.providers import list_providers, provider_supports

        pytest.importorskip(
            {"gemini": "langchain_google_genai", "bedrock": "langchain_aws"}.get(
                name, "langchain_mistralai"
            )
        )
        assert name in list_providers()
        settings = AIForgeSettings(PROVIDER=name, GOOGLE_API_KEY="test", MISTRAL_API_KEY="test")
        assert provider_supports(settings, "supports_tools")
        assert provider_supports(settings, "supports_structured_output")

    def test_get_provider_builds_each(self) -> None:
        from aiforge.providers import get_embedding_provider, get_provider

        for name, key in [
            ("gemini", {"GOOGLE_API_KEY": "test"}),
            ("bedrock", {"AWS_REGION": "us-east-1"}),
            ("mistral", {"MISTRAL_API_KEY": "test"}),
        ]:
            settings = AIForgeSettings(PROVIDER=name, **key)
            assert get_provider(settings).model_name == settings.MODEL
            assert get_embedding_provider(settings).model_name == settings.EMBEDDING_MODEL

    def test_missing_sdk_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import aiforge.providers as registry
        from aiforge.core.types import ConfigError

        monkeypatch.setitem(sys.modules, "langchain_mistralai", None)
        registry._REGISTRY.clear()
        registry._BUILTINS_LOADED = False
        settings = AIForgeSettings(PROVIDER="mistral", MISTRAL_API_KEY="test")
        with pytest.raises(ConfigError, match=r"aiforge-kit\[mistral\]"):
            registry.get_provider(settings)


class TestSettings:
    def test_incompatible_embedding_model_is_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not compatible"):
            AIForgeSettings(PROVIDER="mistral", EMBEDDING_MODEL="voyage-3")

    def test_env_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIFORGE_PROVIDER", "bedrock")
        monkeypatch.setenv("AIFORGE_AWS_REGION", "eu-west-1")
        monkeypatch.setenv("AIFORGE_AWS_PROFILE", "prod")
        monkeypatch.setenv("AIFORGE_MODEL", "eu.amazon.nova-pro-v1:0")
        settings = AIForgeSettings()
        assert (settings.AWS_REGION, settings.AWS_PROFILE) == ("eu-west-1", "prod")
        assert settings.MODEL == "eu.amazon.nova-pro-v1:0"
        assert settings.EMBEDDING_MODEL == "amazon.titan-embed-text-v2:0"

    def test_module_provider_override_gets_provider_defaults(self) -> None:
        pytest.importorskip("langchain_mistralai")
        from aiforge.modules.summarize.module import Summarizer

        summarizer = Summarizer(provider="mistral", MISTRAL_API_KEY="test")
        assert summarizer.settings.MODEL == "mistral-medium-3-5"
        assert summarizer.settings.EMBEDDING_MODEL == "mistral-embed"
        assert summarizer.provider.model_name == "mistral-medium-3-5"

    def test_missing_api_key_warns(self) -> None:
        with pytest.warns(UserWarning, match="AIFORGE_GOOGLE_API_KEY"):
            AIForgeSettings(PROVIDER="gemini")


class TestPricing:
    @pytest.mark.parametrize(
        ("model", "key"),
        [
            ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-haiku-4-5"),
            ("eu.anthropic.claude-sonnet-5-5", "anthropic.claude-sonnet-5-5"),
            ("global.anthropic.claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5"),
            ("global.amazon.nova-2-lite-v1:0", "global.amazon.nova-2-lite-v1"),
            ("apac.amazon.nova-pro-v1:0", "amazon.nova-pro-v1"),
            ("amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v2"),
        ],
    )
    def test_bedrock_ids_resolve(self, model: str, key: str) -> None:
        from aiforge.core.pricing import PRICING

        assert get_model_pricing(model) == PRICING[key]

    def test_global_and_regional_prices_differ(self) -> None:
        regional = calculate_cost("us.anthropic.claude-sonnet-5-5", 1_000_000, 1_000_000)
        global_ = calculate_cost("global.anthropic.claude-sonnet-5-5", 1_000_000, 1_000_000)
        assert (global_, regional) == (pytest.approx(12.0), pytest.approx(13.2))

    def test_context_windows(self) -> None:
        assert get_context_window("gemini-3.8-flash") == 1_048_576
        assert get_context_window("us.anthropic.claude-haiku-4-5-20251001-v1:0") == 200_000
        assert get_context_window("global.anthropic.claude-opus-5-5") == 1_000_000
        assert get_context_window("us.amazon.nova-micro-v1:0") == 128_000
        assert get_context_window("mistral-medium-3-5") == 256_000

    def test_every_default_model_is_priced(self) -> None:
        from aiforge.infra.config import _PROVIDER_DEFAULTS

        for model, embedding in _PROVIDER_DEFAULTS.values():
            assert get_model_pricing(model) is not None
            assert get_model_pricing(embedding) is not None

    def test_bedrock_claude_ids_drop_sampling(self) -> None:
        from aiforge.providers._utils import supports_sampling_params

        assert not supports_sampling_params("global.anthropic.claude-sonnet-5-5")
        assert not supports_sampling_params("anthropic.claude-opus-4-7")
        assert supports_sampling_params("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        assert supports_sampling_params("amazon.nova-pro-v1:0")


class TestErrorMapping:
    def test_wrapped_cause_is_followed(self) -> None:
        class SDKError(Exception):
            status_code = 429

        try:
            try:
                raise SDKError("slow down")
            except SDKError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            assert isinstance(wrap_sdk_error(outer), RateLimitError)

    def test_provider_errors_pass_through(self) -> None:
        err = APITimeoutError("t")
        assert wrap_sdk_error(err) is err

    def test_httpx_timeout(self) -> None:
        import httpx

        assert isinstance(wrap_sdk_error(httpx.ReadTimeout("slow")), APITimeoutError)
