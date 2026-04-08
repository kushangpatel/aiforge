"""Unit tests for the Summarizer module."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.modules.summarize.module import (
    Summarizer,
    _count_tokens,
    _split_text,
)
from aiforge.modules.summarize.prompts import (
    SUMMARIZE_BULLET_POINTS,
    SUMMARIZE_CONCISE,
    SUMMARIZE_DETAILED,
    SUMMARIZE_EXECUTIVE,
    SUMMARIZE_MAP,
    SUMMARIZE_REDUCE,
    SUMMARIZE_SYSTEM,
)
from aiforge.modules.summarize.schemas import SummarizeResponse, SummarizeStyle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sum(
    style: str = "concise",
    max_length: int = 500,
    temperature: float = 0.0,
) -> Summarizer:
    """Instantiate a Summarizer bypassing BaseModule provider initialisation."""
    s = object.__new__(Summarizer)
    s.style = cast(SummarizeStyle, style)
    s.max_length = max_length
    s.temperature = temperature

    import structlog

    s.logger = structlog.get_logger()
    s._provider = None
    s._embedding_provider = None
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.MODEL = "gpt-4o-mini"
    settings.CUSTOM_PROMPTS = {}
    s.settings = settings
    s._system_prompt_override = None
    from aiforge.infra.cost_tracker import CostTracker

    s.cost_tracker = CostTracker.get_session()
    return s


def _fake_llm(text: str) -> MagicMock:
    """Create a fake CompletionResult."""
    r = MagicMock()
    r.text = text
    r.input_tokens = 50
    r.output_tokens = 30
    r.model = "gpt-4o-mini"
    return r


# ---------------------------------------------------------------------------
# _count_tokens helper
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_known_model_uses_tiktoken(self) -> None:
        count = _count_tokens("hello world", "gpt-4o-mini")
        assert isinstance(count, int)
        assert count >= 2

    def test_unknown_model_falls_back_to_heuristic(self) -> None:
        text = "a" * 100
        count = _count_tokens(text, "unknown-model-xyz")
        assert count == 25  # len // 4

    def test_empty_text(self) -> None:
        count = _count_tokens("", "gpt-4o-mini")
        assert count >= 0


# ---------------------------------------------------------------------------
# _split_text helper
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_short_text_single_chunk(self) -> None:
        chunks = _split_text("hello world", chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_multiple_chunks(self) -> None:
        text = "word " * 500  # ~2500 chars
        chunks = _split_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2

    def test_overlap_works(self) -> None:
        text = "a" * 1000
        chunks = _split_text(text, chunk_size=400, overlap=100)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_valid_defaults(self) -> None:
        s = _sum()
        assert s.style == "concise"
        assert s.max_length == 500
        assert s.temperature == 0.0

    def test_all_styles_accepted(self) -> None:
        for style in ("concise", "detailed", "bullet_points", "executive"):
            s = _sum(style=style)
            assert s.style == style

    def test_temperature_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            Summarizer(temperature=2.5)

    def test_temperature_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            Summarizer(temperature=-0.1)

    def test_max_length_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            Summarizer(max_length=0)

    def test_max_length_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            Summarizer(max_length=-1)

    def test_invalid_style_raises(self) -> None:
        with pytest.raises(ValueError, match="style"):
            Summarizer(style="haiku")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    def test_explicit_stuff(self) -> None:
        s = _sum()
        assert s._choose_strategy("short text", "stuff") == "stuff"

    def test_explicit_map_reduce(self) -> None:
        s = _sum()
        assert s._choose_strategy("short text", "map_reduce") == "map_reduce"

    def test_invalid_strategy_raises(self) -> None:
        s = _sum()
        with pytest.raises(ValueError, match="strategy"):
            s._choose_strategy("text", "invalid")

    def test_auto_short_text_picks_stuff(self) -> None:
        s = _sum()
        # gpt-4o-mini has 128K context; short text is well under threshold
        result = s._choose_strategy("hello world", None)
        assert result == "stuff"

    def test_auto_long_text_picks_map_reduce(self) -> None:
        s = _sum()
        # Force model to have a tiny context window
        with (
            patch("aiforge.core.pricing.get_context_window", return_value=100),
            patch("aiforge.modules.summarize.module._count_tokens", return_value=90),
        ):
            result = s._choose_strategy("x" * 1000, None)
        assert result == "map_reduce"


# ---------------------------------------------------------------------------
# Stuff strategy (sync)
# ---------------------------------------------------------------------------


class TestStuffSync:
    def test_basic_summarization(self) -> None:
        s = _sum()
        fake = _fake_llm("This is a concise summary.")
        with patch.object(s, "_call_llm", return_value=fake):
            r = s.summarize("Long article text here.")
        assert r.summary == "This is a concise summary."
        assert r.strategy_used == "stuff"
        assert r.chunks_processed == 1
        assert r.input_tokens == 50
        assert r.output_tokens == 30

    def test_detailed_style(self) -> None:
        s = _sum(style="detailed")
        fake = _fake_llm("Detailed summary here.")
        with patch.object(s, "_call_llm", return_value=fake) as mock_call:
            r = s.summarize("Some text.")
        assert r.summary == "Detailed summary here."
        # Verify the detailed prompt template was used
        call_messages = mock_call.call_args[0][0]
        assert "detailed summary" in call_messages[1]["content"].lower()

    def test_bullet_style(self) -> None:
        s = _sum(style="bullet_points")
        fake = _fake_llm("- Point one\n- Point two")
        with patch.object(s, "_call_llm", return_value=fake) as mock_call:
            s.summarize("Some text.")
        call_messages = mock_call.call_args[0][0]
        assert "bullet" in call_messages[1]["content"].lower()

    def test_executive_style(self) -> None:
        s = _sum(style="executive")
        fake = _fake_llm("Executive summary.")
        with patch.object(s, "_call_llm", return_value=fake) as mock_call:
            s.summarize("Some text.")
        call_messages = mock_call.call_args[0][0]
        assert "executive" in call_messages[1]["content"].lower()

    def test_empty_text_raises(self) -> None:
        s = _sum()
        with pytest.raises(ValueError, match="text must not be empty"):
            s.summarize("")

    def test_whitespace_only_raises(self) -> None:
        s = _sum()
        with pytest.raises(ValueError, match="text must not be empty"):
            s.summarize("   \n  ")

    def test_cost_recorded(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        fake.model = "gpt-4o-mini"
        with patch.object(s, "_call_llm", return_value=fake):
            r = s.summarize("Text.")
        assert r.cost >= 0.0
        assert r.model == "gpt-4o-mini"

    def test_latency_recorded(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with patch.object(s, "_call_llm", return_value=fake):
            r = s.summarize("Text.")
        assert r.latency_ms > 0.0

    def test_system_prompt_used(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with patch.object(s, "_call_llm", return_value=fake) as mock_call:
            s.summarize("Text.")
        messages = mock_call.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SUMMARIZE_SYSTEM

    def test_max_length_in_prompt(self) -> None:
        s = _sum(max_length=200)
        fake = _fake_llm("Summary.")
        with patch.object(s, "_call_llm", return_value=fake) as mock_call:
            s.summarize("Text.")
        messages = mock_call.call_args[0][0]
        assert "200" in messages[1]["content"]

    def test_force_stuff_on_long_text(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with patch.object(s, "_call_llm", return_value=fake):
            r = s.summarize("x" * 10000, strategy="stuff")
        assert r.strategy_used == "stuff"
        assert r.chunks_processed == 1


# ---------------------------------------------------------------------------
# Stuff strategy (async)
# ---------------------------------------------------------------------------


class TestStuffAsync:
    async def test_async_basic_summarization(self) -> None:
        s = _sum()
        fake = _fake_llm("Async summary.")
        with patch.object(s, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            r = await s.asummarize("Text.")
        assert r.summary == "Async summary."
        assert r.strategy_used == "stuff"

    async def test_async_empty_text_raises(self) -> None:
        s = _sum()
        with pytest.raises(ValueError, match="text must not be empty"):
            await s.asummarize("")

    async def test_async_cost_and_tokens(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with patch.object(s, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            r = await s.asummarize("Text.")
        assert r.input_tokens == 50
        assert r.output_tokens == 30
        assert r.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Map-reduce strategy (sync)
# ---------------------------------------------------------------------------


class TestMapReduceSync:
    def test_map_reduce_basic(self) -> None:
        s = _sum()
        map_reply = _fake_llm("Chunk summary.")
        reduce_reply = _fake_llm("Final combined summary.")

        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 2 map calls
                return map_reply
            return reduce_reply

        with (
            patch.object(s, "_call_llm", side_effect=_side_effect),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["chunk one", "chunk two"],
            ),
        ):
            r = s.summarize("long text", strategy="map_reduce")

        assert r.summary == "Final combined summary."
        assert r.strategy_used == "map_reduce"
        assert r.chunks_processed == 2
        # 2 map calls + 1 reduce call = 3 total
        assert r.input_tokens == 50 * 3
        assert r.output_tokens == 30 * 3

    def test_map_reduce_single_chunk(self) -> None:
        s = _sum()
        map_reply = _fake_llm("Only chunk summary.")
        reduce_reply = _fake_llm("Final single.")

        call_count = 0

        def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return map_reply
            return reduce_reply

        with (
            patch.object(s, "_call_llm", side_effect=_side_effect),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["single chunk"],
            ),
        ):
            r = s.summarize("text", strategy="map_reduce")

        assert r.strategy_used == "map_reduce"
        assert r.chunks_processed == 1

    def test_map_reduce_cost_accumulates(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with (
            patch.object(s, "_call_llm", return_value=fake),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["c1", "c2", "c3"],
            ),
        ):
            r = s.summarize("text", strategy="map_reduce")
        # 3 map + 1 reduce = 4 calls
        assert r.input_tokens == 50 * 4
        assert r.output_tokens == 30 * 4
        assert r.chunks_processed == 3

    def test_map_uses_map_prompt(self) -> None:
        s = _sum()
        calls: list[list[dict[str, str]]] = []

        def _record_call(messages: list[dict[str, str]], **kwargs: Any) -> MagicMock:
            calls.append(messages)
            return _fake_llm("Summary.")

        with (
            patch.object(s, "_call_llm", side_effect=_record_call),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["chunk1"],
            ),
        ):
            s.summarize("text", strategy="map_reduce")

        # First call is map, second is reduce
        assert "chunk1" in calls[0][1]["content"]
        # Reduce prompt references max_length
        assert "500" in calls[1][1]["content"]

    def test_latency_recorded(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with (
            patch.object(s, "_call_llm", return_value=fake),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["c1"],
            ),
        ):
            r = s.summarize("text", strategy="map_reduce")
        assert r.latency_ms > 0.0


# ---------------------------------------------------------------------------
# Map-reduce strategy (async)
# ---------------------------------------------------------------------------


class TestMapReduceAsync:
    async def test_async_map_reduce_basic(self) -> None:
        s = _sum()
        map_reply = _fake_llm("Chunk summary.")
        reduce_reply = _fake_llm("Final async combined.")

        call_count = 0

        async def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return map_reply
            return reduce_reply

        with (
            patch.object(s, "_acall_llm", new_callable=AsyncMock, side_effect=_side_effect),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["chunk one", "chunk two"],
            ),
        ):
            r = await s.asummarize("long text", strategy="map_reduce")

        assert r.summary == "Final async combined."
        assert r.strategy_used == "map_reduce"
        assert r.chunks_processed == 2

    async def test_async_map_reduce_cost_accumulates(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        with (
            patch.object(s, "_acall_llm", new_callable=AsyncMock, return_value=fake),
            patch(
                "aiforge.modules.summarize.module._split_text",
                return_value=["c1", "c2"],
            ),
        ):
            r = await s.asummarize("text", strategy="map_reduce")
        # 2 map + 1 reduce = 3 calls
        assert r.input_tokens == 50 * 3
        assert r.output_tokens == 30 * 3


# ---------------------------------------------------------------------------
# Pipeline interface (run / arun)
# ---------------------------------------------------------------------------


class TestPipelineInterface:
    def test_run_delegates_to_summarize(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary via run.")
        with patch.object(s, "_call_llm", return_value=fake):
            r = s.run("Text.")
        assert r.summary == "Summary via run."

    async def test_arun_delegates_to_asummarize(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary via arun.")
        with patch.object(s, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            r = await s.arun("Text.")
        assert r.summary == "Summary via arun."


# ---------------------------------------------------------------------------
# Build messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_stuff_concise_messages(self) -> None:
        s = _sum(style="concise", max_length=200)
        msgs = s._build_stuff_messages("Hello world.", SUMMARIZE_SYSTEM)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SUMMARIZE_SYSTEM
        assert "200" in msgs[1]["content"]
        assert "Hello world." in msgs[1]["content"]

    def test_stuff_detailed_messages(self) -> None:
        s = _sum(style="detailed")
        msgs = s._build_stuff_messages("Text.", SUMMARIZE_SYSTEM)
        assert "detailed" in msgs[1]["content"].lower()

    def test_map_messages(self) -> None:
        s = _sum()
        msgs = s._build_map_messages("chunk content", SUMMARIZE_SYSTEM)
        assert msgs[0]["content"] == SUMMARIZE_SYSTEM
        assert "chunk content" in msgs[1]["content"]

    def test_reduce_messages(self) -> None:
        s = _sum(max_length=300)
        msgs = s._build_reduce_messages("summary 1\n\nsummary 2", SUMMARIZE_SYSTEM)
        assert "300" in msgs[1]["content"]
        assert "summary 1" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# SummarizeResponse schema
# ---------------------------------------------------------------------------


class TestSummarizeResponse:
    def test_to_text_returns_summary(self) -> None:
        r = SummarizeResponse(summary="Hello summary")
        assert r.to_text() == "Hello summary"

    def test_to_text_empty(self) -> None:
        r = SummarizeResponse()
        assert r.to_text() == ""

    def test_default_values(self) -> None:
        r = SummarizeResponse()
        assert r.strategy_used == "stuff"
        assert r.chunks_processed == 1
        assert r.cost == 0.0

    def test_inherits_module_response_fields(self) -> None:
        r = SummarizeResponse(
            summary="Test",
            cost=0.001,
            model="gpt-4o-mini",
            latency_ms=150.0,
        )
        assert r.cost == 0.001
        assert r.model == "gpt-4o-mini"
        assert r.latency_ms == 150.0


# ---------------------------------------------------------------------------
# Prompts — sanity checks
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_concise_has_placeholders(self) -> None:
        assert "{text}" in SUMMARIZE_CONCISE
        assert "{max_length}" in SUMMARIZE_CONCISE

    def test_detailed_has_placeholders(self) -> None:
        assert "{text}" in SUMMARIZE_DETAILED

    def test_bullet_has_placeholders(self) -> None:
        assert "{text}" in SUMMARIZE_BULLET_POINTS

    def test_executive_has_placeholders(self) -> None:
        assert "{text}" in SUMMARIZE_EXECUTIVE

    def test_map_has_chunk_placeholder(self) -> None:
        assert "{chunk}" in SUMMARIZE_MAP

    def test_reduce_has_placeholders(self) -> None:
        assert "{summaries}" in SUMMARIZE_REDUCE
        assert "{max_length}" in SUMMARIZE_REDUCE


# ---------------------------------------------------------------------------
# Batch operations — FR-SUM-006
# ---------------------------------------------------------------------------


class TestBatchOperations:
    def test_run_batch_all_succeed(self) -> None:
        s = _sum()
        fake = _fake_llm("Batch summary.")
        with patch.object(s, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            batch_result = s.run_batch(["text one", "text two", "text three"])
        assert len(batch_result.results) == 3
        assert len(batch_result.failed) == 0
        assert all(isinstance(r, SummarizeResponse) for r in batch_result.results)

    def test_run_batch_partial_failure(self) -> None:
        s = _sum()
        fake = _fake_llm("Summary.")
        call_count = 0

        async def _selective_llm(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("llm error")
            return fake

        with patch.object(s, "_acall_llm", new_callable=AsyncMock, side_effect=_selective_llm):
            batch_result = s.run_batch(["q1", "q2", "q3"])
        assert len(batch_result.results) == 2
        assert len(batch_result.failed) == 1
        assert batch_result.failed[0]["error"] == "llm error"

    def test_run_batch_empty_returns_empty(self) -> None:
        s = _sum()
        batch_result = s.run_batch([])
        assert batch_result.results == []
        assert batch_result.failed == []


# ---------------------------------------------------------------------------
# Server route — /summarize
# ---------------------------------------------------------------------------


class TestSummarizeServerRoute:
    def _client(self) -> Any:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from aiforge.server.routes.summarize import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_summarize_returns_200(self) -> None:
        client = self._client()
        mock_result = SummarizeResponse(
            summary="A great summary.",
            strategy_used="stuff",
            chunks_processed=1,
            cost=0.001,
            input_tokens=50,
            output_tokens=30,
            model="gpt-4o-mini",
            latency_ms=100.0,
        )
        with patch("aiforge.modules.summarize.module.Summarizer") as MockSum:
            MockSum.return_value.asummarize = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/summarize",
                json={"text": "Some long article to summarize."},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"] == "A great summary."
        assert body["strategy_used"] == "stuff"

    def test_summarize_with_options(self) -> None:
        client = self._client()
        mock_result = SummarizeResponse(
            summary="Bullet summary.",
            strategy_used="stuff",
            chunks_processed=1,
        )
        with patch("aiforge.modules.summarize.module.Summarizer") as MockSum:
            MockSum.return_value.asummarize = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/summarize",
                json={
                    "text": "Text here.",
                    "style": "bullet_points",
                    "max_length": 200,
                    "strategy": "stuff",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["summary"] == "Bullet summary."

    def test_summarize_missing_text_returns_422(self) -> None:
        client = self._client()
        resp = client.post("/summarize", json={})
        assert resp.status_code == 422

    def test_summarize_value_error_returns_422(self) -> None:
        client = self._client()
        with patch("aiforge.modules.summarize.module.Summarizer") as MockSum:
            MockSum.return_value.asummarize = AsyncMock(
                side_effect=ValueError("text must not be empty"),
            )
            resp = client.post(
                "/summarize",
                json={"text": ""},
            )
        assert resp.status_code == 422
        assert "text must not be empty" in resp.json()["detail"]

    def test_summarize_invalid_style_returns_422(self) -> None:
        client = self._client()
        resp = client.post(
            "/summarize",
            json={"text": "Hello.", "style": "haiku"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# _split_text edge cases
# ---------------------------------------------------------------------------


class TestSplitTextEdgeCases:
    def test_chunk_size_smaller_than_overlap_no_infinite_loop(self) -> None:
        """Regression: chunk_size <= overlap must not cause an infinite loop."""
        text = "a" * 500
        # Force the fallback path by making the TextSplitter import raise
        with patch.dict("sys.modules", {"aiforge.core.text_splitter": None}):
            chunks = _split_text(text, chunk_size=100, overlap=200)
        assert len(chunks) >= 1
        # Every character in the text must appear in at least one chunk
        joined = "".join(chunks)
        assert len(joined) >= len(text)

    def test_chunk_size_equals_overlap_terminates(self) -> None:
        """chunk_size == overlap should still produce chunks without looping."""
        text = "b" * 300
        with patch.dict("sys.modules", {"aiforge.core.text_splitter": None}):
            chunks = _split_text(text, chunk_size=100, overlap=100)
        assert len(chunks) >= 1

    def test_zero_chunk_size_terminates(self) -> None:
        """chunk_size == 0 should not hang; stride clamps to 1."""
        text = "c" * 50
        with patch.dict("sys.modules", {"aiforge.core.text_splitter": None}):
            chunks = _split_text(text, chunk_size=0, overlap=0)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# _get_chunk_size
# ---------------------------------------------------------------------------


class TestGetChunkSize:
    def test_returns_60_percent_of_context_window(self) -> None:
        s = _sum()
        with patch("aiforge.core.pricing.get_context_window", return_value=10_000):
            result = s._get_chunk_size()
        assert result == 6_000

    def test_small_context_window(self) -> None:
        s = _sum()
        with patch("aiforge.core.pricing.get_context_window", return_value=100):
            result = s._get_chunk_size()
        assert result == 60


# ---------------------------------------------------------------------------
# _count_tokens — additional edge cases
# ---------------------------------------------------------------------------


class TestCountTokensExtra:
    def test_unknown_model_empty_text_returns_at_least_one(self) -> None:
        """Heuristic fallback: max(1, len('') // 4) == 1."""
        count = _count_tokens("", "nonexistent-model-xyz")
        assert count >= 1
