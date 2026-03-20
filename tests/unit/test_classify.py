"""Unit tests for the Classifier module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.modules.classify.module import Classifier, _extract_json, _fmt_examples, _fmt_labels
from aiforge.modules.classify.schemas import ClassifyExample, ClassifyResponse, LabelScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABELS = ["positive", "negative", "neutral"]


def _clf(
    labels: list[str] | None = None,
    examples: list[ClassifyExample] | None = None,
    multi_label: bool = False,
    temperature: float = 0.0,
    min_confidence: float = 0.0,
) -> Classifier:
    """Instantiate a Classifier bypassing BaseModule provider initialisation."""
    c = object.__new__(Classifier)
    c.labels = labels if labels is not None else list(_LABELS)
    c.examples = examples or []
    c.multi_label = multi_label
    c.temperature = temperature
    c.min_confidence = min_confidence

    import structlog
    from unittest.mock import MagicMock as MM

    c.logger = structlog.get_logger()
    c._provider = None
    c._embedding_provider = None
    settings = MM()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.CUSTOM_PROMPTS = {}
    c.settings = settings
    c._system_prompt_override = None
    from aiforge.infra.cost_tracker import CostTracker
    c.cost_tracker = CostTracker.get_session()
    return c


def _fake_llm(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    r.input_tokens = 10
    r.output_tokens = 5
    r.model = "gpt-4o-mini"
    return r


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestClassifierConstructor:
    def test_empty_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="labels must be non-empty"):
            Classifier(labels=[])

    def test_too_many_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="at most 100"):
            Classifier(labels=[str(i) for i in range(101)])

    def test_valid_construction(self) -> None:
        c = Classifier(labels=["a", "b", "c"])
        assert c.labels == ["a", "b", "c"]
        assert c.multi_label is False
        assert c.examples == []

    def test_duplicate_labels_deduplicated(self) -> None:
        c = Classifier(labels=["a", "b", "a", "c"])
        assert c.labels == ["a", "b", "c"]

    def test_bad_example_label_raises(self) -> None:
        with pytest.raises(ValueError, match="not in taxonomy"):
            Classifier(
                labels=["a", "b"],
                examples=[ClassifyExample(text="hello", label="unknown")],
            )

    def test_valid_few_shot_construction(self) -> None:
        c = Classifier(
            labels=["sports", "politics"],
            examples=[ClassifyExample(text="The game was won.", label="sports")],
        )
        assert len(c.examples) == 1

    def test_multi_label_flag(self) -> None:
        c = Classifier(labels=["a", "b"], multi_label=True)
        assert c.multi_label is True

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            Classifier(labels=["a"], temperature=3.0)

    def test_invalid_min_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            Classifier(labels=["a"], min_confidence=1.5)

    def test_valid_temperature_and_min_confidence(self) -> None:
        c = Classifier(labels=["a", "b"], temperature=0.7, min_confidence=0.5)
        assert c.temperature == pytest.approx(0.7)
        assert c.min_confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_json_unchanged(self) -> None:
        raw = '{"label": "positive"}'
        assert _extract_json(raw) == raw

    def test_strips_json_fence(self) -> None:
        raw = '```json\n{"label": "positive"}\n```'
        assert _extract_json(raw) == '{"label": "positive"}'

    def test_strips_plain_fence(self) -> None:
        raw = '```\n{"label": "negative"}\n```'
        assert _extract_json(raw) == '{"label": "negative"}'

    def test_whitespace_trimmed(self) -> None:
        raw = '   {"label": "neutral"}   '
        assert _extract_json(raw) == '{"label": "neutral"}'

    def test_empty_fence_returns_empty_string(self) -> None:
        """Empty fence ```\n``` should not raise (M7)."""
        result = _extract_json("```\n```")
        assert result == ""


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

class TestPromptHelpers:
    def test_fmt_labels_bullets(self) -> None:
        result = _fmt_labels(["a", "b"])
        assert "  - a" in result
        assert "  - b" in result

    def test_fmt_examples_includes_text_and_label(self) -> None:
        exs = [ClassifyExample(text="hello world", label="greet")]
        result = _fmt_examples(exs)
        assert "hello world" in result
        assert "greet" in result

    def test_fmt_examples_escapes_double_quotes(self) -> None:
        """M1: quotes in example text must be backslash-escaped."""
        exs = [ClassifyExample(text='Say "hello" world', label="greet")]
        result = _fmt_examples(exs)
        assert '\\"hello\\"' in result
        assert "greet" in result

    def test_fmt_examples_escapes_backslashes(self) -> None:
        exs = [ClassifyExample(text="path\\to\\file", label="path")]
        result = _fmt_examples(exs)
        assert "path\\\\to\\\\file" in result


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_zero_shot_single_label(self) -> None:
        c = _clf()
        msgs = c._build_messages("test text")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # Zero-shot single: no examples block
        assert "Examples:" not in msgs[1]["content"]
        # Labels present
        assert "positive" in msgs[1]["content"]

    def test_few_shot_single_label(self) -> None:
        c = _clf(examples=[ClassifyExample(text="great!", label="positive")])
        msgs = c._build_messages("test text")
        assert "Examples:" in msgs[1]["content"]
        assert "great!" in msgs[1]["content"]

    def test_zero_shot_multi_label(self) -> None:
        c = _clf(multi_label=True)
        msgs = c._build_messages("test text")
        # Multi-label prompt expects list response
        assert '"labels"' in msgs[1]["content"]
        assert "Examples:" not in msgs[1]["content"]

    def test_few_shot_multi_label(self) -> None:
        c = _clf(
            multi_label=True,
            examples=[ClassifyExample(text="great!", label="positive")],
        )
        msgs = c._build_messages("test text")
        assert '"labels"' in msgs[1]["content"]
        assert "Examples:" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Edge case input validation (I1)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_classify_empty_text_raises(self) -> None:
        c = _clf()
        with pytest.raises(ValueError, match="empty"):
            c.classify("")

    def test_classify_whitespace_text_raises(self) -> None:
        c = _clf()
        with pytest.raises(ValueError, match="empty"):
            c.classify("   ")

    @pytest.mark.asyncio
    async def test_aclassify_empty_text_raises(self) -> None:
        c = _clf()
        with pytest.raises(ValueError, match="empty"):
            await c.aclassify("")


# ---------------------------------------------------------------------------
# Confidence clamping (C1)
# ---------------------------------------------------------------------------

class TestConfidenceClamping:
    def test_confidence_above_one_clamped(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "positive", "confidence": 1.5, "rationale": ""}')
        assert result.label == "positive"
        assert result.scores[0].confidence == pytest.approx(1.0)

    def test_confidence_below_zero_clamped(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "positive", "confidence": -0.3, "rationale": ""}')
        assert result.label == "positive"
        assert result.scores[0].confidence == pytest.approx(0.0)

    def test_multi_label_confidence_clamped(self) -> None:
        c = _clf(multi_label=True)
        json_str = '{"labels": [{"label": "positive", "confidence": 2.0}], "rationale": ""}'
        result = c._parse_response(json_str)
        assert result.scores[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# min_confidence threshold (M6)
# ---------------------------------------------------------------------------

class TestMinConfidence:
    def test_low_confidence_returns_none(self) -> None:
        c = _clf(min_confidence=0.8)
        result = c._parse_response('{"label": "positive", "confidence": 0.5, "rationale": ""}')
        assert result.label is None

    def test_exact_threshold_passes(self) -> None:
        c = _clf(min_confidence=0.8)
        result = c._parse_response('{"label": "positive", "confidence": 0.8, "rationale": ""}')
        assert result.label == "positive"

    def test_multi_label_filters_below_threshold(self) -> None:
        c = _clf(multi_label=True, min_confidence=0.7)
        json_str = '{"labels": [{"label": "positive", "confidence": 0.9}, {"label": "neutral", "confidence": 0.4}], "rationale": ""}'
        result = c._parse_response(json_str)
        assert len(result.scores) == 1
        assert result.scores[0].label == "positive"


# ---------------------------------------------------------------------------
# Case-insensitive label matching (I6 / I7)
# ---------------------------------------------------------------------------

class TestCaseInsensitiveLabels:
    def test_case_mismatch_matched_via_fallback(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "Positive", "confidence": 0.9, "rationale": ""}')
        assert result.label == "positive"  # case-insensitive fallback

    def test_all_caps_matched(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "NEUTRAL", "confidence": 0.7, "rationale": ""}')
        assert result.label == "neutral"

    def test_completely_unknown_label_returns_none(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "amazing", "confidence": 0.9, "rationale": ""}')
        assert result.label is None  # truly unknown — no case-insensitive match

    def test_multi_label_case_mismatch_resolved(self) -> None:
        c = _clf(multi_label=True)
        json_str = '{"labels": [{"label": "POSITIVE", "confidence": 0.9}], "rationale": ""}'
        result = c._parse_response(json_str)
        assert result.label == "positive"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_single_label_known(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "positive", "confidence": 0.9, "rationale": "happy"}')
        assert result.label == "positive"
        assert result.scores[0].confidence == pytest.approx(0.9)
        assert result.rationale == "happy"

    def test_single_label_unknown_returns_none(self) -> None:
        c = _clf()
        result = c._parse_response('{"label": "amazing", "confidence": 0.95, "rationale": ""}')
        assert result.label is None
        assert result.scores == []

    def test_invalid_json_returns_none(self) -> None:
        c = _clf()
        result = c._parse_response("NOT JSON")
        assert result.label is None

    def test_fenced_json_parsed(self) -> None:
        c = _clf()
        result = c._parse_response(
            '```json\n{"label": "neutral", "confidence": 0.7, "rationale": "balanced"}\n```'
        )
        assert result.label == "neutral"

    def test_multi_label_parsed(self) -> None:
        c = _clf(multi_label=True)
        json_str = '{"labels": [{"label": "positive", "confidence": 0.9}, {"label": "neutral", "confidence": 0.4}], "rationale": "mixed"}'
        result = c._parse_response(json_str)
        assert result.label == "positive"  # highest confidence
        assert len(result.scores) == 2
        assert result.scores[0].label == "positive"
        assert result.rationale == "mixed"

    def test_multi_label_unknown_filtered(self) -> None:
        c = _clf(multi_label=True)
        json_str = '{"labels": [{"label": "positive", "confidence": 0.9}, {"label": "unknown_label", "confidence": 0.8}], "rationale": ""}'
        result = c._parse_response(json_str)
        labels = [s.label for s in result.scores]
        assert "unknown_label" not in labels
        assert "positive" in labels

    def test_multi_label_sorted_by_confidence(self) -> None:
        c = _clf(multi_label=True)
        json_str = '{"labels": [{"label": "neutral", "confidence": 0.3}, {"label": "positive", "confidence": 0.9}], "rationale": ""}'
        result = c._parse_response(json_str)
        assert result.scores[0].label == "positive"

    def test_multi_label_empty_list_returns_none(self) -> None:
        c = _clf(multi_label=True)
        result = c._parse_response('{"labels": [], "rationale": "unclear"}')
        assert result.label is None


# ---------------------------------------------------------------------------
# classify() — mocked LLM
# ---------------------------------------------------------------------------

class TestClassify:
    def test_classify_positive(self) -> None:
        c = _clf()
        resp_json = '{"label": "positive", "confidence": 0.92, "rationale": "enthusiastic"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp_json)):
            result = c.classify("I absolutely love this!")
        assert isinstance(result, ClassifyResponse)
        assert result.label == "positive"
        assert result.scores[0].confidence == pytest.approx(0.92)
        assert "enthusiastic" in result.rationale

    def test_classify_negative(self) -> None:
        c = _clf()
        resp_json = '{"label": "negative", "confidence": 0.88, "rationale": "complaint"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp_json)):
            result = c.classify("This is terrible.")
        assert result.label == "negative"

    def test_classify_tracks_tokens(self) -> None:
        c = _clf()
        resp_json = '{"label": "neutral", "confidence": 0.75, "rationale": ""}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp_json)):
            result = c.classify("It is what it is.")
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_classify_bad_json_returns_none_label(self) -> None:
        c = _clf()
        with patch.object(c, "_call_llm", return_value=_fake_llm("oops not json")):
            result = c.classify("some text")
        assert result.label is None


# ---------------------------------------------------------------------------
# aclassify() — mocked async LLM
# ---------------------------------------------------------------------------

class TestAClassify:
    @pytest.mark.asyncio
    async def test_aclassify_returns_classify_response(self) -> None:
        c = _clf()
        fake = _fake_llm('{"label": "positive", "confidence": 0.95, "rationale": "great"}')
        with patch.object(c, "_acall_llm", AsyncMock(return_value=fake)):
            result = await c.aclassify("Wonderful day!")
        assert result.label == "positive"

    @pytest.mark.asyncio
    async def test_aclassify_multi_label(self) -> None:
        c = _clf(labels=["sports", "finance"], multi_label=True)
        fake = _fake_llm(
            '{"labels": [{"label": "sports", "confidence": 0.9}, {"label": "finance", "confidence": 0.7}], "rationale": "both"}'
        )
        with patch.object(c, "_acall_llm", AsyncMock(return_value=fake)):
            result = await c.aclassify("Nike stock surged after the championship.")
        assert result.label == "sports"
        assert len(result.scores) == 2


# ---------------------------------------------------------------------------
# Pipeline compatibility — run() / arun()
# ---------------------------------------------------------------------------

class TestPipelineCompatibility:
    def test_run_returns_classify_response(self) -> None:
        c = _clf()
        resp_json = '{"label": "neutral", "confidence": 0.8, "rationale": ""}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp_json)):
            result = c.run("Some neutral text")
        assert isinstance(result, ClassifyResponse)
        assert result.label == "neutral"

    @pytest.mark.asyncio
    async def test_arun_returns_classify_response(self) -> None:
        c = _clf()
        fake = _fake_llm('{"label": "negative", "confidence": 0.85, "rationale": ""}')
        with patch.object(c, "_acall_llm", AsyncMock(return_value=fake)):
            result = await c.arun("Terrible experience.")
        assert result.label == "negative"

    def test_to_text_returns_label(self) -> None:
        assert ClassifyResponse(label="positive").to_text() == "positive"

    def test_to_text_unknown_when_no_label(self) -> None:
        assert ClassifyResponse(label=None).to_text() == "unknown"

    @pytest.mark.asyncio
    async def test_sync_async_parity(self) -> None:
        """M10: sync classify() and async aclassify() must produce identical results."""
        c = _clf()
        resp_json = '{"label": "positive", "confidence": 0.9, "rationale": "great tone"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp_json)):
            sync_result = c.classify("I love this!")
        with patch.object(c, "_acall_llm", AsyncMock(return_value=_fake_llm(resp_json))):
            async_result = await c.aclassify("I love this!")
        assert sync_result.label == async_result.label
        assert sync_result.rationale == async_result.rationale
        assert sync_result.scores[0].confidence == async_result.scores[0].confidence


# ---------------------------------------------------------------------------
# run_batch()
# ---------------------------------------------------------------------------

class TestRunBatch:
    def test_run_batch_classifies_all(self) -> None:
        c = _clf()
        resp_pos = '{"label": "positive", "confidence": 0.9, "rationale": ""}'
        resp_neg = '{"label": "negative", "confidence": 0.85, "rationale": ""}'
        resp_neu = '{"label": "neutral", "confidence": 0.7, "rationale": ""}'
        llm_responses = [_fake_llm(r) for r in [resp_pos, resp_neg, resp_neu]]
        with patch.object(c, "_acall_llm", AsyncMock(side_effect=llm_responses)):
            batch = c.run_batch(["I love it!", "I hate it.", "It is fine."])
        assert len(batch.results) == 3
        assert batch.results[0].label == "positive"
        assert batch.results[1].label == "negative"
        assert batch.results[2].label == "neutral"

    def test_run_batch_empty(self) -> None:
        c = _clf()
        batch = c.run_batch([])
        assert batch.results == []


# ---------------------------------------------------------------------------
# Server route
# ---------------------------------------------------------------------------

class TestClassifyServerRoute:
    @pytest.fixture()
    def client(self):  # type: ignore[no-untyped-def]
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from aiforge.server.routes.classify import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_classify_returns_200_with_label(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch(
            "aiforge.modules.classify.module.Classifier.aclassify",
            AsyncMock(
                return_value=ClassifyResponse(
                    label="positive",
                    scores=[LabelScore(label="positive", confidence=0.9)],
                    rationale="enthusiastic",
                )
            ),
        ):
            resp = client.post(
                "/classify",
                json={"text": "I love this!", "labels": ["positive", "negative", "neutral"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "positive"

    def test_empty_labels_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post("/classify", json={"text": "hello", "labels": []})
        assert resp.status_code == 422

    def test_missing_labels_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post("/classify", json={"text": "hello"})
        assert resp.status_code == 422

    def test_too_many_labels_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        resp = client.post(
            "/classify",
            json={"text": "hello", "labels": [str(i) for i in range(101)]},
        )
        assert resp.status_code == 422

    def test_example_label_not_in_taxonomy_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        """I5/I6: example label outside label list must return 422, not 500."""
        resp = client.post(
            "/classify",
            json={
                "text": "hello",
                "labels": ["a", "b"],
                "examples": [{"text": "foo", "label": "c"}],
            },
        )
        assert resp.status_code == 422

    def test_few_shot_examples_accepted(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch(
            "aiforge.modules.classify.module.Classifier.aclassify",
            AsyncMock(
                return_value=ClassifyResponse(
                    label="sports",
                    scores=[LabelScore(label="sports", confidence=0.95)],
                    rationale="game mention",
                )
            ),
        ):
            resp = client.post(
                "/classify",
                json={
                    "text": "The team won the championship.",
                    "labels": ["sports", "politics"],
                    "examples": [{"text": "The match was thrilling.", "label": "sports"}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["label"] == "sports"

    def test_multi_label_flag_accepted(self, client) -> None:  # type: ignore[no-untyped-def]
        with patch(
            "aiforge.modules.classify.module.Classifier.aclassify",
            AsyncMock(
                return_value=ClassifyResponse(
                    label="sports",
                    scores=[
                        LabelScore(label="sports", confidence=0.9),
                        LabelScore(label="finance", confidence=0.6),
                    ],
                    rationale="both apply",
                )
            ),
        ):
            resp = client.post(
                "/classify",
                json={
                    "text": "Nike stock surged after the championship.",
                    "labels": ["sports", "finance", "politics"],
                    "multi_label": True,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["scores"]) == 2
