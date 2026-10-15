"""Unit tests for aiforge.evals: datasets, scorers, runner, reports and the CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from rich.console import Console
from typer.testing import CliRunner

from aiforge.cli.commands_eval import eval_app
from aiforge.core.provider import LLMProvider
from aiforge.core.types import CompletionResult, ProviderError, RateLimitError
from aiforge.evals import (
    CaseResult,
    Dataset,
    EvalCase,
    EvalReport,
    JudgeVerdict,
    ScoreResult,
    aevaluate,
    assert_eval,
    contains,
    embedding_similarity,
    evaluate,
    exact_match,
    get_scorer,
    json_match,
    label_match,
    llm_judge,
    not_contains,
    numeric_close,
    output_data,
    output_text,
    regex,
    resolve_target,
    scorer,
)
from aiforge.modules.classify.module import Classifier
from aiforge.modules.classify.schemas import ClassifyResponse, LabelScore
from aiforge.modules.extract.schemas import ExtractResponse
from aiforge.registry import register_module

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _rule_label(text: str) -> str:
    return "positive" if "love" in text.lower() else "negative"


class FakeClassifier(Classifier):
    """Classifier whose LLM call is replaced by a keyword rule."""

    def __init__(self, labels: list[str] | None = None, **_: Any) -> None:
        self.labels = labels or ["positive", "negative"]
        self.calls: list[str] = []

    def classify(self, text: str, *, system_prompt: str | None = None) -> ClassifyResponse:
        return self._respond(text)

    async def aclassify(self, text: str, *, system_prompt: str | None = None) -> ClassifyResponse:
        if text == "boom":
            raise RuntimeError("provider exploded")
        return self._respond(text)

    def _respond(self, text: str) -> ClassifyResponse:
        self.calls.append(text)
        label = _rule_label(text)
        return ClassifyResponse(
            label=label,
            scores=[LabelScore(label=label, confidence=0.9)],
            cost=0.001,
            input_tokens=10,
            output_tokens=2,
            model="fake",
        )


def _judge_result(text: str, parsed: dict[str, Any] | None = None) -> CompletionResult:
    return CompletionResult(
        text=text, input_tokens=100, output_tokens=20, model="gpt-4o-mini", parsed=parsed
    )


class FakeJudgeLLM(LLMProvider):
    """Records calls; replies from a queue of results or exceptions."""

    def __init__(self, replies: list[CompletionResult | Exception], structured: bool = True):
        self.supports_structured_output = structured
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        raise NotImplementedError

    async def complete_async(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> CompletionResult:
        self.calls.append({"messages": messages, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncGenerator[str]:
        yield ""  # pragma: no cover

    @property
    def model_name(self) -> str:
        return "fake-judge"


class FakeEmbedder:
    """Maps known texts to fixed vectors."""

    vectors = {
        "cat": [1.0, 0.0],
        "kitten": [0.9, 0.1],
        "car": [0.0, 1.0],
    }

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0]) for t in texts]


def _case(expected: Any = None, input: Any = "x", **kw: Any) -> EvalCase:
    return EvalCase(id="c", input=input, expected=expected, **kw)


async def _score(s: Any, output: Any, expected: Any = None, **kw: Any) -> ScoreResult:
    result: ScoreResult = await s.score(output, _case(expected, **kw))
    return result


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


class TestDataset:
    def test_from_list_assigns_index_ids(self) -> None:
        ds = Dataset.from_list([{"input": "a"}, {"input": "b", "id": "named"}])
        assert [c.id for c in ds] == ["0", "named"]
        assert len(ds) == 2
        assert ds[0].expected is None

    def test_rejects_missing_input_unknown_fields_and_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="no 'input'"):
            Dataset.from_list([{"expected": "a"}])
        with pytest.raises(ValueError, match="unknown fields"):
            Dataset.from_list([{"input": "a", "expect": "typo"}])
        with pytest.raises(ValueError, match="Duplicate"):
            Dataset.from_list([{"input": "a", "id": 1}, {"input": "b", "id": "1"}])

    def test_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "cases.jsonl"
        p.write_text(
            '{"input": "a", "expected": "x", "kwargs": {"k": 1}}\n\n{"input": "b"}\n',
            encoding="utf-8",
        )
        ds = Dataset.load(p)
        assert ds.name == "cases"
        assert [c.input for c in ds] == ["a", "b"]
        assert ds[0].kwargs == {"k": 1}

    def test_jsonl_reports_bad_line(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text('{"input": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            Dataset.load(p)

    def test_json_and_yaml_with_cases_key(self, tmp_path: Path) -> None:
        j = tmp_path / "d.json"
        j.write_text(json.dumps({"cases": [{"input": "a", "expected": [1, 2]}]}), encoding="utf-8")
        assert Dataset.load(j)[0].expected == [1, 2]

        y = tmp_path / "d.yaml"
        y.write_text(
            "cases:\n  - id: 7\n    input: hello\n    expected: positive\n"
            "    metadata: {tag: smoke}\n",
            encoding="utf-8",
        )
        case = Dataset.load(y)[0]
        assert case.id == "7"
        assert case.metadata == {"tag": "smoke"}

        bare = tmp_path / "list.yml"
        bare.write_text("- input: a\n- input: b\n", encoding="utf-8")
        assert len(Dataset.load(bare)) == 2

    def test_yaml_must_be_list(self, tmp_path: Path) -> None:
        y = tmp_path / "d.yaml"
        y.write_text("foo: bar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="list of cases"):
            Dataset.load(y)

    def test_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "d.csv"
        p.write_text(
            "id,input,expected,kwargs,metadata,lang\n"
            'a,hello,positive,"{""k"": 1}","{""tag"": ""x""}",en\n'
            'b,multi,"[""a"", ""b""]",,,fr\n'
            "c,no expectation,,,,de\n",
            encoding="utf-8",
        )
        ds = Dataset.load(p)
        assert ds[0].id == "a"
        assert ds[0].expected == "positive"
        assert ds[0].kwargs == {"k": 1}
        assert ds[0].metadata == {"tag": "x", "lang": "en"}
        assert ds[1].expected == ["a", "b"]
        assert ds[2].expected is None

    def test_csv_requires_input_column(self, tmp_path: Path) -> None:
        p = tmp_path / "d.csv"
        p.write_text("text,expected\nhi,x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'input' column"):
            Dataset.load(p)

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        p = tmp_path / "d.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            Dataset.load(p)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    name: str
    age: int


def test_output_text_and_data_views() -> None:
    resp = ClassifyResponse(label="bug", cost=0.1)
    assert output_text(resp) == "bug"
    assert output_text(None) == ""
    assert output_text({"a": 1}) == '{"a": 1}'
    assert output_data(ExtractResponse(data={"x": 1})) == {"x": 1}
    assert output_data(_Person(name="A", age=3)) == {"name": "A", "age": 3}
    assert output_data('{"y": 2}') == {"y": 2}
    assert output_data("plain") is None
    dumped = output_data(resp)
    assert dumped is not None and "cost" not in dumped and dumped["label"] == "bug"


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


class TestScorers:
    async def test_exact_match_normalisation(self) -> None:
        assert (await _score(exact_match(), "  Hello   world ", "Hello world")).passed
        assert not (await _score(exact_match(), "hello world", "Hello world")).passed
        assert (await _score(exact_match(ignore_case=True), "hello", "HELLO")).passed
        strict = exact_match(normalize_whitespace=False)
        assert not (await _score(strict, "a  b", "a b")).passed
        assert (await _score(exact_match(), "b", ["a", "b"])).passed
        assert (await _score(exact_match("fixed"), "fixed", "ignored")).passed

    async def test_missing_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="no expected"):
            await _score(exact_match(), "x", None)

    async def test_contains_and_not_contains(self) -> None:
        r = await _score(contains(["refund", "sorry"]), "We will refund you", None)
        assert not r.passed and r.score == 0.5 and "sorry" in r.reason
        assert (
            await _score(contains(["refund", "sorry"], require_all=False), "refund", None)
        ).passed
        assert (await _score(contains(ignore_case=True), "REFUND now", "refund")).passed
        r = await _score(not_contains(["password", "ssn"]), "your password is", None)
        assert not r.passed and r.score == 0.5
        assert (await _score(not_contains("secret"), "all clear", None)).passed

    async def test_regex(self) -> None:
        assert (await _score(regex(r"\d{3}-\d{4}"), "call 555-1234", None)).passed
        assert not (await _score(regex(r"\d+", full_match=True), "abc 12", None)).passed
        assert (await _score(regex(), "ORD-42", r"ORD-\d+")).passed

    async def test_json_match_subset_strict_and_fields(self) -> None:
        out = {"name": "Ada", "age": 36, "city": "London"}
        assert (await _score(json_match(), out, {"name": "Ada", "age": 36.0})).passed
        strict = await _score(json_match(subset=False), out, {"name": "Ada", "age": 36})
        assert not strict.passed and strict.score == pytest.approx(2 / 3)
        partial = await _score(json_match(), out, {"name": "Ada", "age": 40})
        assert not partial.passed and partial.score == 0.5 and "age" in partial.reason
        only_name = json_match(fields=["name"])
        assert (await _score(only_name, out, {"name": "Ada", "age": 40})).passed
        assert (await _score(json_match(ignore_case=True), out, {"city": "london"})).passed

    async def test_json_match_models_nested_and_unstructured(self) -> None:
        nested = {"user": {"name": "A", "tags": ["x", "y"]}, "n": 1}
        assert (
            await _score(json_match(), ExtractResponse(data=nested), {"user": {"name": "A"}})
        ).passed
        assert not (
            await _score(json_match(), nested, {"user": {"tags": ["x"]}})
        ).passed  # list lengths differ
        assert (
            await _score(json_match(), _Person(name="B", age=1), _Person(name="B", age=1))
        ).passed
        r = await _score(json_match(), "not json", {"a": 1})
        assert not r.passed and "not structured" in r.reason

    async def test_numeric_close(self) -> None:
        assert (await _score(numeric_close(tolerance=0.01), "The total is 3.141", 3.14)).passed
        assert not (await _score(numeric_close(tolerance=0.0001), 3.141, "3.14")).passed
        assert (await _score(numeric_close(tolerance=0.05, relative=True), 104, 100)).passed
        assert (await _score(numeric_close(field="total"), {"total": "1,000"}, 1000)).passed
        r = await _score(numeric_close(field="total"), {"sum": 1}, 1)
        assert not r.passed and "missing" in r.reason
        assert not (await _score(numeric_close(), "no digits", 1)).passed

    async def test_label_match(self) -> None:
        resp = ClassifyResponse(label="Bug", scores=[LabelScore(label="Bug", confidence=0.8)])
        assert (await _score(label_match(), resp, "bug")).passed
        assert not (await _score(label_match(ignore_case=False), resp, "bug")).passed
        multi = ClassifyResponse(label="a", scores=[LabelScore(label="a"), LabelScore(label="b")])
        r = await _score(label_match(), multi, ["a", "c"])
        assert not r.passed and r.score == pytest.approx(1 / 3)
        assert (await _score(label_match(), multi, ["b", "a"])).passed
        assert (await _score(label_match(), "spam", "spam")).passed
        assert (await _score(label_match(), {"label": "x"}, "x")).passed

    async def test_embedding_similarity(self) -> None:
        s = embedding_similarity(threshold=0.8, embedder=FakeEmbedder())  # type: ignore[arg-type]
        close = await _score(s, "kitten", "cat")
        assert close.passed and close.score > 0.9
        far = await _score(s, "car", "cat")
        assert not far.passed and far.score == 0.0

    def test_embedding_similarity_resolves_configured_provider(self) -> None:
        s = embedding_similarity(provider="openai", model="text-embedding-3-large")
        with patch("aiforge.providers.get_embedding_provider") as get:
            get.return_value = FakeEmbedder()
            assert s.embedder is get.return_value  # type: ignore[attr-defined]
        settings = get.call_args.args[0]
        assert settings.EMBEDDING_MODEL == "text-embedding-3-large"

    async def test_custom_function_scorers(self) -> None:
        def is_short(output: Any, case: EvalCase) -> bool:
            return len(output) < 5

        @scorer(name="ratio", threshold=0.7)
        def ratio(output: Any, case: EvalCase) -> float:
            return 0.6

        @scorer
        async def tupled(output: Any, case: EvalCase) -> tuple[float, str]:
            return 1.0, "fine"

        @scorer
        def as_dict(output: Any, case: EvalCase) -> dict[str, Any]:
            return {"passed": False, "reason": "nope"}

        report = await aevaluate(
            lambda x: x, [{"input": "abc"}], [is_short, ratio, tupled, as_dict]
        )
        scores = {s.name: s for s in report.cases[0].scores}
        assert scores["is_short"].passed
        assert scores["ratio"].score == 0.6 and not scores["ratio"].passed
        assert scores["tupled"].reason == "fine"
        assert scores["as_dict"].score == 0.0 and scores["as_dict"].reason == "nope"

    async def test_bad_custom_return_is_recorded_as_scorer_error(self) -> None:
        report = await aevaluate(lambda x: x, [{"input": "a"}], [lambda o, c: "yes"])
        result = report.cases[0].scores[0]
        assert not result.passed and "scorer error: TypeError" in result.reason

    def test_get_scorer(self) -> None:
        s = get_scorer("contains", value="x", name="has_x")
        assert s.name == "has_x"
        with pytest.raises(KeyError, match="Unknown scorer"):
            get_scorer("nope")


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    async def test_structured_path(self) -> None:
        verdict = {"score": 0.9, "passed": True, "reason": "polite and correct"}
        llm = FakeJudgeLLM([_judge_result(json.dumps(verdict), parsed=verdict)])
        s = llm_judge("Is it polite?", llm=llm)
        r = await _score(s, "Thanks!", "a thank-you", input="say thanks")
        assert r.passed and r.score == 0.9 and r.reason == "polite and correct"
        assert r.cost > 0
        assert len(llm.calls) == 1
        assert llm.calls[0]["response_format"] is JudgeVerdict
        prompt = llm.calls[0]["messages"][1]["content"]
        assert "Is it polite?" in prompt and "say thanks" in prompt and "Thanks!" in prompt
        assert "a thank-you" in prompt

    async def test_threshold_overrides_judge_passed(self) -> None:
        verdict = {"score": 0.6, "passed": True, "reason": "ok"}
        llm = FakeJudgeLLM([_judge_result("", parsed=verdict)])
        r = await _score(llm_judge("r", llm=llm, threshold=0.7), "out")
        assert not r.passed and r.score == 0.6

    async def test_score_is_clamped(self) -> None:
        verdict = {"score": 7, "passed": True, "reason": "great"}
        llm = FakeJudgeLLM([_judge_result("", parsed=verdict)])
        assert (await _score(llm_judge("r", llm=llm), "out")).score == 1.0

    async def test_plain_path_without_structured_support(self) -> None:
        text = 'Verdict: {"score": 0.2, "passed": false, "reason": "rude"}'
        llm = FakeJudgeLLM([_judge_result(text)], structured=False)
        r = await _score(llm_judge("polite?", llm=llm), "go away")
        assert not r.passed and r.score == 0.2 and r.reason == "rude"
        assert "response_format" not in llm.calls[0]

    async def test_falls_back_when_structured_call_fails(self) -> None:
        text = '{"score": 1, "passed": true, "reason": "fine"}'
        llm = FakeJudgeLLM([ProviderError("schema not supported"), _judge_result(text)])
        r = await _score(llm_judge("r", llm=llm), "out")
        assert r.passed and r.score == 1.0
        assert "response_format" in llm.calls[0] and "response_format" not in llm.calls[1]

    async def test_falls_back_when_structured_reply_unparsed(self) -> None:
        good = '{"score": 0.8, "passed": true, "reason": "ok"}'
        llm = FakeJudgeLLM([_judge_result("I think it's fine"), _judge_result(good)])
        r = await _score(llm_judge("r", llm=llm), "out")
        assert r.passed and len(llm.calls) == 2
        assert r.cost == pytest.approx(2 * _judge_result("").cost)

    async def test_unparseable_verdict_scores_zero(self) -> None:
        llm = FakeJudgeLLM([_judge_result("no idea")], structured=False)
        r = await _score(llm_judge("r", llm=llm), "out")
        assert not r.passed and r.score == 0.0 and "no parseable verdict" in r.reason

    async def test_transient_error_is_not_swallowed(self) -> None:
        llm = FakeJudgeLLM([RateLimitError("slow down")])
        report = await aevaluate(lambda x: x, [{"input": "a"}], [llm_judge("r", llm=llm)])
        result = report.cases[0].scores[0]
        assert not result.passed and "RateLimitError" in result.reason
        assert len(llm.calls) == 1

    def test_judge_provider_is_configurable(self) -> None:
        s = llm_judge("r", provider="anthropic", model="claude-haiku-4-5")
        with patch("aiforge.providers.get_provider") as get:
            get.return_value = FakeJudgeLLM([])
            assert s.llm is get.return_value  # type: ignore[attr-defined]
        settings = get.call_args.args[0]
        assert settings.PROVIDER == "anthropic"
        assert settings.MODEL == "claude-haiku-4-5"

    def test_empty_rubric_rejected(self) -> None:
        with pytest.raises(ValueError, match="rubric"):
            llm_judge("  ")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SENTIMENT_CASES = [
    {"id": "p1", "input": "I love it", "expected": "positive"},
    {"id": "p2", "input": "love love", "expected": "positive"},
    {"id": "n1", "input": "terrible", "expected": "negative"},
    {"id": "n2", "input": "meh, I love nothing", "expected": "negative"},  # rule gets it wrong
]


class TestRunner:
    def test_module_maps_to_main_method(self) -> None:
        clf = FakeClassifier()
        call, name = resolve_target(clf)
        assert name == "FakeClassifier.aclassify"
        _, name = resolve_target(clf, method="classify")
        assert name == "FakeClassifier.aclassify"  # async twin preferred
        _, name = resolve_target(clf, method="_respond")
        assert name == "FakeClassifier._respond"
        with pytest.raises(AttributeError):
            resolve_target(clf, method="missing")
        with pytest.raises(TypeError):
            resolve_target(42)

    def test_evaluate_module_captures_usage(self) -> None:
        clf = FakeClassifier()
        report = evaluate(clf, _SENTIMENT_CASES, [label_match()], min_pass_rate=0.7)
        assert report.target == "FakeClassifier.aclassify"
        assert [c.passed for c in report.cases] == [True, True, True, False]
        assert report.cases[0].output["label"] == "positive"
        assert report.total_cost == pytest.approx(0.004)
        assert report.total_input_tokens == 40 and report.total_output_tokens == 8
        assert all(c.latency_ms >= 0 for c in report.cases)
        assert report.passed  # 75% >= 70%

    async def test_sync_and_async_callables_with_kwargs(self) -> None:
        def sync_fn(text: str, *, suffix: str = "", prefix: str = "") -> str:
            return prefix + text + suffix

        async def async_fn(text: str, **kw: Any) -> str:
            return text.upper()

        cases = [{"input": "a", "expected": ">a!", "kwargs": {"suffix": "!"}}]
        report = await aevaluate(sync_fn, cases, [exact_match()], call_kwargs={"prefix": ">"})
        assert report.passed
        report = await aevaluate(async_fn, [{"input": "a", "expected": "A"}], [exact_match()])
        assert report.passed

    async def test_target_errors_are_captured(self) -> None:
        cases = [{"input": "I love it", "expected": "positive"}, {"input": "boom", "expected": "x"}]
        report = await aevaluate(FakeClassifier(), cases, [label_match(), contains("o")])
        bad = report.cases[1]
        assert bad.error == "RuntimeError: provider exploded"
        assert bad.output is None
        assert all(not s.passed and "target error" in s.reason for s in bad.scores)
        assert report.error_count == 1
        assert report.scorer_stats()["label_match"].pass_rate == 0.5

    async def test_timeout(self) -> None:
        async def slow(text: str) -> str:
            await asyncio.sleep(1)
            return text

        report = await aevaluate(slow, [{"input": "a"}], timeout=0.01)
        assert report.cases[0].error is not None
        assert report.cases[0].error.startswith("TimeoutError")
        assert not report.passed

    async def test_bounded_concurrency(self) -> None:
        in_flight = 0
        peak = 0

        async def target(text: str) -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return text

        seen: list[CaseResult] = []
        cases = [{"input": str(i)} for i in range(12)]
        report = await aevaluate(target, cases, concurrency=3, on_result=seen.append)
        assert peak == 3
        assert len(seen) == 12
        assert [c.id for c in report.cases] == [str(i) for i in range(12)]  # order kept

    async def test_validation(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            await aevaluate(str, [{"input": "a"}], [contains("a"), contains("b")])
        with pytest.raises(ValueError, match="unknown scorers"):
            await aevaluate(str, [{"input": "a"}], [contains("a")], thresholds={"x": 1})
        with pytest.raises(ValueError, match="concurrency"):
            await aevaluate(str, [{"input": "a"}], concurrency=0)

    async def test_sync_evaluate_inside_running_loop(self, tmp_path: Path) -> None:
        p = tmp_path / "d.jsonl"
        p.write_text('{"input": "a", "expected": "a"}\n', encoding="utf-8")
        report = evaluate(str, p, [exact_match()])
        assert report.passed and report.name == "d"


# ---------------------------------------------------------------------------
# Report math, serialisation and comparison
# ---------------------------------------------------------------------------


def _report(
    outcomes: dict[str, list[tuple[float, bool]]],
    latencies: list[float] | None = None,
    **kw: Any,
) -> EvalReport:
    """Build a report from {scorer: [(score, passed) per case]}."""
    n = len(next(iter(outcomes.values())))
    cases = []
    for i in range(n):
        cases.append(
            CaseResult(
                id=f"c{i}",
                input=f"in{i}",
                scores=[
                    ScoreResult(name=name, score=vals[i][0], passed=vals[i][1])
                    for name, vals in outcomes.items()
                ],
                cost=0.01,
                input_tokens=5,
                output_tokens=1,
                latency_ms=(latencies or [10.0] * n)[i],
            )
        )
    return EvalReport(name="t", scorer_names=list(outcomes), cases=cases, **kw)


class TestReport:
    def test_aggregates(self) -> None:
        r = _report(
            {
                "a": [(1.0, True), (0.5, False), (1.0, True), (0.0, False)],
                "b": [(1.0, True), (1.0, True), (1.0, True), (0.8, True)],
            },
            latencies=[10, 20, 30, 40],
        )
        stats = r.scorer_stats()
        assert stats["a"].mean == pytest.approx(0.625)
        assert stats["a"].pass_rate == 0.5 and stats["a"].passed_count == 2
        assert stats["b"].mean == pytest.approx(0.95) and stats["b"].pass_rate == 1.0
        assert r.pass_rate == 0.5
        assert r.total_cost == pytest.approx(0.04)
        assert r.total_tokens == 24
        lat = r.latency_percentiles()
        assert lat["p50"] == 25 and lat["max"] == 40 and lat["mean"] == 25
        assert lat["p90"] == pytest.approx(37)

    def test_thresholds(self) -> None:
        outcomes = {"a": [(1.0, True), (0.0, False)], "b": [(1.0, True), (1.0, True)]}
        assert not _report(outcomes).passed  # default min 1.0
        assert _report(outcomes, min_pass_rate=0.5).passed
        assert not _report(outcomes, min_pass_rate=0.5, thresholds={"a": 0.6}).passed
        assert _report(outcomes, thresholds={"a": 0.5}).passed

    def test_empty_report_does_not_pass(self) -> None:
        empty = EvalReport(scorer_names=["a"])
        assert not empty.passed and empty.pass_rate == 0.0
        assert empty.latency_percentiles()["p95"] == 0.0

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        r = _report({"a": [(1.0, True), (0.0, False)]}, thresholds={"a": 0.5})
        path = r.save(tmp_path / "out" / "report.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["pass_rate"] == 0.5
        assert data["summary"]["scorers"]["a"]["pass_rate"] == 0.5
        loaded = EvalReport.load(path)
        assert loaded.passed == r.passed is True
        assert loaded.scorer_stats() == r.scorer_stats()

    def test_compare_flags_regressions(self) -> None:
        base = _report({"a": [(1.0, True), (1.0, True)], "b": [(0.0, False), (1.0, True)]})
        cur = _report({"a": [(1.0, True), (0.0, False)], "b": [(1.0, True), (1.0, True)]})
        cmp = cur.compare(base)
        assert cmp.regressed
        assert [d.name for d in cmp.regressions] == ["a"]
        a = cmp.regressions[0]
        assert a.pass_rate_delta == -0.5 and a.mean_delta == -0.5
        assert cmp.newly_failing == ["c1"] and cmp.newly_passing == ["c0"]
        assert not cur.compare(base, tolerance=0.5).regressed

    def test_compare_mean_drop_and_added_removed(self) -> None:
        base = _report({"a": [(1.0, True)], "old": [(1.0, True)]})
        cur = _report({"a": [(0.9, True)], "new": [(1.0, True)]})
        cmp = cur.compare(base)
        statuses = {d.name: d.status for d in cmp.scorers}
        assert statuses == {"a": "compared", "new": "added", "old": "removed"}
        assert [d.name for d in cmp.regressions] == ["a"]
        assert not cur.compare(base, tolerance=0.1).regressed

    def test_print_renders_tables(self) -> None:
        r = _report({"a": [(1.0, True), (0.0, False)]})
        r.cases[1].scores[0].reason = "wrong label"
        con = Console(record=True, width=160)
        r.print(con)
        r.compare(_report({"a": [(1.0, True), (1.0, True)]})).print(con)
        text = con.export_text()
        assert "FAILED" in text and "wrong label" in text and "REGRESSED" in text

    def test_assert_eval(self, tmp_path: Path) -> None:
        good = _report({"a": [(1.0, True), (1.0, True)]})
        assert_eval(good)
        bad = _report({"a": [(1.0, True), (0.0, False)]})
        bad.cases[1].scores[0].reason = "nope"
        with pytest.raises(AssertionError, match=r"a: pass rate 50\.0% below") as exc:
            assert_eval(bad)
        assert "[c1] a: nope" in str(exc.value)

        baseline = good.save(tmp_path / "base.json")
        lenient = _report({"a": [(1.0, True), (0.0, False)]}, min_pass_rate=0.5)
        assert_eval(lenient)
        with pytest.raises(AssertionError, match="regressed"):
            assert_eval(lenient, baseline=baseline)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def sentiment_dataset(tmp_path: Path) -> Path:
    p = tmp_path / "sentiment.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in _SENTIMENT_CASES), encoding="utf-8")
    register_module("fakeclf", FakeClassifier)
    return p


def _run(*args: str) -> Any:
    return runner.invoke(eval_app, ["run", *args])


class TestCLI:
    def test_passes_and_writes_report(self, sentiment_dataset: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        result = _run(
            str(sentiment_dataset),
            "--module", "fakeclf",
            "--module-args", '{"labels": ["positive", "negative"]}',
            "--scorer", "label_match",
            "--min-pass-rate", "0.75",
            "--output", str(out),
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        report = EvalReport.load(out)
        assert report.target == "FakeClassifier.aclassify"
        assert report.pass_rate == 0.75

    def test_below_threshold_exits_1(self, sentiment_dataset: Path) -> None:
        result = _run(
            str(sentiment_dataset), "-m", "fakeclf", "-s", "label_match", "--min-pass-rate", "0.9"
        )
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_per_scorer_threshold(self, sentiment_dataset: Path) -> None:
        result = _run(
            str(sentiment_dataset),
            "-m", "fakeclf",
            "-s", "label_match",
            "-s", 'not_contains:{"value": "xyz"}',
            "--threshold", "label_match=0.75",
        )  # fmt: skip
        assert result.exit_code == 0, result.output

    def test_regression_exits_3(self, sentiment_dataset: Path, tmp_path: Path) -> None:
        perfect = _report({"label_match": [(1.0, True)] * 4})
        for i, c in enumerate(perfect.cases):
            c.id = _SENTIMENT_CASES[i]["id"]
        baseline = perfect.save(tmp_path / "base.json")
        args = [str(sentiment_dataset), "-m", "fakeclf", "-s", "label_match"]
        result = _run(*args, "--min-pass-rate", "0.5", "--baseline", str(baseline))
        assert result.exit_code == 3, result.output
        assert "REGRESSED" in result.output
        result = _run(
            *args, "--min-pass-rate", "0.5", "--baseline", str(baseline), "--tolerance", "0.3"
        )
        assert result.exit_code == 0, result.output

    def test_compare_command(self, tmp_path: Path) -> None:
        good = _report({"a": [(1.0, True), (1.0, True)]}).save(tmp_path / "good.json")
        bad = _report({"a": [(1.0, True), (0.0, False)]}).save(tmp_path / "bad.json")
        assert runner.invoke(eval_app, ["compare", str(bad), str(good)]).exit_code == 3
        assert runner.invoke(eval_app, ["compare", str(good), str(bad)]).exit_code == 0

    def test_callable_target(self, tmp_path: Path) -> None:
        p = tmp_path / "d.yaml"
        p.write_text("- input: abc\n  expected: abc\n", encoding="utf-8")
        result = _run(str(p), "--target", "builtins:str", "-s", "exact_match")
        assert result.exit_code == 0, result.output

    def test_judge_options_reach_scorer(self, sentiment_dataset: Path) -> None:
        from aiforge.cli.commands_eval import _parse_scorer

        s = _parse_scorer('llm_judge:{"rubric": "polite?"}', "anthropic", "claude-haiku-4-5")
        assert s._provider == "anthropic" and s._model == "claude-haiku-4-5"

    @pytest.mark.parametrize(
        "extra",
        [
            ["-m", "fakeclf", "--module-args", "not-json", "-s", "label_match"],
            ["-m", "nosuchmodule", "-s", "label_match"],
            ["-m", "fakeclf", "-s", "nosuchscorer"],
            ["-m", "fakeclf"],
            ["-m", "fakeclf", "--target", "builtins:str", "-s", "label_match"],
            ["-m", "fakeclf", "-s", "label_match", "--threshold", "label_match"],
            ["-m", "fakeclf", "-s", "label_match", "--threshold", "other=0.5"],
        ],
    )
    def test_usage_errors_exit_2(self, sentiment_dataset: Path, extra: list[str]) -> None:
        result = _run(str(sentiment_dataset), *extra)
        assert result.exit_code == 2, result.output
