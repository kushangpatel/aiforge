"""Scorers: turn a target's output for one case into a 0..1 score with pass/fail.

Every scorer implements :class:`Scorer`. Built-ins are created with the
factory functions below (``exact_match()``, ``llm_judge(rubric=...)`` …);
custom scorers are plain functions wrapped by :func:`scorer` or passed
directly to :func:`~aiforge.evals.evaluate`.

Scorers that compare against a reference take an optional explicit value and
otherwise use the case's ``expected``.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

import structlog
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from aiforge.evals.dataset import EvalCase

if TYPE_CHECKING:
    from aiforge.core.provider import EmbeddingProvider, LLMProvider
    from aiforge.core.types import CompletionResult
    from aiforge.infra.config import AIForgeSettings

logger = structlog.get_logger(__name__)

_MODULE_RESPONSE_FIELDS = {
    "cost",
    "input_tokens",
    "output_tokens",
    "model",
    "trace_id",
    "latency_ms",
}


class ScoreResult(BaseModel):
    """The verdict of one scorer on one case."""

    name: str = ""
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    cost: float = 0.0
    """USD spent by the scorer itself (e.g. an LLM judge call)."""


class Scorer(ABC):
    """Base class for scorers. Subclasses set :attr:`name` and implement :meth:`score`."""

    name: str = "scorer"

    @abstractmethod
    async def score(self, output: Any, case: EvalCase) -> ScoreResult:
        """Score *output* produced for *case*."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


ScorerFn = Callable[[Any, EvalCase], Any]
"""A custom scorer function: ``fn(output, case)`` returning a :class:`ScoreResult`,
``bool``, ``float`` in 0..1, ``(score, reason)`` tuple or ``{"score", "passed", "reason"}`` dict.
It may be ``async``."""


class FunctionScorer(Scorer):
    """Adapts a plain (sync or async) function to the :class:`Scorer` protocol."""

    def __init__(self, fn: ScorerFn, *, name: str | None = None, threshold: float = 0.5) -> None:
        self.fn = fn
        self.name = name or str(getattr(fn, "__name__", "custom"))
        self.threshold = threshold

    async def score(self, output: Any, case: EvalCase) -> ScoreResult:
        value = self.fn(output, case)
        if inspect.isawaitable(value):
            value = await value
        return _coerce_result(value, self.name, self.threshold)


@overload
def scorer(fn: ScorerFn, *, name: str | None = None, threshold: float = 0.5) -> Scorer: ...


@overload
def scorer(
    fn: None = None, *, name: str | None = None, threshold: float = 0.5
) -> Callable[[ScorerFn], Scorer]: ...


def scorer(
    fn: ScorerFn | None = None, *, name: str | None = None, threshold: float = 0.5
) -> Scorer | Callable[[ScorerFn], Scorer]:
    """Turn a function into a scorer; usable bare (``@scorer``) or with options.

    Example::

        @scorer(threshold=0.8)
        def short_enough(output, case):
            return 1.0 if len(output_text(output)) < 200 else 0.0
    """

    def wrap(f: ScorerFn) -> Scorer:
        return FunctionScorer(f, name=name, threshold=threshold)

    return wrap(fn) if fn is not None else wrap


def as_scorer(obj: Scorer | ScorerFn) -> Scorer:
    """Return *obj* unchanged if it is a :class:`Scorer`, else wrap the function."""
    if isinstance(obj, Scorer):
        return obj
    if callable(obj):
        return FunctionScorer(obj)
    raise TypeError(f"Not a scorer: {obj!r}")


def _coerce_result(value: Any, name: str, threshold: float) -> ScoreResult:
    if isinstance(value, ScoreResult):
        return value if value.name else value.model_copy(update={"name": name})
    reason = ""
    if isinstance(value, tuple) and len(value) == 2:
        value, reason = value[0], str(value[1])
    if isinstance(value, dict):
        score = float(value.get("score", 1.0 if value.get("passed") else 0.0))
        passed = bool(value.get("passed", score >= threshold))
        return ScoreResult(
            name=name,
            score=_clamp(score),
            passed=passed,
            reason=str(value.get("reason", reason)),
            cost=float(value.get("cost", 0.0)),
        )
    if isinstance(value, bool):
        return ScoreResult(name=name, score=1.0 if value else 0.0, passed=value, reason=reason)
    if isinstance(value, int | float):
        score = _clamp(float(value))
        return ScoreResult(name=name, score=score, passed=score >= threshold, reason=reason)
    raise TypeError(
        f"Scorer {name!r} returned {type(value).__name__}; "
        "expected ScoreResult, bool, float, (score, reason) or dict"
    )


def _clamp(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _result(name: str, passed: bool, reason: str, score: float | None = None) -> ScoreResult:
    return ScoreResult(
        name=name,
        score=_clamp(score if score is not None else (1.0 if passed else 0.0)),
        passed=passed,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def output_text(output: Any) -> str:
    """Best-effort text view of a target's output (``ModuleResponse.to_text()`` when available)."""
    from aiforge.core.types import ModuleResponse

    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, ModuleResponse):
        try:
            return output.to_text()
        except NotImplementedError:
            return json.dumps(output_data(output), default=str)
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, dict | list):
        return json.dumps(output, default=str)
    return str(output)


def output_data(output: Any) -> dict[str, Any] | None:
    """Structured view of an output: dicts as-is, ``ExtractResponse.data``, other models
    dumped without the usage fields, JSON strings parsed. ``None`` when not structured."""
    from aiforge.core.types import ModuleResponse

    if isinstance(output, dict):
        return output
    if isinstance(output, ModuleResponse):
        data = getattr(output, "data", None)
        if isinstance(data, dict):
            return data
        return output.model_dump(mode="json", exclude=_MODULE_RESPONSE_FIELDS)
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _reference(explicit: Any, case: EvalCase, scorer_name: str) -> Any:
    value = explicit if explicit is not None else case.expected
    if value is None:
        raise ValueError(f"{scorer_name}: case {case.id!r} has no expected value")
    return value


def _norm(text: str, *, ignore_case: bool, normalize_whitespace: bool) -> str:
    if normalize_whitespace:
        text = " ".join(text.split())
    return text.casefold() if ignore_case else text


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value]
    return [str(value)]


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


class _Builtin(Scorer):
    def __init__(self, name: str, fn: Callable[[Any, EvalCase], ScoreResult]) -> None:
        self.name = name
        self._fn = fn

    async def score(self, output: Any, case: EvalCase) -> ScoreResult:
        return self._fn(output, case)


def exact_match(
    expected: Any = None,
    *,
    ignore_case: bool = False,
    normalize_whitespace: bool = True,
    name: str = "exact_match",
) -> Scorer:
    """Pass when the output text equals the expected text (or any item of an expected list).

    Args:
        ignore_case: Compare case-insensitively.
        normalize_whitespace: Strip and collapse runs of whitespace before comparing.
    """

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        refs = _as_str_list(_reference(expected, case, name))
        got = _norm(
            output_text(output), ignore_case=ignore_case, normalize_whitespace=normalize_whitespace
        )
        for ref in refs:
            if got == _norm(
                ref, ignore_case=ignore_case, normalize_whitespace=normalize_whitespace
            ):
                return _result(name, True, "exact match")
        return _result(name, False, f"expected {refs[0]!r}, got {output_text(output)[:200]!r}")

    return _Builtin(name, fn)


def contains(
    value: str | list[str] | None = None,
    *,
    ignore_case: bool = False,
    require_all: bool = True,
    name: str = "contains",
) -> Scorer:
    """Pass when the output contains the substring(s). Score is the fraction found.

    Args:
        value: Substring or list of substrings (defaults to the case's expected).
        require_all: Pass only if every substring is present (else any one suffices).
    """

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        needles = _as_str_list(_reference(value, case, name))
        text = _norm(output_text(output), ignore_case=ignore_case, normalize_whitespace=False)
        found = [
            n
            for n in needles
            if _norm(n, ignore_case=ignore_case, normalize_whitespace=False) in text
        ]
        missing = [n for n in needles if n not in found]
        passed = not missing if require_all else bool(found)
        reason = "all found" if not missing else f"missing {missing}"
        return _result(name, passed, reason, len(found) / len(needles))

    return _Builtin(name, fn)


def not_contains(
    value: str | list[str] | None = None,
    *,
    ignore_case: bool = False,
    name: str = "not_contains",
) -> Scorer:
    """Pass when none of the substring(s) appear in the output. Score is the fraction absent."""

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        needles = _as_str_list(_reference(value, case, name))
        text = _norm(output_text(output), ignore_case=ignore_case, normalize_whitespace=False)
        present = [
            n
            for n in needles
            if _norm(n, ignore_case=ignore_case, normalize_whitespace=False) in text
        ]
        reason = "none present" if not present else f"found forbidden {present}"
        return _result(name, not present, reason, 1 - len(present) / len(needles))

    return _Builtin(name, fn)


def regex(
    pattern: str | None = None,
    *,
    flags: int = 0,
    full_match: bool = False,
    name: str = "regex",
) -> Scorer:
    """Pass when the pattern matches the output text (``re.search``, or ``re.fullmatch``)."""

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        pat = str(_reference(pattern, case, name))
        text = output_text(output)
        match = re.fullmatch(pat, text, flags) if full_match else re.search(pat, text, flags)
        return _result(
            name, match is not None, f"pattern {pat!r} {'matched' if match else 'not found'}"
        )

    return _Builtin(name, fn)


def _values_match(exp: Any, got: Any, *, subset: bool, ignore_case: bool) -> bool:
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            return False
        if not subset and set(exp) != set(got):
            return False
        return all(
            k in got and _values_match(v, got[k], subset=subset, ignore_case=ignore_case)
            for k, v in exp.items()
        )
    if isinstance(exp, list):
        return (
            isinstance(got, list)
            and len(exp) == len(got)
            and all(
                _values_match(e, g, subset=subset, ignore_case=ignore_case)
                for e, g in zip(exp, got, strict=True)
            )
        )
    if isinstance(exp, str) and isinstance(got, str):
        return _norm(exp, ignore_case=ignore_case, normalize_whitespace=True) == _norm(
            got, ignore_case=ignore_case, normalize_whitespace=True
        )
    if (
        isinstance(exp, int | float)
        and isinstance(got, int | float)
        and not isinstance(exp, bool)
        and not isinstance(got, bool)
    ):
        return math.isclose(exp, got, rel_tol=1e-9, abs_tol=1e-9)
    return bool(exp == got)


def json_match(
    expected: dict[str, Any] | BaseModel | None = None,
    *,
    subset: bool = True,
    fields: list[str] | None = None,
    ignore_case: bool = False,
    name: str = "json_match",
) -> Scorer:
    """Field-by-field compare of structured output (dict, Pydantic model, ``ExtractResponse``).

    Score is the fraction of compared fields that match; the case passes only when all do.

    Args:
        subset: Compare only the fields present in *expected*. When ``False``, extra
            fields in the output count as mismatches.
        fields: Restrict the comparison to these top-level fields.
        ignore_case: Compare strings case-insensitively.
    """

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        ref = _reference(expected, case, name)
        exp = ref.model_dump(mode="json") if isinstance(ref, BaseModel) else ref
        if isinstance(exp, str):
            exp = json.loads(exp)
        if not isinstance(exp, dict):
            raise ValueError(f"{name}: expected value must be a dict or model")
        got = output_data(output)
        if got is None:
            return _result(name, False, "output is not structured data")
        keys = list(fields) if fields is not None else list(exp)
        if not subset and fields is None:
            keys += [k for k in got if k not in exp]
        if not keys:
            return _result(name, True, "no fields to compare")
        mismatched = [
            k
            for k in keys
            if k not in exp
            or k not in got
            or not _values_match(exp[k], got[k], subset=subset, ignore_case=ignore_case)
        ]
        matched = len(keys) - len(mismatched)
        reason = "all fields match" if not mismatched else f"mismatched fields {mismatched}"
        return _result(name, not mismatched, reason, matched / len(keys))

    return _Builtin(name, fn)


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = output_text(value).replace(",", "")
    try:
        return float(text.strip())
    except ValueError:
        match = _NUMBER_RE.search(text)
        return float(match.group()) if match else None


def numeric_close(
    expected: float | None = None,
    *,
    tolerance: float = 1e-6,
    relative: bool = False,
    field: str | None = None,
    name: str = "numeric_close",
) -> Scorer:
    """Pass when the output number is within *tolerance* of the expected number.

    Args:
        tolerance: Allowed absolute difference, or fraction of ``|expected|`` when *relative*.
        field: Read the number from this field of structured output instead of the text
            (otherwise the first number in the text is used).
    """

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        exp = _to_number(_reference(expected, case, name))
        if exp is None:
            raise ValueError(f"{name}: expected value for case {case.id!r} is not a number")
        source: Any = output
        if field is not None:
            data = output_data(output) or {}
            if field not in data:
                return _result(name, False, f"field {field!r} missing from output")
            source = data[field]
        got = _to_number(source)
        if got is None:
            return _result(name, False, "no number in output")
        diff = abs(got - exp)
        allowed = tolerance * abs(exp) if relative else tolerance
        return _result(name, diff <= allowed, f"got {got:g}, expected {exp:g} (diff {diff:g})")

    return _Builtin(name, fn)


def _predicted_labels(output: Any) -> list[str]:
    scores = getattr(output, "scores", None)
    if isinstance(scores, list) and scores:
        labels = [str(getattr(s, "label", s)) for s in scores]
        return labels
    label = getattr(output, "label", None)
    if label is not None:
        return [str(label)]
    if isinstance(output, dict) and "label" in output:
        return [str(output["label"])]
    return [output_text(output).strip()] if output is not None else []


def label_match(
    expected: str | list[str] | None = None,
    *,
    ignore_case: bool = True,
    name: str = "label_match",
) -> Scorer:
    """Compare a ``ClassifyResponse`` label against the expected label.

    With a single expected label, the top predicted label must equal it. With a list
    (multi-label), the predicted label set must equal it; the score is the Jaccard overlap.
    """

    def norm(s: str) -> str:
        return s.strip().casefold() if ignore_case else s.strip()

    def fn(output: Any, case: EvalCase) -> ScoreResult:
        ref = _reference(expected, case, name)
        predicted = _predicted_labels(output)
        top = getattr(output, "label", None)
        if isinstance(ref, list | tuple | set):
            want = {norm(str(r)) for r in ref}
            got = {norm(p) for p in predicted}
            union = want | got
            score = len(want & got) / len(union) if union else 1.0
            return _result(
                name, want == got, f"predicted {sorted(got)}, expected {sorted(want)}", score
            )
        top_label = str(top) if top is not None else (predicted[0] if predicted else "")
        ok = norm(top_label) == norm(str(ref))
        return _result(name, ok, f"predicted {top_label!r}, expected {ref!r}")

    return _Builtin(name, fn)


# ---------------------------------------------------------------------------
# Model-backed scorers
# ---------------------------------------------------------------------------


def _settings_for(provider: str | None, model: str | None, field: str = "MODEL") -> AIForgeSettings:
    from aiforge.infra.config import AIForgeSettings, get_settings

    base = get_settings()
    overrides: dict[str, Any] = {}
    if provider:
        overrides["PROVIDER"] = provider
    if model:
        overrides[field] = model
    if not overrides:
        return base
    merged = base.model_dump()
    merged.update(overrides)
    if "EMBEDDING_MODEL" not in overrides:
        # Switching provider must not trip the provider/embedding-model check.
        from aiforge.infra.config import _PROVIDER_EMBEDDING_MAP

        allowed = _PROVIDER_EMBEDDING_MAP.get(str(merged["PROVIDER"]), [])
        if allowed and merged["EMBEDDING_MODEL"] not in allowed:
            merged["EMBEDDING_MODEL"] = allowed[0]
    return AIForgeSettings(**merged)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingSimilarity(Scorer):
    """Cosine similarity between output and expected text embeddings."""

    def __init__(
        self,
        expected: str | None = None,
        *,
        threshold: float = 0.8,
        embedder: EmbeddingProvider | None = None,
        provider: str | None = None,
        model: str | None = None,
        name: str = "embedding_similarity",
    ) -> None:
        self.name = name
        self.expected = expected
        self.threshold = threshold
        self._embedder = embedder
        self._provider = provider
        self._model = model

    @property
    def embedder(self) -> EmbeddingProvider:
        """The embedding provider, resolved from settings on first use."""
        if self._embedder is None:
            from aiforge.providers import get_embedding_provider

            self._embedder = get_embedding_provider(
                _settings_for(self._provider, self._model, "EMBEDDING_MODEL")
            )
        return self._embedder

    async def score(self, output: Any, case: EvalCase) -> ScoreResult:
        ref = str(_reference(self.expected, case, self.name))
        vectors = await self.embedder.embed_async([output_text(output), ref])
        sim = _cosine(vectors[0], vectors[1])
        return ScoreResult(
            name=self.name,
            score=_clamp(sim),
            passed=sim >= self.threshold,
            reason=f"cosine similarity {sim:.3f} (threshold {self.threshold})",
        )


def embedding_similarity(
    expected: str | None = None,
    *,
    threshold: float = 0.8,
    embedder: EmbeddingProvider | None = None,
    provider: str | None = None,
    model: str | None = None,
    name: str = "embedding_similarity",
) -> Scorer:
    """Pass when the cosine similarity of output and expected embeddings is ≥ *threshold*.

    Uses the configured :class:`~aiforge.core.provider.EmbeddingProvider` unless
    *embedder* is given; *provider* / *model* override ``PROVIDER`` / ``EMBEDDING_MODEL``.
    """
    return EmbeddingSimilarity(
        expected,
        threshold=threshold,
        embedder=embedder,
        provider=provider,
        model=model,
        name=name,
    )


class JudgeVerdict(BaseModel):
    """Fixed schema the LLM judge must return."""

    score: float = Field(
        description="Quality score from 0.0 (fails the rubric) to 1.0 (fully meets it)"
    )
    passed: bool = Field(description="Whether the response meets the rubric")
    reason: str = Field(description="One or two sentences justifying the score")


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. Grade the RESPONSE against the RUBRIC. "
    "Use the INPUT and the EXPECTED reference (when given) as context. "
    'Reply with only a JSON object: {"score": <number 0.0-1.0>, "passed": <true|false>, '
    '"reason": "<one or two sentences>"}.'
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(text: str) -> JudgeVerdict | None:
    match = _JSON_OBJECT_RE.search(text or "")
    if not match:
        return None
    try:
        return JudgeVerdict.model_validate_json(match.group())
    except PydanticValidationError:
        return None


class LLMJudge(Scorer):
    """Rubric-based grading by an LLM; see :func:`llm_judge`."""

    def __init__(
        self,
        rubric: str,
        *,
        llm: LLMProvider | None = None,
        provider: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        temperature: float = 0.0,
        include_input: bool = True,
        name: str = "llm_judge",
    ) -> None:
        if not rubric or not rubric.strip():
            raise ValueError("llm_judge requires a non-empty rubric")
        self.name = name
        self.rubric = rubric
        self.threshold = threshold
        self.temperature = temperature
        self.include_input = include_input
        self._llm = llm
        self._provider = provider
        self._model = model

    @property
    def llm(self) -> LLMProvider:
        """The judge provider, resolved from settings on first use."""
        if self._llm is None:
            from aiforge.providers import get_provider

            self._llm = get_provider(_settings_for(self._provider, self._model))
        return self._llm

    def build_messages(self, output: Any, case: EvalCase) -> list[dict[str, Any]]:
        """The judge prompt for one case."""
        parts = [f"RUBRIC:\n{self.rubric}"]
        if self.include_input:
            parts.append(f"INPUT:\n{output_text(case.input)}")
        if case.expected is not None:
            parts.append(f"EXPECTED (reference):\n{output_text(case.expected)}")
        parts.append(f"RESPONSE:\n{output_text(output)}")
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def score(self, output: Any, case: EvalCase) -> ScoreResult:
        from aiforge.core.types import APIConnectionError, APITimeoutError, RateLimitError

        messages = self.build_messages(output, case)
        llm = self.llm
        cost = 0.0
        verdict: JudgeVerdict | None = None
        raw_text = ""

        if getattr(llm, "supports_structured_output", False) is True:
            try:
                res = await llm.complete_async(
                    messages, response_format=JudgeVerdict, temperature=self.temperature
                )
            except (RateLimitError, APITimeoutError, APIConnectionError):
                raise
            except Exception as exc:  # noqa: BLE001 — native path unsupported; fall back
                logger.warning("evals.judge_structured_fallback", error=str(exc)[:200])
            else:
                cost += _cost(res)
                raw_text = res.text
                if res.parsed is not None:
                    try:
                        verdict = JudgeVerdict.model_validate(res.parsed)
                    except PydanticValidationError:
                        verdict = None
                if verdict is None:
                    verdict = _parse_verdict(res.text)

        if verdict is None:
            res = await llm.complete_async(messages, temperature=self.temperature)
            cost += _cost(res)
            raw_text = res.text
            verdict = _parse_verdict(res.text)

        if verdict is None:
            return ScoreResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason=f"judge returned no parseable verdict: {raw_text[:200]!r}",
                cost=cost,
            )
        score = _clamp(verdict.score)
        passed = score >= self.threshold if self.threshold is not None else verdict.passed
        return ScoreResult(
            name=self.name, score=score, passed=passed, reason=verdict.reason, cost=cost
        )


def _cost(res: CompletionResult) -> float:
    try:
        return float(res.cost)
    except Exception:  # noqa: BLE001 — unknown model pricing should not fail scoring
        return 0.0


def llm_judge(
    rubric: str,
    *,
    llm: LLMProvider | None = None,
    provider: str | None = None,
    model: str | None = None,
    threshold: float | None = None,
    temperature: float = 0.0,
    include_input: bool = True,
    name: str = "llm_judge",
) -> Scorer:
    """Grade outputs against a free-text rubric with an LLM.

    Uses the provider's native structured output (``response_format=JudgeVerdict``)
    when supported; otherwise, or if that call fails, it asks for plain JSON and
    parses it. A verdict that cannot be parsed scores 0 with the raw text as reason.

    Args:
        rubric: What a good response looks like.
        llm: A ready :class:`~aiforge.core.provider.LLMProvider` to use as judge.
        provider / model: Judge provider and model, independent of the target's
            (defaults to the configured ones).
        threshold: Pass when ``score >= threshold`` instead of using the judge's ``passed``.
    """
    return LLMJudge(
        rubric,
        llm=llm,
        provider=provider,
        model=model,
        threshold=threshold,
        temperature=temperature,
        include_input=include_input,
        name=name,
    )


# ---------------------------------------------------------------------------
# Name registry (used by the CLI)
# ---------------------------------------------------------------------------

SCORERS: dict[str, Callable[..., Scorer]] = {
    "exact_match": exact_match,
    "contains": contains,
    "not_contains": not_contains,
    "regex": regex,
    "json_match": json_match,
    "numeric_close": numeric_close,
    "label_match": label_match,
    "embedding_similarity": embedding_similarity,
    "llm_judge": llm_judge,
}


def get_scorer(name: str, /, **params: Any) -> Scorer:
    """Build a built-in scorer by name, e.g. ``get_scorer("contains", value="refund")``."""
    try:
        factory = SCORERS[name]
    except KeyError:
        raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}") from None
    return factory(**params)


__all__ = [
    "EmbeddingSimilarity",
    "FunctionScorer",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeVerdict",
    "LLMJudge",
    "SCORERS",
    "ScoreResult",
    "Scorer",
    "ScorerFn",
    "as_scorer",
    "contains",
    "embedding_similarity",
    "exact_match",
    "get_scorer",
    "json_match",
    "label_match",
    "llm_judge",
    "not_contains",
    "numeric_close",
    "output_data",
    "output_text",
    "regex",
    "scorer",
]
