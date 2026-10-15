"""Run a target over a dataset with bounded concurrency and score every case."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aiforge.evals.dataset import Dataset, EvalCase
from aiforge.evals.report import CaseResult, EvalReport
from aiforge.evals.scorers import Scorer, ScoreResult, ScorerFn, as_scorer

# Main (sync, async) method for each built-in module; anything else uses run/arun.
MODULE_METHODS: dict[str, tuple[str, str]] = {
    "Classifier": ("classify", "aclassify"),
    "Extractor": ("extract", "aextract"),
    "Summarizer": ("summarize", "asummarize"),
    "Generator": ("generate", "agenerate"),
    "Moderator": ("check", "acheck"),
    "Translator": ("translate", "atranslate"),
    "Vision": ("analyze", "aanalyze"),
    "DocQA": ("ask", "aask"),
    "SemanticSearch": ("query", "aquery"),
    "ChatAgent": ("chat", "achat"),
}

AsyncCall = Callable[..., Awaitable[Any]]
DatasetLike = Dataset | Iterable[EvalCase | dict[str, Any]] | str | Path


def _wrap_callable(fn: Callable[..., Any]) -> AsyncCall:
    """Return an async callable; sync functions run in a worker thread."""
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)  # noqa: B004 — async callable objects
    ):
        return fn

    async def call(*args: Any, **kwargs: Any) -> Any:
        result = await asyncio.to_thread(fn, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    return call


def resolve_target(target: Any, method: str | None = None) -> tuple[AsyncCall, str]:
    """Map *target* to an async ``call(input, **kwargs)`` and a display name.

    AI Forge modules use their main method (``Classifier.aclassify``, …), or
    *method* when given (its ``a``-prefixed async twin is preferred when it
    exists). Plain callables and async callables are used as-is.
    """
    from aiforge.core.types import BaseModule

    if isinstance(target, BaseModule):
        cls_name = type(target).__name__
        if method:
            for candidate in (f"a{method}", method):
                if inspect.iscoroutinefunction(getattr(target, candidate, None)):
                    return getattr(target, candidate), f"{cls_name}.{candidate}"
            if not callable(getattr(target, method, None)):
                raise AttributeError(f"{cls_name} has no method {method!r}")
            return _wrap_callable(getattr(target, method)), f"{cls_name}.{method}"
        for klass in type(target).__mro__:
            if klass.__name__ in MODULE_METHODS:
                _, async_name = MODULE_METHODS[klass.__name__]
                return getattr(target, async_name), f"{cls_name}.{async_name}"
        return target.arun, f"{cls_name}.arun"
    if method:
        bound = getattr(target, method, None)
        if not callable(bound):
            raise AttributeError(f"{type(target).__name__} has no method {method!r}")
        return _wrap_callable(bound), f"{type(target).__name__}.{method}"
    if not callable(target):
        raise TypeError(f"Target must be an AI Forge module or a callable, got {target!r}")
    return _wrap_callable(target), getattr(target, "__qualname__", type(target).__name__)


def _as_dataset(dataset: DatasetLike) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if isinstance(dataset, str | Path):
        return Dataset.load(dataset)
    return Dataset.from_list(dataset)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _usage(output: Any) -> tuple[float, int, int]:
    def num(attr: str) -> float:
        value = output.get(attr) if isinstance(output, dict) else getattr(output, attr, 0)
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0

    return num("cost"), int(num("input_tokens")), int(num("output_tokens"))


async def _score_one(scorer: Scorer, output: Any, case: EvalCase) -> ScoreResult:
    try:
        result = await scorer.score(output, case)
    except Exception as exc:  # noqa: BLE001 — a broken scorer fails the case, not the run
        return ScoreResult(
            name=scorer.name,
            score=0.0,
            passed=False,
            reason=f"scorer error: {type(exc).__name__}: {exc}",
        )
    return result if result.name == scorer.name else result.model_copy(update={"name": scorer.name})


async def aevaluate(
    target: Any,
    dataset: DatasetLike,
    scorers: Sequence[Scorer | ScorerFn] = (),
    *,
    method: str | None = None,
    concurrency: int = 5,
    timeout: float | None = None,
    call_kwargs: dict[str, Any] | None = None,
    name: str = "",
    min_pass_rate: float = 1.0,
    thresholds: dict[str, float] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Run *target* on every case of *dataset* and score the outputs.

    Args:
        target: An AI Forge module instance (its main method is used) or any
            sync/async callable taking the case input as first argument.
        dataset: A :class:`Dataset`, a list of case dicts, or a path to a
            ``.jsonl`` / ``.json`` / ``.yaml`` / ``.csv`` file.
        scorers: Built-in scorers or plain ``fn(output, case)`` functions.
        method: Override the module method to call (e.g. ``"sentiment"``).
        concurrency: Maximum cases in flight at once.
        timeout: Per-case timeout in seconds for the target call.
        call_kwargs: Keyword arguments passed on every call (per-case
            ``kwargs`` take precedence).
        min_pass_rate: Default minimum pass rate each scorer must reach.
        thresholds: Per-scorer minimum pass rates, by scorer name.
        on_result: Called with each :class:`CaseResult` as it completes.

    Returns:
        An :class:`EvalReport`; failures of the target or a scorer are
        recorded on the case, never raised.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    ds = _as_dataset(dataset)
    call, target_name = resolve_target(target, method)
    scorer_objs = [as_scorer(s) for s in scorers]
    names = [s.name for s in scorer_objs]
    if len(set(names)) != len(names):
        raise ValueError(f"Scorer names must be unique, got {names}; pass name= to tell them apart")
    unknown = set(thresholds or {}) - set(names)
    if unknown:
        raise ValueError(f"Thresholds given for unknown scorers {sorted(unknown)}")
    base_kwargs = dict(call_kwargs or {})
    sem = asyncio.Semaphore(concurrency)

    async def run_case(case: EvalCase) -> CaseResult:
        async with sem:
            output: Any = None
            error: str | None = None
            t0 = time.perf_counter()
            try:
                pending = call(case.input, **{**base_kwargs, **case.kwargs})
                output = await (asyncio.wait_for(pending, timeout) if timeout else pending)
            except TimeoutError:
                error = f"TimeoutError: no result within {timeout}s"
            except Exception as exc:  # noqa: BLE001 — recorded on the case
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - t0) * 1000

            if error is None:
                scores = list(
                    await asyncio.gather(*(_score_one(s, output, case) for s in scorer_objs))
                )
            else:
                scores = [
                    ScoreResult(
                        name=s.name, score=0.0, passed=False, reason=f"target error: {error}"
                    )
                    for s in scorer_objs
                ]
        cost, in_tok, out_tok = _usage(output)
        result = CaseResult(
            id=case.id,
            input=_jsonable(case.input),
            expected=_jsonable(case.expected),
            output=_jsonable(output),
            error=error,
            scores=scores,
            cost=cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            metadata=case.metadata,
        )
        if on_result is not None:
            on_result(result)
        return result

    results = await asyncio.gather(*(run_case(c) for c in ds.cases))
    return EvalReport(
        name=name or ds.name,
        target=target_name,
        scorer_names=names,
        cases=list(results),
        min_pass_rate=min_pass_rate,
        thresholds=dict(thresholds or {}),
    )


def evaluate(
    target: Any,
    dataset: DatasetLike,
    scorers: Sequence[Scorer | ScorerFn] = (),
    **kwargs: Any,
) -> EvalReport:
    """Synchronous :func:`aevaluate`; safe to call from inside a running event loop."""

    def run() -> EvalReport:
        return asyncio.run(aevaluate(target, dataset, scorers, **kwargs))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()
