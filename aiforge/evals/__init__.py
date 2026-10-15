"""
AI Forge evals: score module outputs against test sets and catch regressions.

Example::

    from aiforge import Classifier
    from aiforge.evals import evaluate, label_match

    report = evaluate(
        Classifier(labels=["bug", "feature", "question"]),
        "evals/tickets.yaml",
        scorers=[label_match()],
        min_pass_rate=0.9,
    )
    report.print()
    report.save("eval-report.json")
"""

from __future__ import annotations

from aiforge.evals.dataset import Dataset, EvalCase
from aiforge.evals.report import (
    CaseResult,
    EvalComparison,
    EvalReport,
    ScorerDelta,
    ScorerStats,
    assert_eval,
)
from aiforge.evals.runner import MODULE_METHODS, aevaluate, evaluate, resolve_target
from aiforge.evals.scorers import (
    SCORERS,
    JudgeVerdict,
    Scorer,
    ScoreResult,
    as_scorer,
    contains,
    embedding_similarity,
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
    scorer,
)

__all__ = [
    "MODULE_METHODS",
    "SCORERS",
    "CaseResult",
    "Dataset",
    "EvalCase",
    "EvalComparison",
    "EvalReport",
    "JudgeVerdict",
    "ScoreResult",
    "Scorer",
    "ScorerDelta",
    "ScorerStats",
    "aevaluate",
    "as_scorer",
    "assert_eval",
    "contains",
    "embedding_similarity",
    "evaluate",
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
    "resolve_target",
    "scorer",
]
