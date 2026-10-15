"""Evaluation reports: per-case results, per-scorer aggregates, thresholds and regressions."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aiforge.evals.scorers import ScoreResult

if TYPE_CHECKING:
    from rich.console import Console
    from rich.table import Table

_EPS = 1e-9


class CaseResult(BaseModel):
    """Outcome of running the target and every scorer on one case."""

    id: str
    input: Any = None
    expected: Any = None
    output: Any = None
    """JSON-friendly copy of the target's output (``None`` on error)."""
    error: str | None = None
    """``"ExceptionType: message"`` when the target raised or timed out."""
    scores: list[ScoreResult] = Field(default_factory=list)
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """``True`` when the target succeeded and every scorer passed."""
        return self.error is None and all(s.passed for s in self.scores)

    def score(self, name: str) -> ScoreResult | None:
        """The result of scorer *name* for this case, if it ran."""
        return next((s for s in self.scores if s.name == name), None)


class ScorerStats(BaseModel):
    """Aggregate of one scorer across all cases."""

    name: str
    count: int
    mean: float
    pass_rate: float
    passed_count: int
    min_pass_rate: float
    passed: bool
    """``pass_rate >= min_pass_rate``."""


class ScorerDelta(BaseModel):
    """Change in one scorer between a baseline report and the current one."""

    name: str
    status: Literal["compared", "added", "removed"] = "compared"
    baseline_mean: float | None = None
    current_mean: float | None = None
    baseline_pass_rate: float | None = None
    current_pass_rate: float | None = None
    regressed: bool = False

    @property
    def mean_delta(self) -> float:
        if self.baseline_mean is None or self.current_mean is None:
            return 0.0
        return self.current_mean - self.baseline_mean

    @property
    def pass_rate_delta(self) -> float:
        if self.baseline_pass_rate is None or self.current_pass_rate is None:
            return 0.0
        return self.current_pass_rate - self.baseline_pass_rate


class EvalComparison(BaseModel):
    """Result of :meth:`EvalReport.compare`."""

    scorers: list[ScorerDelta]
    newly_failing: list[str] = Field(default_factory=list)
    """Case ids that passed in the baseline and fail now."""
    newly_passing: list[str] = Field(default_factory=list)
    baseline_pass_rate: float
    current_pass_rate: float
    cost_delta: float = 0.0
    tolerance: float = 0.0

    @property
    def regressions(self) -> list[ScorerDelta]:
        """Scorers whose mean or pass rate dropped by more than the tolerance."""
        return [s for s in self.scorers if s.regressed]

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)

    def to_table(self) -> Table:
        """A rich table of per-scorer deltas."""
        from rich.table import Table

        table = Table(title="Comparison with baseline", show_header=True)
        for col, justify in (
            ("Scorer", "left"),
            ("Pass rate", "right"),
            ("Δ pass", "right"),
            ("Mean", "right"),
            ("Δ mean", "right"),
            ("Status", "left"),
        ):
            table.add_column(col, justify=justify)  # type: ignore[arg-type]
        for d in self.scorers:
            if d.status != "compared":
                table.add_row(d.name, "", "", "", "", d.status)
                continue
            status = "[red]REGRESSED[/red]" if d.regressed else "[green]ok[/green]"
            table.add_row(
                d.name,
                f"{d.current_pass_rate:.1%}",
                f"{d.pass_rate_delta:+.1%}",
                f"{d.current_mean:.3f}",
                f"{d.mean_delta:+.3f}",
                status,
            )
        return table

    def print(self, console: Console | None = None) -> None:
        """Print the comparison table and newly failing cases."""
        from rich.console import Console

        con = console or Console()
        con.print(self.to_table())
        if self.newly_failing:
            con.print(f"[red]Newly failing cases:[/red] {', '.join(self.newly_failing)}")


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of *values* (``pct`` in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class EvalReport(BaseModel):
    """Results of :func:`~aiforge.evals.evaluate`.

    Aggregates (scorer stats, pass rates, cost, latency percentiles) are
    computed from :attr:`cases`, so a report loaded from JSON behaves the same
    as a fresh one.
    """

    name: str = ""
    target: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    scorer_names: list[str] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    min_pass_rate: float = 1.0
    """Default minimum pass rate for every scorer."""
    thresholds: dict[str, float] = Field(default_factory=dict)
    """Per-scorer minimum pass rates overriding :attr:`min_pass_rate`."""

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def threshold_for(self, scorer: str) -> float:
        return self.thresholds.get(scorer, self.min_pass_rate)

    def scorer_stats(self) -> dict[str, ScorerStats]:
        """Mean score and pass rate per scorer, in scorer order."""
        stats: dict[str, ScorerStats] = {}
        for name in self.scorer_names:
            results = [r for c in self.cases if (r := c.score(name)) is not None]
            count = len(results)
            passed_count = sum(1 for r in results if r.passed)
            pass_rate = passed_count / count if count else 0.0
            minimum = self.threshold_for(name)
            stats[name] = ScorerStats(
                name=name,
                count=count,
                mean=sum(r.score for r in results) / count if count else 0.0,
                pass_rate=pass_rate,
                passed_count=passed_count,
                min_pass_rate=minimum,
                passed=count > 0 and pass_rate + _EPS >= minimum,
            )
        return stats

    @property
    def pass_rate(self) -> float:
        """Fraction of cases where the target succeeded and every scorer passed."""
        return sum(1 for c in self.cases if c.passed) / len(self.cases) if self.cases else 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.cases if c.error is not None)

    @property
    def total_cost(self) -> float:
        """USD spent by the target across all cases."""
        return sum(c.cost for c in self.cases)

    @property
    def scoring_cost(self) -> float:
        """USD spent by scorers (LLM judge calls)."""
        return sum(s.cost for c in self.cases for s in c.scores)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.cases)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.cases)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def latency_percentiles(self) -> dict[str, float]:
        """Per-case latency in ms: ``mean``, ``p50``, ``p90``, ``p95``, ``p99`` and ``max``."""
        values = [c.latency_ms for c in self.cases]
        return {
            "mean": sum(values) / len(values) if values else 0.0,
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
            "max": max(values) if values else 0.0,
        }

    @property
    def passed(self) -> bool:
        """``True`` when there is at least one case and every scorer meets its minimum pass rate.

        With no scorers, every case must have run without error.
        """
        if not self.cases:
            return False
        if not self.scorer_names:
            return self.error_count == 0
        return all(s.passed for s in self.scorer_stats().values())

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def summary(self) -> dict[str, Any]:
        """The aggregates as a JSON-friendly dict."""
        return {
            "passed": self.passed,
            "cases": len(self.cases),
            "pass_rate": self.pass_rate,
            "errors": self.error_count,
            "total_cost": self.total_cost,
            "scoring_cost": self.scoring_cost,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "latency_ms": self.latency_percentiles(),
            "scorers": {k: v.model_dump() for k, v in self.scorer_stats().items()},
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, baseline: EvalReport, *, tolerance: float = 0.0) -> EvalComparison:
        """Compare against a *baseline* report and flag per-scorer regressions.

        A scorer regresses when its pass rate or mean score drops by more than
        *tolerance* (absolute, 0..1). Cases are matched by id.
        """
        current_stats = self.scorer_stats()
        base_stats = baseline.scorer_stats()
        deltas: list[ScorerDelta] = []
        for name in [
            *self.scorer_names,
            *(n for n in baseline.scorer_names if n not in current_stats),
        ]:
            cur, base = current_stats.get(name), base_stats.get(name)
            if cur is None or base is None:
                deltas.append(ScorerDelta(name=name, status="added" if base is None else "removed"))
                continue
            regressed = (
                cur.pass_rate < base.pass_rate - tolerance - _EPS
                or cur.mean < base.mean - tolerance - _EPS
            )
            deltas.append(
                ScorerDelta(
                    name=name,
                    baseline_mean=base.mean,
                    current_mean=cur.mean,
                    baseline_pass_rate=base.pass_rate,
                    current_pass_rate=cur.pass_rate,
                    regressed=regressed,
                )
            )
        base_cases = {c.id: c.passed for c in baseline.cases}
        newly_failing = [c.id for c in self.cases if base_cases.get(c.id) is True and not c.passed]
        newly_passing = [c.id for c in self.cases if base_cases.get(c.id) is False and c.passed]
        return EvalComparison(
            scorers=deltas,
            newly_failing=newly_failing,
            newly_passing=newly_passing,
            baseline_pass_rate=baseline.pass_rate,
            current_pass_rate=self.pass_rate,
            cost_delta=self.total_cost - baseline.total_cost,
            tolerance=tolerance,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The full report plus a computed ``summary`` block."""
        data = self.model_dump(mode="json")
        data["summary"] = self.summary()
        return data

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, text: str) -> EvalReport:
        data = json.loads(text)
        data.pop("summary", None)
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        """Load a report written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_table(self) -> Table:
        """A rich table with one row per scorer."""
        from rich.table import Table

        title = f"Eval: {self.name}" if self.name else "Eval results"
        table = Table(title=title, show_header=True)
        table.add_column("Scorer", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Pass rate", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Status")
        for s in self.scorer_stats().values():
            status = "[green]PASS[/green]" if s.passed else "[red]FAIL[/red]"
            table.add_row(
                s.name,
                f"{s.mean:.3f}",
                f"{s.pass_rate:.1%} ({s.passed_count}/{s.count})",
                f"{s.min_pass_rate:.0%}",
                status,
            )
        return table

    def print(self, console: Console | None = None, *, max_failures: int = 10) -> None:
        """Print the scorer table, totals and up to *max_failures* failing cases."""
        from rich.console import Console
        from rich.table import Table

        con = console or Console()
        con.print(self.to_table())
        lat = self.latency_percentiles()
        verdict = "[green]PASSED[/green]" if self.passed else "[red]FAILED[/red]"
        con.print(
            f"{verdict}  cases={len(self.cases)}  pass_rate={self.pass_rate:.1%}  "
            f"errors={self.error_count}  cost=${self.total_cost:.4f}"
            f" (+${self.scoring_cost:.4f} scoring)  tokens={self.total_tokens}  "
            f"latency p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms"
        )
        failures = self.failed_cases[:max_failures]
        if not failures:
            return
        table = Table(title="Failing cases", show_header=True)
        table.add_column("Case", style="cyan")
        table.add_column("Input", overflow="fold", max_width=40)
        table.add_column("Reason", overflow="fold")
        for c in failures:
            reason = c.error or "; ".join(f"{s.name}: {s.reason}" for s in c.scores if not s.passed)
            text = c.input if isinstance(c.input, str) else json.dumps(c.input, default=str)
            table.add_row(c.id, text[:200], reason[:300])
        con.print(table)
        if len(self.failed_cases) > max_failures:
            con.print(f"[dim]… {len(self.failed_cases) - max_failures} more failing cases[/dim]")


def assert_eval(
    report: EvalReport,
    *,
    baseline: EvalReport | str | Path | None = None,
    tolerance: float = 0.0,
    max_failures: int = 5,
) -> None:
    """Raise ``AssertionError`` with a readable summary unless *report* passed.

    With a *baseline* (report or path), regressions also fail. Use it in pytest::

        def test_ticket_classifier():
            report = evaluate(Classifier(labels=LABELS), "evals/tickets.yaml",
                              scorers=[label_match()], min_pass_rate=0.9)
            assert_eval(report)
    """
    problems: list[str] = []
    if not report.cases:
        problems.append("report has no cases")
    for s in report.scorer_stats().values():
        if not s.passed:
            problems.append(
                f"{s.name}: pass rate {s.pass_rate:.1%} below minimum {s.min_pass_rate:.1%}"
            )
    if not report.scorer_names and report.error_count:
        problems.append(f"{report.error_count} case(s) raised errors")
    if baseline is not None:
        base = baseline if isinstance(baseline, EvalReport) else EvalReport.load(baseline)
        comparison = report.compare(base, tolerance=tolerance)
        for d in comparison.regressions:
            problems.append(
                f"{d.name}: regressed (pass rate {d.pass_rate_delta:+.1%}, mean {d.mean_delta:+.3f})"
            )
    if not problems:
        return
    lines = [
        f"Eval {report.name or ''} failed:".replace("  ", " "),
        *(f"  - {p}" for p in problems),
    ]
    failures = report.failed_cases[:max_failures]
    if failures:
        lines.append("Failing cases:")
        for c in failures:
            reason = c.error or "; ".join(f"{s.name}: {s.reason}" for s in c.scores if not s.passed)
            lines.append(f"  [{c.id}] {reason[:200]}")
    raise AssertionError("\n".join(lines))
