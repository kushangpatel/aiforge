"""``aiforge eval`` — run evaluation datasets and gate CI on the results.

Exit codes: 0 passed, 1 below a pass-rate threshold, 2 bad usage,
3 regressed against the baseline (thresholds met).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer

from aiforge.cli._helpers import console

eval_app = typer.Typer(name="eval", help="Evaluate modules against test datasets.")

EXIT_BELOW_THRESHOLD = 1
EXIT_REGRESSED = 3


def _json_object(raw: str | None, option: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option} must be a JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{option} must be a JSON object")
    return value


def _parse_scorer(spec: str, judge_provider: str | None, judge_model: str | None) -> Any:
    """``name`` or ``name:{"param": ...}`` → a built-in scorer."""
    from aiforge.evals import get_scorer

    name, _, raw = spec.partition(":")
    params = _json_object(raw or None, f"--scorer {name}")
    if name.strip() == "llm_judge":
        if judge_provider:
            params.setdefault("provider", judge_provider)
        if judge_model:
            params.setdefault("model", judge_model)
    try:
        return get_scorer(name.strip(), **params)
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"--scorer {spec!r}: {exc}") from exc


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        try:
            if not sep:
                raise ValueError
            out[name.strip()] = float(value)
        except ValueError:
            raise typer.BadParameter(f"--threshold must look like name=0.9, got {item!r}") from None
    return out


def _load_target(module: str | None, target: str | None, module_args: dict[str, Any]) -> Any:
    if bool(module) == bool(target):
        raise typer.BadParameter("Pass exactly one of --module or --target")
    if target:
        mod_name, _, attr = target.partition(":")
        if not attr:
            raise typer.BadParameter("--target must look like 'package.module:function'")
        try:
            obj = getattr(importlib.import_module(mod_name), attr)
        except (ImportError, AttributeError) as exc:
            raise typer.BadParameter(f"--target {target!r}: {exc}") from exc
        return obj(**module_args) if module_args else obj

    import aiforge  # noqa: F401 — registers the built-in modules
    from aiforge.registry import get_module

    try:
        cls = get_module(str(module))
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        return cls(**module_args)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"--module-args for {module!r}: {exc}") from exc


@eval_app.command("run")
def run(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False, help="Dataset file."),
    module: str | None = typer.Option(
        None, "--module", "-m", help="Registered module name, e.g. classify."
    ),
    module_args: str | None = typer.Option(
        None, "--module-args", help='Module constructor kwargs as JSON, e.g. \'{"labels": ["a"]}\'.'
    ),
    target: str | None = typer.Option(
        None, "--target", help="Callable to evaluate instead of a module: 'pkg.mod:func'."
    ),
    method: str | None = typer.Option(
        None, "--method", help="Module method to call (default: its main method)."
    ),
    call_args: str | None = typer.Option(
        None, "--call-args", help="Kwargs passed on every call, as JSON."
    ),
    scorers: list[str] | None = typer.Option(
        None,
        "--scorer",
        "-s",
        help='Scorer name, optionally with JSON params: \'contains:{"value": "refund"}\'. Repeatable.',
    ),
    min_pass_rate: float = typer.Option(
        1.0, "--min-pass-rate", min=0.0, max=1.0, help="Minimum pass rate for every scorer."
    ),
    threshold: list[str] | None = typer.Option(
        None, "--threshold", help="Per-scorer minimum, e.g. llm_judge=0.8. Repeatable."
    ),
    concurrency: int = typer.Option(5, "--concurrency", "-c", min=1, help="Cases in flight."),
    timeout: float | None = typer.Option(None, "--timeout", help="Per-case timeout (seconds)."),
    judge_provider: str | None = typer.Option(
        None, "--judge-provider", help="Provider for llm_judge (default: configured)."
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model for llm_judge (default: configured)."
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the JSON report here."),
    baseline: Path | None = typer.Option(
        None, "--baseline", "-b", exists=True, dir_okay=False, help="Baseline report to compare."
    ),
    tolerance: float = typer.Option(
        0.0, "--tolerance", min=0.0, max=1.0, help="Allowed drop before a regression is flagged."
    ),
    name: str = typer.Option("", "--name", help="Report name (default: dataset file stem)."),
    max_failures: int = typer.Option(10, "--max-failures", help="Failing cases to print."),
) -> None:
    """Run a dataset through a module and score it. Exits non-zero on failure."""
    from aiforge.evals import Dataset, EvalReport, evaluate

    if not scorers:
        raise typer.BadParameter("Pass at least one --scorer")
    scorer_objs = [_parse_scorer(s, judge_provider, judge_model) for s in scorers]
    thresholds = _parse_thresholds(threshold or [])
    try:
        ds = Dataset.load(dataset)
    except (ValueError, TypeError) as exc:
        raise typer.BadParameter(f"dataset: {exc}") from exc
    tgt = _load_target(module, target, _json_object(module_args, "--module-args"))

    try:
        report = evaluate(
            tgt,
            ds,
            scorer_objs,
            method=method,
            concurrency=concurrency,
            timeout=timeout,
            call_kwargs=_json_object(call_args, "--call-args"),
            name=name,
            min_pass_rate=min_pass_rate,
            thresholds=thresholds,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    report.print(console, max_failures=max_failures)
    if output:
        report.save(output)
        console.print(f"[green]✓[/green] Report written to {output}")

    regressed = False
    if baseline:
        comparison = report.compare(EvalReport.load(baseline), tolerance=tolerance)
        comparison.print(console)
        regressed = comparison.regressed

    if not report.passed:
        raise typer.Exit(EXIT_BELOW_THRESHOLD)
    if regressed:
        raise typer.Exit(EXIT_REGRESSED)


@eval_app.command("compare")
def compare(
    current: Path = typer.Argument(..., exists=True, dir_okay=False, help="Current report JSON."),
    baseline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Baseline report JSON."),
    tolerance: float = typer.Option(
        0.0, "--tolerance", min=0.0, max=1.0, help="Allowed drop before a regression is flagged."
    ),
) -> None:
    """Compare two saved reports. Exits 3 when the current one regressed."""
    from aiforge.evals import EvalReport

    comparison = EvalReport.load(current).compare(EvalReport.load(baseline), tolerance=tolerance)
    comparison.print(console)
    if comparison.regressed:
        raise typer.Exit(EXIT_REGRESSED)
