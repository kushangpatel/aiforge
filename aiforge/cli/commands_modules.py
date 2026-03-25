"""Module execution CLI commands: summarize, classify, extract, generate, moderate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from aiforge.cli._helpers import (
    console,
    format_output,
    print_metrics,
    read_input,
    response_to_dict,
)

# ---------------------------------------------------------------------------
# aiforge summarize
# ---------------------------------------------------------------------------

summarize_app = typer.Typer(name="summarize", invoke_without_command=True)


@summarize_app.callback(invoke_without_command=True)
def summarize(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Input text."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read input from file."),
    style: str = typer.Option(
        "concise", "--style", "-s",
        help="Summary style: concise|detailed|bullet_points|executive.",
    ),
    max_length: int = typer.Option(500, "--max-length", help="Target summary length."),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", help="Chunking strategy: stuff|map_reduce (auto if omitted).",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Summarize text."""
    body = read_input(text, file)

    from aiforge import Summarizer

    s = Summarizer(style=style, max_length=max_length)
    resp = s.summarize(body, strategy=strategy)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)
    print_metrics(resp)


# ---------------------------------------------------------------------------
# aiforge classify
# ---------------------------------------------------------------------------

classify_app = typer.Typer(name="classify", invoke_without_command=True)


@classify_app.callback(invoke_without_command=True)
def classify(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Input text."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read input from file."),
    labels: str = typer.Option(
        ..., "--labels", "-l",
        help="Comma-separated labels (e.g. 'bug,feature,question').",
    ),
    multi_label: bool = typer.Option(False, "--multi-label", help="Allow multiple labels."),
    examples_file: Optional[Path] = typer.Option(
        None, "--examples-file", help="JSON file with few-shot examples [{text, label}].",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Classify text into labels."""
    body = read_input(text, file)
    label_list = [l.strip() for l in labels.split(",") if l.strip()]  # noqa: E741

    examples = None
    if examples_file:
        examples_raw = json.loads(examples_file.read_text(encoding="utf-8"))
        from aiforge.modules.classify.module import ClassifyExample

        examples = [ClassifyExample(**e) for e in examples_raw]

    from aiforge import Classifier

    c = Classifier(labels=label_list, multi_label=multi_label, examples=examples)
    resp = c.classify(body)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)
    print_metrics(resp)


# ---------------------------------------------------------------------------
# aiforge extract
# ---------------------------------------------------------------------------

extract_app = typer.Typer(name="extract", invoke_without_command=True)


@extract_app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Input text."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read input from file."),
    schema_file: Optional[Path] = typer.Option(
        None, "--schema-file", help="JSON Schema file defining output structure.",
    ),
    max_retries: int = typer.Option(3, "--max-retries", help="Validation retry attempts."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Extract structured data from text."""
    body = read_input(text, file)

    schema = None
    if schema_file:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))

    from aiforge import Extractor

    e = Extractor(output_schema=schema, max_retries=max_retries)
    resp = e.extract(body)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)
    print_metrics(resp)


# ---------------------------------------------------------------------------
# aiforge generate
# ---------------------------------------------------------------------------

generate_app = typer.Typer(name="generate", invoke_without_command=True)


@generate_app.callback(invoke_without_command=True)
def generate(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt or template text."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read prompt from file."),
    var: Optional[list[str]] = typer.Option(
        None, "--var", help="Template variable as key=value (repeatable).",
    ),
    guardrails: Optional[str] = typer.Option(
        None, "--guardrails", "-g", help="Comma-separated guardrails (e.g. 'no_pii,professional_tone').",
    ),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max output tokens."),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Generate text from a prompt or template."""
    body = read_input(prompt, file)

    variables: dict[str, str] = {}
    if var:
        for v in var:
            if "=" not in v:
                console.print(f"[red]Invalid --var '{v}'. Use key=value format.[/red]")
                raise typer.Exit(1)
            k, val = v.split("=", 1)
            variables[k.strip()] = val.strip()

    guard_list = None
    if guardrails:
        guard_list = [g.strip() for g in guardrails.split(",") if g.strip()]

    from aiforge import Generator

    g = Generator(
        template=body if variables else None,
        guardrails=guard_list,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    resp = g.generate(prompt=body if not variables else None, **variables)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)
    print_metrics(resp)


# ---------------------------------------------------------------------------
# aiforge moderate
# ---------------------------------------------------------------------------

moderate_app = typer.Typer(name="moderate", invoke_without_command=True)


@moderate_app.callback(invoke_without_command=True)
def moderate(
    ctx: typer.Context,
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Input text."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read input from file."),
    checks: str = typer.Option(
        "pii,toxicity,prompt_injection", "--checks", "-c",
        help="Comma-separated checks: pii,toxicity,prompt_injection.",
    ),
    pii_types: Optional[str] = typer.Option(
        None, "--pii-types", help="Comma-separated PII types: email,phone,ssn,credit_card.",
    ),
    redact: bool = typer.Option(False, "--redact", "-r", help="Redact PII instead of just flagging."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Check text for safety issues or redact PII."""
    body = read_input(text, file)
    check_list = [c.strip() for c in checks.split(",") if c.strip()]
    pii_list = [p.strip() for p in pii_types.split(",") if p.strip()] if pii_types else None

    from aiforge import Moderator

    m = Moderator(checks=check_list, pii_types=pii_list)

    if redact:
        resp = m.redact(body)
    else:
        resp = m.check(body)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)

    if not redact:
        print_metrics(resp)
