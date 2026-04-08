"""Discovery & diagnostic CLI commands (config, models, registry)."""

from __future__ import annotations

from typing import cast

import typer
from rich.table import Table

from aiforge.cli._helpers import console, format_output

# ---------------------------------------------------------------------------
# aiforge config
# ---------------------------------------------------------------------------

config_app = typer.Typer(name="config", help="Show resolved configuration.")


@config_app.callback(invoke_without_command=True)
def config_show(
    ctx: typer.Context,
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
    reveal: bool = typer.Option(False, "--reveal", help="Show secret values (keys/tokens)."),
) -> None:
    """Display the resolved AIForge configuration."""
    if ctx.resilient_parsing:
        return

    from aiforge.core.config import AIForgeSettings

    settings = AIForgeSettings()
    data = settings.model_dump()

    if not reveal:
        _mask_secrets(data)

    format_output(data, fmt)


def _mask_secrets(d: dict, _keys: set[str] | None = None) -> None:
    """Replace values whose key contains 'key', 'secret', or 'token' with '***'."""
    sensitive = _keys or {"key", "secret", "token", "password", "credential"}
    for k, v in d.items():
        if isinstance(v, dict):
            _mask_secrets(v, sensitive)
        elif isinstance(v, str) and any(s in k.lower() for s in sensitive):
            d[k] = "***"


# ---------------------------------------------------------------------------
# aiforge models  (list | estimate)
# ---------------------------------------------------------------------------

models_app = typer.Typer(name="models", help="List available models and estimate costs.")


@models_app.command("list")
def models_list(
    quality: str | None = typer.Option(
        None,
        "--quality",
        "-q",
        help="Filter by quality tier: fast|balanced|quality.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Filter by provider name.",
    ),
    fmt: str = typer.Option("table", "--format", help="Output format: json|yaml|table."),
) -> None:
    """List models known to the model router."""
    from aiforge.core.model_router import ModelRouter, QualityTier

    router = ModelRouter()
    models = router.list_models(quality=cast("QualityTier | None", quality), provider=provider)

    if fmt == "table":
        tbl = Table(title="Available Models", show_header=True)
        tbl.add_column("Model", style="cyan")
        tbl.add_column("Provider", style="green")
        tbl.add_column("Quality", style="yellow")
        tbl.add_column("Context", justify="right")
        tbl.add_column("Input $/1K", justify="right")
        tbl.add_column("Output $/1K", justify="right")
        for m in models:
            tbl.add_row(
                m.get("model", ""),
                m.get("provider", ""),
                m.get("quality", ""),
                str(m.get("context_window", "")),
                str(m.get("input_cost_per_1k", "")),
                str(m.get("output_cost_per_1k", "")),
            )
        console.print(tbl)
    else:
        format_output(models, fmt)


@models_app.command("estimate")
def models_estimate(
    model: str = typer.Argument(..., help="Model name (e.g. gpt-4o-mini)."),
    input_tokens: int = typer.Option(1000, "--input", "-i", help="Estimated input tokens."),
    output_tokens: int = typer.Option(500, "--output", "-o", help="Estimated output tokens."),
) -> None:
    """Estimate the cost of a model call."""
    from aiforge.core.model_router import ModelRouter

    router = ModelRouter()
    cost = router.estimate_cost(model, input_tokens, output_tokens)

    console.print(
        f"[bold]Model:[/bold]  {model}\n"
        f"[bold]Input:[/bold]  {input_tokens:,} tokens\n"
        f"[bold]Output:[/bold] {output_tokens:,} tokens\n"
        f"[bold]Cost:[/bold]   [green]${cost:.6f}[/green]"
    )


# ---------------------------------------------------------------------------
# aiforge registry
# ---------------------------------------------------------------------------

registry_app = typer.Typer(
    name="registry", help="Explore registered modules, providers, and stores."
)


@registry_app.callback(invoke_without_command=True)
def registry_list(
    ctx: typer.Context,
    fmt: str = typer.Option("table", "--format", help="Output format: json|yaml|table."),
) -> None:
    """Show all registered modules, providers, and vector stores."""
    if ctx.resilient_parsing:
        return

    from aiforge.core.registry import list_modules, list_providers, list_vectorstores

    data = {
        "modules": list_modules(),
        "providers": list_providers(),
        "vectorstores": list_vectorstores(),
    }

    if fmt == "table":
        tbl = Table(title="AIForge Registry", show_header=True)
        tbl.add_column("Category", style="bold cyan")
        tbl.add_column("Items", style="green")
        for cat, items in data.items():
            tbl.add_row(cat, ", ".join(items) if items else "[dim]none[/dim]")
        console.print(tbl)
    else:
        format_output(data, fmt)
