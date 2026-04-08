"""Shared helpers for CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def read_input(
    text: str | None,
    file: Path | None,
    *,
    stdin_ok: bool = True,
) -> str:
    """Resolve input text from --text, --file, or stdin."""
    if text:
        return text
    if file:
        return file.read_text(encoding="utf-8")
    if stdin_ok and not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]Provide --text, --file, or pipe via stdin.[/red]")
    raise SystemExit(1)


def format_output(
    data: dict[str, Any] | list[Any],
    fmt: str,
    *,
    table_columns: list[tuple[str, str]] | None = None,
) -> None:
    """Print output in the requested format."""
    if fmt == "json":
        console.print_json(json.dumps(data, default=str))
    elif fmt == "yaml":
        try:
            import yaml

            console.print(yaml.dump(data, default_flow_style=False, sort_keys=False))
        except ImportError:
            console.print_json(json.dumps(data, default=str))
    elif fmt == "table" and table_columns:
        tbl = Table(show_header=True)
        for col_name, style in table_columns:
            tbl.add_column(col_name, style=style)
        if isinstance(data, list):
            for row in data:
                tbl.add_row(*(str(row.get(c, "")) for c, _ in table_columns))
        console.print(tbl)
    else:
        console.print_json(json.dumps(data, default=str))


def response_to_dict(resp: Any) -> dict[str, Any]:
    """Convert a ModuleResponse (Pydantic) to a JSON-serialisable dict."""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return dict(resp)  # type: ignore[arg-type]


def print_metrics(resp: Any) -> None:
    """Print cost / token / latency footer."""
    cost = getattr(resp, "cost", 0.0)
    inp = getattr(resp, "input_tokens", 0)
    out = getattr(resp, "output_tokens", 0)
    latency = getattr(resp, "latency_ms", 0.0)
    model = getattr(resp, "model", "")
    console.print(
        f"\n[dim]model={model}  tokens={inp}+{out}  cost=${cost:.4f}  latency={latency:.0f}ms[/dim]"
    )
