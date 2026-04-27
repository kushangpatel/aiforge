"""Interactive chat CLI command."""

from __future__ import annotations

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from aiforge.cli._helpers import console, print_metrics

chat_app = typer.Typer(name="chat", help="Interactive chat with an AI agent.")


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    message: str | None = typer.Argument(None, help="Single message (omit for interactive mode)."),
    system_prompt: str | None = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the conversation.",
    ),
    memory: str = typer.Option(
        "buffer",
        "--memory",
        "-m",
        help="Memory mode: buffer|summary|none.",
    ),
    memory_k: int = typer.Option(10, "--memory-k", help="Messages to keep in memory."),
    mcp_config: str | None = typer.Option(
        None,
        "--mcp-config",
        help="Path to JSON/YAML file with MCP server configs.",
    ),
) -> None:
    """Chat with an AI agent.

    If a message is provided it sends a single turn and exits.
    Without a message it starts an interactive REPL loop.
    """
    if ctx.resilient_parsing:
        return

    from aiforge import ChatAgent

    mcp_servers = None
    if mcp_config:
        import json
        from pathlib import Path

        path = Path(mcp_config)
        if not path.exists():
            console.print(f"[red]MCP config file not found: {mcp_config}[/red]")
            raise typer.Exit(1)
        raw = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            mcp_servers = yaml.safe_load(raw)
        else:
            mcp_servers = json.loads(raw)

    agent = ChatAgent(
        system_prompt=system_prompt,
        memory=memory or "buffer",
        memory_k=memory_k,
        mcp_servers=mcp_servers,
    )

    # ── Single-turn mode ──────────────────────────────────────────────
    if message:
        try:
            resp = agent.chat(message)
            console.print(Markdown(resp.message))
            print_metrics(resp)
        finally:
            agent.close()
        return

    # ── Interactive REPL ──────────────────────────────────────────────
    console.print(
        Panel(
            "[bold]AIForge Chat[/bold]\n"
            "Type your message and press Enter.  Commands:\n"
            "  [dim]/reset[/dim]  — clear conversation history\n"
            "  [dim]/quit[/dim]   — exit chat",
            border_style="blue",
        )
    )

    try:
        while True:
            try:
                user_input = console.input("[bold blue]you >[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]bye![/dim]")
                break
            if user_input.lower() == "/reset":
                agent.reset()
                console.print("[yellow]conversation reset[/yellow]")
                continue

            resp = agent.chat(user_input)
            console.print()
            console.print(Markdown(resp.message))
            print_metrics(resp)
            console.print()
    finally:
        agent.close()
