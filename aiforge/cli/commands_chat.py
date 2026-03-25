"""Interactive chat CLI command."""

from __future__ import annotations

from typing import Optional

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from aiforge.cli._helpers import console, print_metrics

chat_app = typer.Typer(name="chat", help="Interactive chat with an AI agent.")


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    message: Optional[str] = typer.Argument(None, help="Single message (omit for interactive mode)."),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", help="System prompt for the conversation.",
    ),
    memory: str = typer.Option(
        "buffer", "--memory", "-m", help="Memory mode: buffer|summary|none.",
    ),
    memory_k: int = typer.Option(10, "--memory-k", help="Messages to keep in memory."),
) -> None:
    """Chat with an AI agent.

    If a message is provided it sends a single turn and exits.
    Without a message it starts an interactive REPL loop.
    """
    if ctx.resilient_parsing:
        return

    from aiforge import ChatAgent

    agent = ChatAgent(
        system_prompt=system_prompt,
        memory=memory if memory != "none" else None,
        memory_k=memory_k,
    )

    # ── Single-turn mode ──────────────────────────────────────────────
    if message:
        resp = agent.chat(message)
        console.print(Markdown(resp.message))
        print_metrics(resp)
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
