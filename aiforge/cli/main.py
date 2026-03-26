"""
AIForge CLI entry point.

Commands:
    aiforge init        — scaffold aiforge.yaml + .env.example
    aiforge serve       — start FastAPI server
    aiforge playground  — start Streamlit UI
    aiforge migrate     — migrate / inspect vector store collections
    aiforge summarize   — summarise text
    aiforge classify    — classify text into labels
    aiforge extract     — extract structured data from text
    aiforge generate    — generate text from a prompt / template
    aiforge rewrite     — rewrite text according to instructions
    aiforge moderate    — content moderation & PII redaction
    aiforge translate   — translate text between languages
    aiforge vision      — analyze images with vision models
    aiforge sentiment   — analyze text sentiment
    aiforge qa          — document Q&A (ingest + ask)
    aiforge search      — semantic search (index + query)
    aiforge chat        — interactive AI chat agent
    aiforge config      — show resolved configuration
    aiforge models      — list models & estimate costs
    aiforge registry    — explore registered components
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from aiforge.cli.commands_chat import chat_app
from aiforge.cli.commands_info import config_app, models_app, registry_app
from aiforge.cli.commands_modules import (
    classify_app,
    extract_app,
    generate_app,
    moderate_app,
    rewrite_app,
    sentiment_app,
    summarize_app,
    translate_app,
    vision_app,
)
from aiforge.cli.commands_rag import qa_app, search_app

app = typer.Typer(
    name="aiforge",
    help="AIForge — plug-and-play AI toolkit",
    add_completion=False,
)
console = Console()

# ── Register sub-apps ─────────────────────────────────────────────────────
app.add_typer(summarize_app, name="summarize", help="Summarise text.")
app.add_typer(classify_app, name="classify", help="Classify text into labels.")
app.add_typer(extract_app, name="extract", help="Extract structured data from text.")
app.add_typer(generate_app, name="generate", help="Generate text from a prompt / template.")
app.add_typer(moderate_app, name="moderate", help="Content moderation & PII redaction.")
app.add_typer(translate_app, name="translate", help="Translate text between languages.")
app.add_typer(vision_app, name="vision", help="Analyze images with vision models.")
app.add_typer(rewrite_app, name="rewrite", help="Rewrite text according to instructions.")
app.add_typer(sentiment_app, name="sentiment", help="Analyze text sentiment.")
app.add_typer(qa_app, name="qa", help="Document Q&A: ingest documents and ask questions.")
app.add_typer(search_app, name="search", help="Semantic search: index documents and query.")
app.add_typer(chat_app, name="chat", help="Interactive AI chat agent.")
app.add_typer(config_app, name="config", help="Show resolved configuration.")
app.add_typer(models_app, name="models", help="List models & estimate costs.")
app.add_typer(registry_app, name="registry", help="Explore registered components.")


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config files."
    ),
) -> None:
    """Scaffold aiforge.yaml + .env.example in the current directory."""
    _copy_template("aiforge.yaml.example", "aiforge.yaml", force=force)
    _copy_template(".env.example", ".env", force=force)
    console.print(
        "[green]✓[/green] Config files created. "
        "Edit [bold]aiforge.yaml[/bold] and [bold].env[/bold] to get started."
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),  # noqa: S104
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Enable hot-reload (dev mode)."),
) -> None:
    """Start the FastAPI REST API server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed.[/red] Run: pip install \"aiforge[server]\""
        )
        raise typer.Exit(1) from None

    console.print(f"[cyan]Starting AIForge API server on http://{host}:{port}[/cyan]")
    console.print("[dim]Swagger docs: http://{}:{}/docs[/dim]".format(host, port))
    uvicorn.run(
        "aiforge.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def playground(
    port: int = typer.Option(8501, help="Streamlit port."),
) -> None:
    """Start the Streamlit interactive playground."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[red]Streamlit not installed.[/red] Run: pip install \"aiforge[playground]\""
        )
        raise typer.Exit(1) from None

    playground_path = Path(__file__).parent.parent / "playground" / "app.py"
    console.print(f"[cyan]Starting AIForge Playground on http://localhost:{port}[/cyan]")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(playground_path),
            "--server.port",
            str(port),
        ],
        check=False,
    )


@app.command()
def migrate(
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Specific collection to inspect."
    ),
    all: bool = typer.Option(False, "--all", "-a", help="List all collections."),
    status: bool = typer.Option(False, "--status", "-s", help="Show collection stats."),
    delete: Optional[str] = typer.Option(
        None, "--delete", "-d", help="Delete a named collection."
    ),
) -> None:
    """Inspect or manage vector store collections."""
    from aiforge.core.vectorstore import ChromaVectorStore
    from aiforge.infra.config import get_settings

    settings = get_settings()
    store = ChromaVectorStore(persist_dir=settings.CHROMA_PERSIST_DIR)

    if delete:
        if typer.confirm(f"Delete collection '{delete}'? This cannot be undone."):
            store.delete_collection(delete)
            console.print(f"[green]✓[/green] Deleted collection '{delete}'.")
        return

    if not (all or status or collection):
        console.print("[yellow]No action specified.[/yellow] Use --all, --status, --collection, or --delete.")
        raise typer.Exit()

    if all or status:
        collections = store.list_collections()
        if not collections:
            console.print("[yellow]No collections found.[/yellow]")
            return

        table = Table(title="ChromaDB Collections", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Chunks", justify="right")

        for cname in collections:
            if collection and cname != collection:
                continue
            try:
                stats = store.collection_stats(cname)
                table.add_row(cname, str(stats.get("chunks", "?")))
            except Exception:  # noqa: BLE001
                table.add_row(cname, "[red]error[/red]")

        console.print(table)


def _copy_template(src_name: str, dest_name: str, force: bool) -> None:
    """Copy a template file from the package root to cwd."""
    # Template lives next to pyproject.toml (package root)
    pkg_root = Path(__file__).parent.parent.parent
    src = pkg_root / src_name
    dest = Path.cwd() / dest_name

    if dest.exists() and not force:
        console.print(f"[yellow]Skipping[/yellow] {dest_name} (already exists — use --force to overwrite)")
        return

    if src.exists():
        shutil.copy2(src, dest)
        console.print(f"[green]✓[/green] Created {dest_name}")
    else:
        console.print(f"[yellow]Template {src_name!r} not found in package root.[/yellow]")


if __name__ == "__main__":
    app()
