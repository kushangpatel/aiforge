"""Q&A and Search CLI sub-commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from aiforge.cli._helpers import (
    console,
    format_output,
    print_metrics,
    response_to_dict,
)

# ---------------------------------------------------------------------------
# aiforge qa  (ingest | ask)
# ---------------------------------------------------------------------------

qa_app = typer.Typer(name="qa", help="Document Q&A: ingest documents and ask questions.")


@qa_app.command()
def ingest(
    source: str = typer.Argument(..., help="File path, directory, or git URL to ingest."),
    collection: str = typer.Option("default", "--collection", "-c", help="Target collection name."),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions to include (e.g. '.py,.md,.txt').",
    ),
    batch_size: int = typer.Option(100, "--batch-size", help="Embedding batch size."),
    mode: str = typer.Option(
        "full",
        "--mode",
        help="Ingest mode: full|incremental.",
        case_sensitive=False,
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Ingest documents into a collection for RAG queries."""
    ext_list = [e.strip() for e in extensions.split(",") if e.strip()] if extensions else None
    mode_norm = mode.strip().lower()
    if mode_norm not in {"full", "incremental"}:
        raise typer.BadParameter("--mode must be either 'full' or 'incremental'")

    from aiforge import DocQA

    qa = DocQA(collection=collection)
    if mode_norm == "incremental":
        resp = qa.ingest_incremental(source, file_extensions=ext_list, batch_size=batch_size)
    else:
        resp = qa.ingest(source, file_extensions=ext_list, batch_size=batch_size)

    data = response_to_dict(resp)
    format_output(data, fmt)
    console.print(
        f"\n[green]✓[/green] Indexed [bold]{resp.chunks_indexed}[/bold] chunks "
        f"from [bold]{resp.files_processed}[/bold] files into '{collection}'."
    )
    print_metrics(resp)


@qa_app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask."),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to search."),
    filter_repo: str | None = typer.Option(
        None,
        "--filter",
        help="Filter results by source path substring.",
    ),
    include_trace: bool = typer.Option(
        False,
        "--trace",
        help="Include agent reasoning trace in output.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Ask a question against ingested documents."""
    from aiforge import DocQA

    qa = DocQA(collection=collection)
    resp = qa.ask(query, filter_repo=filter_repo, include_trace=include_trace)

    data = response_to_dict(resp)
    if output:
        output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]✓[/green] Written to {output}")
    else:
        format_output(data, fmt)
    print_metrics(resp)


# ---------------------------------------------------------------------------
# aiforge search  (index | query)
# ---------------------------------------------------------------------------

search_app = typer.Typer(name="search", help="Semantic search: index documents and query.")


@search_app.command()
def index(
    source: str = typer.Argument(..., help="File path or directory to index."),
    collection: str = typer.Option("default", "--collection", "-c", help="Target collection."),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions (e.g. '.py,.md').",
    ),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml."),
) -> None:
    """Index files into a collection for semantic search."""
    ext_list = [e.strip() for e in extensions.split(",") if e.strip()] if extensions else None

    from aiforge import SemanticSearch

    ss = SemanticSearch(collection=collection)
    resp = ss.index_files(source, file_extensions=ext_list)

    data = response_to_dict(resp)
    format_output(data, fmt)
    console.print(
        f"\n[green]✓[/green] Indexed [bold]{resp.indexed_count}[/bold] documents "
        f"into '{collection}' (total: {resp.total_documents})."
    )
    print_metrics(resp)


@search_app.command()
def query(
    query_text: str = typer.Argument(..., help="Search query."),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection to search."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results (1-100)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write result to file."),
    fmt: str = typer.Option("json", "--format", help="Output format: json|yaml|table."),
) -> None:
    """Query a collection with semantic search."""
    from aiforge import SemanticSearch

    ss = SemanticSearch(collection=collection, top_k=top_k)
    resp = ss.query(query_text)

    if fmt == "table":
        tbl = Table(title=f"Search results for: {query_text}", show_header=True)
        tbl.add_column("#", style="dim", width=3)
        tbl.add_column("Score", justify="right", style="green")
        tbl.add_column("ID", style="cyan")
        tbl.add_column("Text", max_width=80)
        for i, hit in enumerate(resp.hits, 1):
            tbl.add_row(str(i), f"{hit.score:.4f}", hit.id, hit.text[:120])
        console.print(tbl)
    else:
        data = response_to_dict(resp)
        if output:
            output.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            console.print(f"[green]✓[/green] Written to {output}")
        else:
            format_output(data, fmt)
    print_metrics(resp)
