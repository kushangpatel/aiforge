"""DocQA endpoints (/qa)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/qa", tags=["qa"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, description="File path, directory, or git URL")
    collection: str = "default"
    file_extensions: list[str] | None = None
    batch_size: int = Field(default=100, ge=1)


class IngestResponseOut(BaseModel):
    chunks_indexed: int
    files_processed: int
    languages: dict[str, int]
    skipped_files: list[str]
    collection: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="Question to answer")
    collection: str = "default"
    filter_repo: str | None = None
    max_iterations: int = Field(default=3, ge=1, le=10)
    include_trace: bool = False


class AskResponseOut(BaseModel):
    text: str
    sources: list[str]
    confidence: float
    iterations: int
    trace: list[dict[str, Any]] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0


class CollectionsOut(BaseModel):
    collections: list[str]


class CollectionStatsOut(BaseModel):
    name: str
    chunks: int = 0
    persist_dir: str = ""


class DeletedOut(BaseModel):
    status: str = "deleted"
    collection: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponseOut)
async def ingest(body: IngestRequest) -> Any:
    """Ingest documents into a collection."""
    from aiforge.modules.qa.module import DocQA

    try:
        qa = DocQA(collection=body.collection)
        result = await qa.aingest(
            body.source,
            file_extensions=body.file_extensions,
            batch_size=body.batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return IngestResponseOut(
        chunks_indexed=result.chunks_indexed,
        files_processed=result.files_processed,
        languages=result.languages,
        skipped_files=result.skipped_files,
        collection=result.collection,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/ask", response_model=AskResponseOut)
async def ask(body: AskRequest) -> Any:
    """Ask a question against an ingested collection."""
    from aiforge.core.types import AIForgeError
    from aiforge.modules.qa.module import DocQA

    try:
        qa = DocQA(
            collection=body.collection,
            max_iterations=body.max_iterations,
        )
        result = await qa.aask(
            body.query,
            filter_repo=body.filter_repo,
            include_trace=body.include_trace,
        )
    except AIForgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return AskResponseOut(
        text=result.text,
        sources=result.sources,
        confidence=result.confidence,
        iterations=result.iterations,
        trace=result.trace,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost,
        model=result.model,
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
    )


@router.get("/collections", response_model=CollectionsOut)
async def list_collections() -> Any:
    """List all collections."""
    from aiforge.modules.qa.module import DocQA

    qa = DocQA()
    return CollectionsOut(collections=qa.list_collections())


@router.get("/collections/{name}/stats", response_model=CollectionStatsOut)
async def collection_stats(name: str) -> Any:
    """Get stats for a collection."""
    from aiforge.modules.qa.module import DocQA

    qa = DocQA()
    stats = qa.collection_stats(name)
    return CollectionStatsOut(
        name=name,
        chunks=stats.get("chunks", 0),
        persist_dir=stats.get("persist_dir", ""),
    )


@router.delete("/collections/{name}", response_model=DeletedOut)
async def delete_collection(name: str) -> Any:
    """Delete a collection."""
    from aiforge.modules.qa.module import DocQA

    qa = DocQA()
    qa.delete_collection(name)
    return DeletedOut(collection=name)
