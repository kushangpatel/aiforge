"""SemanticSearch endpoints (/search/index and /search/query)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/search", tags=["search"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class _DocumentIn(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class IndexRequest(BaseModel):
    documents: list[_DocumentIn]
    collection: str = "default"
    webhook_url: str | None = None

    @field_validator("documents")
    @classmethod
    def _validate_documents(cls, v: list) -> list:
        if not v:
            raise ValueError("documents must not be empty")
        return v


class IndexResponseOut(BaseModel):
    indexed_count: int
    collection: str
    total_documents: int
    cost_usd: float = 0.0
    input_tokens: int = 0
    latency_ms: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=100)
    filter: dict | None = None
    include_embeddings: bool = False


class _HitOut(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict = {}
    embeddings: list[float] | None = None


class SearchResponseOut(BaseModel):
    hits: list[_HitOut]
    query: str
    total_searched: int
    cost_usd: float = 0.0
    input_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/index")
async def run_index(body: IndexRequest, background_tasks: BackgroundTasks) -> Any:
    """Embed and upsert documents into a vector store collection.

    If ``webhook_url`` is provided, returns 202 Accepted and delivers the result
    asynchronously via HTTP POST.  Otherwise behaves synchronously.
    """
    from aiforge.modules.search.module import SemanticSearch
    from aiforge.modules.search.schemas import SearchDocument

    docs = [
        SearchDocument(id=d.id, text=d.text, metadata=d.metadata)
        for d in body.documents
    ]

    if body.webhook_url:
        from aiforge.infra.config import get_settings
        from aiforge.server.webhooks import (
            complete_job,
            create_job,
            fail_job,
            validate_webhook_url,
        )

        settings = get_settings()
        if not settings.WEBHOOK_ENABLED:
            raise HTTPException(status_code=422, detail="Webhook support is disabled")

        try:
            validate_webhook_url(body.webhook_url, allow_insecure=settings.WEBHOOK_ALLOW_INSECURE)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        job = create_job(webhook_url=body.webhook_url)
        webhook_timeout = settings.WEBHOOK_TIMEOUT

        async def _process() -> None:
            from aiforge.server.webhooks import mark_running

            mark_running(job.job_id)
            try:
                ss = SemanticSearch(collection=body.collection)
                result = await ss.aindex(docs)
                await complete_job(job.job_id, result.model_dump(), timeout=webhook_timeout)
            except Exception as exc:
                await fail_job(job.job_id, str(exc), timeout=webhook_timeout)

        background_tasks.add_task(_process)
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": "accepted"},
        )

    try:
        ss = SemanticSearch(collection=body.collection)
        result = await ss.aindex(docs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return IndexResponseOut(
        indexed_count=result.indexed_count,
        collection=result.collection,
        total_documents=result.total_documents,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        latency_ms=result.latency_ms,
    )


@router.post("/query", response_model=SearchResponseOut)
async def run_query(body: QueryRequest) -> Any:
    """Retrieve documents semantically similar to the query."""
    from aiforge.modules.search.module import SemanticSearch

    try:
        ss = SemanticSearch(collection=body.collection)
        result = await ss.aquery(
            body.query,
            top_k=body.top_k,
            filter=body.filter,
            include_embeddings=body.include_embeddings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return SearchResponseOut(
        hits=[
            _HitOut(
                id=h.id,
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                embeddings=h.embeddings,
            )
            for h in result.hits
        ],
        query=result.query,
        total_searched=result.total_searched,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        latency_ms=result.latency_ms,
    )

