"""SemanticSearch endpoints (/search/index and /search/query)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
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


class _HitOut(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict = {}


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


@router.post("/index", response_model=IndexResponseOut)
async def run_index(body: IndexRequest) -> Any:
    """Embed and upsert documents into a vector store collection."""
    from aiforge.modules.search.module import SemanticSearch
    from aiforge.modules.search.schemas import SearchDocument

    docs = [
        SearchDocument(id=d.id, text=d.text, metadata=d.metadata)
        for d in body.documents
    ]
    try:
        ss = SemanticSearch(collection=body.collection)
        result = await ss.aindex(docs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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
            body.query, top_k=body.top_k, filter=body.filter
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return SearchResponseOut(
        hits=[
            _HitOut(id=h.id, text=h.text, score=h.score, metadata=h.metadata)
            for h in result.hits
        ],
        query=result.query,
        total_searched=result.total_searched,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        latency_ms=result.latency_ms,
    )

