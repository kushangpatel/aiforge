"""Summarization endpoint (/summarize)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    text: str
    style: str = "concise"
    max_length: int | None = None
    model: str | None = None


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: float


@router.post("", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest) -> Any:
    """Summarize the provided text."""
    try:
        from aiforge.modules.summarize.module import Summarizer
    except ImportError:
        raise HTTPException(status_code=501, detail="Summarizer module not yet available")

    kwargs: dict[str, Any] = {"style": body.style}
    if body.max_length:
        kwargs["max_length"] = body.max_length
    if body.model:
        kwargs["model"] = body.model

    module = Summarizer(**kwargs)
    result = await module.arun(body.text)

    return SummarizeResponse(
        summary=result.to_text(),
        model=result.model,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
