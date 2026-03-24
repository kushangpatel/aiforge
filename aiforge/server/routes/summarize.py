"""Summarization endpoint (/summarize)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    text: str
    style: Literal["concise", "detailed", "bullet_points", "executive"] = "concise"
    max_length: int = Field(default=500, ge=1)
    strategy: Literal["stuff", "map_reduce"] | None = Field(
        default=None,
        description="Force a strategy, or null/omit for auto-detection",
    )


class SummarizeRouteResponse(BaseModel):
    summary: str = ""
    strategy_used: Literal["stuff", "map_reduce"] = "stuff"
    chunks_processed: int = 1
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@router.post("", response_model=SummarizeRouteResponse)
async def summarize(body: SummarizeRequest) -> Any:
    """Summarize the provided text."""
    from aiforge.modules.summarize.module import Summarizer

    try:
        s = Summarizer(style=body.style, max_length=body.max_length)
        result = await s.asummarize(body.text, strategy=body.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return SummarizeRouteResponse(
        summary=result.summary,
        strategy_used=result.strategy_used,
        chunks_processed=result.chunks_processed,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )
