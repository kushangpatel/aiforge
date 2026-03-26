"""Vision endpoint (/vision)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/vision", tags=["vision"])


class VisionRequest(BaseModel):
    source: str = Field(description="Image URL or base64-encoded data URI.")
    task: str = Field(default="describe", pattern="^(describe|qa|extract|classify)$")
    question: str | None = None
    focus: str | None = None
    labels: list[str] = Field(default_factory=list)


class VisionRouteResponse(BaseModel):
    text: str = ""
    task: str = ""
    image_source: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@router.post("", response_model=VisionRouteResponse)
async def run_vision(body: VisionRequest) -> Any:
    """Analyze an image using a vision-capable model."""
    from aiforge.modules.vision.module import Vision

    try:
        v = Vision(task=body.task)
        result = await v.aanalyze(
            source=body.source,
            question=body.question,
            focus=body.focus,
            labels=body.labels or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return VisionRouteResponse(
        text=result.text,
        task=result.task,
        image_source=result.image_source,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )
