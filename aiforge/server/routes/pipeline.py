"""Pipeline endpoint (/pipeline)."""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelineRunRequest(BaseModel):
    pipeline_id: str
    text: str
    context: dict[str, Any] | None = None


@router.post("")
async def run_pipeline(body: PipelineRunRequest) -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="Pipeline endpoint not yet implemented")
