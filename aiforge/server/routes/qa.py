"""DocQA endpoint (/qa)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("")
async def run_qa() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="DocQA module not yet implemented")
