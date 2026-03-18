"""Generator endpoint (/generate)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("")
async def run_generate() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="Generator module not yet implemented")
