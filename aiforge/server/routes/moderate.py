"""Moderator endpoint (/moderate)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/moderate", tags=["moderate"])


@router.post("")
async def run_moderate() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="Moderator module not yet implemented")
