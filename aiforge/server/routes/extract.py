"""Extractor endpoint (/extract)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/extract", tags=["extract"])


@router.post("")
async def run_extract() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="Extractor module not yet implemented")
