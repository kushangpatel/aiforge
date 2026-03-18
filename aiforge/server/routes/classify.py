"""Classifier endpoint (/classify)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/classify", tags=["classify"])


@router.post("")
async def run_classify() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="Classifier module not yet implemented")
