"""SemanticSearch endpoint (/search)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def run_search() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="SemanticSearch module not yet implemented")
