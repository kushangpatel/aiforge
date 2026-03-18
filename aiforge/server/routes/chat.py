"""ChatAgent endpoint (/chat)."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def run_chat() -> dict:  # type: ignore[return]
    raise HTTPException(status_code=501, detail="ChatAgent module not yet implemented")
