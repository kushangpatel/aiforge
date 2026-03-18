"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from aiforge.__version__ import __version__
from aiforge.infra.cost_tracker import CostTracker

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe — always returns 200 if the server is running."""
    return {"status": "ok", "version": __version__}


@router.get("/health/ready")
async def readiness() -> dict[str, Any]:
    """Readiness probe — checks that settings load without error."""
    from aiforge.infra.config import get_settings

    try:
        settings = get_settings()
        provider = settings.PROVIDER
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_ready", "detail": str(exc)}

    return {"status": "ready", "provider": provider, "version": __version__}


@router.get("/stats")
async def stats() -> dict[str, Any]:
    """Return session-level cost and token usage statistics."""
    tracker = CostTracker.get_session()
    return {
        "total_cost_usd": round(tracker.total_cost, 6),
        "total_tokens": tracker.total_tokens,
        "call_count": tracker.call_count,
        "by_module": dict(tracker.by_module),
        "by_model": dict(tracker.by_model),
    }
