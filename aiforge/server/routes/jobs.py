"""Job status endpoint for webhook-based async operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aiforge.server.webhooks import get_job

router = APIRouter(tags=["jobs"])


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
async def check_job_status(job_id: str) -> JobStatusOut:
    """Poll the status of an async job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusOut(
        job_id=job.job_id,
        status=job.status.value,
        result=job.result,
        error=job.error,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )
