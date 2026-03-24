"""Webhook dispatcher and in-memory job store.

Long-running operations (document indexing, ingestion) can return 202 Accepted
and POST the result to a caller-supplied ``webhook_url`` when finished.
"""

from __future__ import annotations

import ipaddress
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.ACCEPTED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    webhook_url: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
]


def validate_webhook_url(url: str, *, allow_insecure: bool = False) -> str:
    """Validate a webhook URL.

    In production (``allow_insecure=False``):
    - Only ``https://`` URLs are accepted.
    - ``localhost``, ``127.0.0.1``, and RFC-1918 private IP ranges are rejected.

    Returns the validated URL or raises ``ValueError``.
    """
    parsed = urlparse(url)

    if not allow_insecure and parsed.scheme != "https":
        raise ValueError(
            f"Webhook URL must use https:// in production (got {parsed.scheme}://)"
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Webhook URL scheme must be http or https, got {parsed.scheme!r}")

    hostname = parsed.hostname or ""

    # Block localhost variants
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise ValueError(f"Webhook URL must not target localhost ({hostname})")

    # Block private IPs
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _PRIVATE_NETS:
            if addr in net:
                raise ValueError(
                    f"Webhook URL must not target private IP ranges ({hostname})"
                )
    except ValueError as exc:
        # Re-raise our own validation errors; swallow parse errors for hostnames
        if "Webhook URL" in str(exc):
            raise
    return url


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, JobRecord] = {}


def create_job(webhook_url: str) -> JobRecord:
    """Register a new job and return its record."""
    job = JobRecord(webhook_url=webhook_url)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> JobRecord | None:
    """Look up a job by ID."""
    return _jobs.get(job_id)


async def complete_job(job_id: str, result: dict[str, Any], *, timeout: float = 10.0) -> None:
    """Mark a job as completed and fire the webhook."""
    job = _jobs.get(job_id)
    if not job:
        return
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.now(timezone.utc)
    job.result = result
    if job.webhook_url:
        await _deliver_webhook(job, timeout=timeout)


async def fail_job(job_id: str, error: str, *, timeout: float = 10.0) -> None:
    """Mark a job as failed and fire the webhook."""
    job = _jobs.get(job_id)
    if not job:
        return
    job.status = JobStatus.FAILED
    job.completed_at = datetime.now(timezone.utc)
    job.error = error
    if job.webhook_url:
        await _deliver_webhook(job, timeout=timeout)


async def _deliver_webhook(job: JobRecord, *, timeout: float = 10.0) -> None:
    """POST the job result to the webhook URL.

    Single attempt with configurable timeout.  Failures are logged, not raised.
    """
    import httpx

    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "result": job.result,
        "error": job.error,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(str(job.webhook_url), json=payload)
            logger.info(
                "webhook.delivered",
                job_id=job.job_id,
                webhook_url=job.webhook_url,
                status_code=resp.status_code,
            )
    except Exception as exc:
        logger.error(
            "webhook.delivery_failed",
            job_id=job.job_id,
            webhook_url=job.webhook_url,
            error=str(exc),
        )


def mark_running(job_id: str) -> None:
    """Transition a job to RUNNING status."""
    job = _jobs.get(job_id)
    if job and job.status == JobStatus.ACCEPTED:
        job.status = JobStatus.RUNNING


def clear_jobs() -> None:
    """Clear all jobs from the in-memory store (testing helper)."""
    _jobs.clear()
