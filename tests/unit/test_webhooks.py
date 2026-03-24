"""Unit tests for aiforge.server.webhooks — Webhook Support."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.server.webhooks import (
    JobRecord,
    JobStatus,
    clear_jobs,
    complete_job,
    create_job,
    fail_job,
    get_job,
    mark_running,
    validate_webhook_url,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _clean_jobs() -> Any:
    """Reset the in-memory job store before and after each test."""
    clear_jobs()
    yield
    clear_jobs()


# ======================================================================
# Job Store
# ======================================================================


class TestCreateJob:
    def test_returns_record_with_uuid_and_accepted_status(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")
        assert isinstance(job, JobRecord)
        assert len(job.job_id) == 36  # UUID format
        assert job.status == JobStatus.ACCEPTED
        assert job.webhook_url == "https://example.com/hook"

    def test_job_has_created_at(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")
        assert job.created_at is not None

    def test_job_has_no_result_initially(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")
        assert job.result is None
        assert job.error is None
        assert job.completed_at is None


class TestGetJob:
    def test_found(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")
        fetched = get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    def test_not_found_returns_none(self) -> None:
        assert get_job("nonexistent-id") is None


class TestCompleteJob:
    @pytest.mark.asyncio
    async def test_updates_status_and_result(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")

        with patch("aiforge.server.webhooks._deliver_webhook", new_callable=AsyncMock):
            await complete_job(job.job_id, {"indexed_count": 50})

        updated = get_job(job.job_id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result == {"indexed_count": 50}
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_missing_job_is_noop(self) -> None:
        # Should not raise
        await complete_job("nonexistent", {"x": 1})

    @pytest.mark.asyncio
    async def test_fires_webhook_on_complete(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")

        with patch("aiforge.server.webhooks._deliver_webhook", new_callable=AsyncMock) as mock_deliver:
            await complete_job(job.job_id, {"indexed_count": 50})
            mock_deliver.assert_awaited_once()
            delivered_job = mock_deliver.call_args[0][0]
            assert delivered_job.job_id == job.job_id


class TestFailJob:
    @pytest.mark.asyncio
    async def test_updates_status_and_error(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")

        with patch("aiforge.server.webhooks._deliver_webhook", new_callable=AsyncMock):
            await fail_job(job.job_id, "Something went wrong")

        updated = get_job(job.job_id)
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.error == "Something went wrong"
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_missing_job_is_noop(self) -> None:
        await fail_job("nonexistent", "error")


# ======================================================================
# Webhook Delivery
# ======================================================================


class TestDeliverWebhook:
    @pytest.mark.asyncio
    async def test_posts_correct_payload(self) -> None:
        from aiforge.server.webhooks import _deliver_webhook

        job = create_job(webhook_url="https://example.com/hook")
        job.status = JobStatus.COMPLETED
        job.result = {"indexed_count": 50}
        from datetime import datetime, timezone

        job.completed_at = datetime.now(timezone.utc)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            await _deliver_webhook(job)

            mock_client_instance.post.assert_awaited_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == "https://example.com/hook"
            payload = call_args[1]["json"]
            assert payload["job_id"] == job.job_id
            assert payload["status"] == "completed"
            assert payload["result"] == {"indexed_count": 50}

    @pytest.mark.asyncio
    async def test_delivery_failure_logged_not_raised(self) -> None:
        from aiforge.server.webhooks import _deliver_webhook

        job = create_job(webhook_url="https://example.com/hook")
        job.status = JobStatus.FAILED
        job.error = "boom"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = Exception("connection refused")
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            # Must not raise
            await _deliver_webhook(job)


# ======================================================================
# URL Validation / SSRF Protection
# ======================================================================


class TestValidateWebhookUrl:
    def test_valid_https_url(self) -> None:
        url = validate_webhook_url("https://example.com/webhook")
        assert url == "https://example.com/webhook"

    def test_http_rejected_in_production(self) -> None:
        with pytest.raises(ValueError, match="https://"):
            validate_webhook_url("http://example.com/webhook")

    def test_http_allowed_when_insecure(self) -> None:
        url = validate_webhook_url("http://example.com/webhook", allow_insecure=True)
        assert url == "http://example.com/webhook"

    def test_localhost_rejected(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            validate_webhook_url("https://localhost/webhook")

    def test_127_0_0_1_rejected(self) -> None:
        with pytest.raises(ValueError, match="localhost"):
            validate_webhook_url("https://127.0.0.1/webhook")

    def test_private_ip_10_rejected(self) -> None:
        with pytest.raises(ValueError, match="private IP"):
            validate_webhook_url("https://10.0.0.1/webhook")

    def test_private_ip_172_rejected(self) -> None:
        with pytest.raises(ValueError, match="private IP"):
            validate_webhook_url("https://172.16.0.1/webhook")

    def test_private_ip_192_rejected(self) -> None:
        with pytest.raises(ValueError, match="private IP"):
            validate_webhook_url("https://192.168.1.1/webhook")

    def test_invalid_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            validate_webhook_url("ftp://example.com/webhook", allow_insecure=True)

    def test_public_ip_allowed(self) -> None:
        url = validate_webhook_url("https://8.8.8.8/webhook")
        assert url == "https://8.8.8.8/webhook"

    def test_hostname_with_port_allowed(self) -> None:
        url = validate_webhook_url("https://example.com:8443/webhook")
        assert url == "https://example.com:8443/webhook"


# ======================================================================
# Job Status Endpoint
# ======================================================================


class TestJobStatusEndpoint:
    @pytest.mark.asyncio
    async def test_returns_record(self) -> None:
        from aiforge.server.routes.jobs import check_job_status

        job = create_job(webhook_url="https://example.com/hook")

        result = await check_job_status(job.job_id)
        assert result.job_id == job.job_id
        assert result.status == "accepted"

    @pytest.mark.asyncio
    async def test_404_for_missing_job(self) -> None:
        from fastapi import HTTPException

        from aiforge.server.routes.jobs import check_job_status

        with pytest.raises(HTTPException) as exc_info:
            await check_job_status("nonexistent-id")
        assert exc_info.value.status_code == 404


# ======================================================================
# Config Keys
# ======================================================================


class TestWebhookConfig:
    def test_default_values(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            s = AIForgeSettings()
            assert s.WEBHOOK_ENABLED is True
            assert s.WEBHOOK_TIMEOUT == 10.0

    def test_override_values(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            s = AIForgeSettings(WEBHOOK_ENABLED=False, WEBHOOK_TIMEOUT=30.0)
            assert s.WEBHOOK_ENABLED is False
            assert s.WEBHOOK_TIMEOUT == 30.0

    def test_allow_insecure_default_false(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}):
            from aiforge.infra.config import AIForgeSettings

            s = AIForgeSettings()
            assert s.WEBHOOK_ALLOW_INSECURE is False


# ======================================================================
# mark_running
# ======================================================================


class TestMarkRunning:
    def test_transitions_accepted_to_running(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")
        assert job.status == JobStatus.ACCEPTED
        mark_running(job.job_id)
        assert job.status == JobStatus.RUNNING

    def test_noop_if_not_accepted(self) -> None:
        """Only ACCEPTED jobs should transition to RUNNING."""
        job = create_job(webhook_url="https://example.com/hook")
        job.status = JobStatus.COMPLETED
        mark_running(job.job_id)
        assert job.status == JobStatus.COMPLETED

    def test_noop_for_missing_job(self) -> None:
        mark_running("nonexistent")  # should not raise


# ======================================================================
# Timeout propagation
# ======================================================================


class TestTimeoutPropagation:
    @pytest.mark.asyncio
    async def test_complete_job_passes_timeout(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")

        with patch("aiforge.server.webhooks._deliver_webhook", new_callable=AsyncMock) as mock_deliver:
            await complete_job(job.job_id, {"x": 1}, timeout=25.0)
            mock_deliver.assert_awaited_once()
            assert mock_deliver.call_args[1]["timeout"] == 25.0

    @pytest.mark.asyncio
    async def test_fail_job_passes_timeout(self) -> None:
        job = create_job(webhook_url="https://example.com/hook")

        with patch("aiforge.server.webhooks._deliver_webhook", new_callable=AsyncMock) as mock_deliver:
            await fail_job(job.job_id, "err", timeout=15.0)
            mock_deliver.assert_awaited_once()
            assert mock_deliver.call_args[1]["timeout"] == 15.0
