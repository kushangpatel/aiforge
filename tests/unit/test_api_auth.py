"""Tests for Bearer-token authentication middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiforge.server.middleware import AuthMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRET = "test-secret-key-42"


def _make_app(api_key: str | None = None) -> TestClient:
    """Build a minimal FastAPI app with AuthMiddleware attached."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware, api_key=api_key)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/docs", include_in_schema=False)
    async def docs_page():
        return {"page": "docs"}

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_spec():
        return {"openapi": "3.0.0"}

    @app.get("/redoc", include_in_schema=False)
    async def redoc_page():
        return {"page": "redoc"}

    @app.post("/classify")
    async def classify():
        return {"label": "test"}

    @app.post("/summarize")
    async def summarize():
        return {"summary": "ok"}

    return TestClient(app)


# ---------------------------------------------------------------------------
# No key configured → all endpoints open
# ---------------------------------------------------------------------------


class TestNoKeyConfigured:
    """When api_key is empty/None, auth is disabled (local dev mode)."""

    @pytest.fixture()
    def client(self) -> TestClient:
        return _make_app(api_key=None)

    def test_protected_endpoint_open(self, client: TestClient) -> None:
        resp = client.post("/classify")
        assert resp.status_code == 200

    def test_health_open(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_empty_string_key_also_disables_auth(self) -> None:
        client = _make_app(api_key="")
        resp = client.post("/classify")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Key configured → auth enforced
# ---------------------------------------------------------------------------


class TestKeyConfigured:
    """When api_key is set, protected endpoints require valid Bearer token."""

    @pytest.fixture()
    def client(self) -> TestClient:
        return _make_app(api_key=_SECRET)

    # ---- Valid token ----

    def test_correct_bearer_token_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/classify",
            headers={"Authorization": f"Bearer {_SECRET}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"label": "test"}

    # ---- Missing / malformed header ----

    def test_missing_auth_header_returns_401(self, client: TestClient) -> None:
        resp = client.post("/classify")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_malformed_auth_header_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/classify",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_empty_bearer_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/classify",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    # ---- Wrong token ----

    def test_wrong_token_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/classify",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    # ---- Public paths always open ----

    def test_health_endpoint_always_open(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_endpoint_always_open(self, client: TestClient) -> None:
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_endpoint_always_open(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_redoc_endpoint_always_open(self, client: TestClient) -> None:
        resp = client.get("/redoc")
        assert resp.status_code == 200

    # ---- Auth required on different endpoints ----

    def test_summarize_also_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/summarize")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Security: constant-time comparison
# ---------------------------------------------------------------------------


class TestSecurityProperties:
    """Verify security-critical implementation details."""

    def test_uses_secrets_compare_digest(self) -> None:
        """AuthMiddleware must use secrets.compare_digest for timing safety."""
        import inspect

        from aiforge.server.middleware import AuthMiddleware

        source = inspect.getsource(AuthMiddleware)
        assert "secrets.compare_digest" in source
