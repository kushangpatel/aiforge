"""
FastAPI middleware for the AIForge server.

Provides:
- Bearer-token authentication (when AIFORGE_API_KEY is set)
- Request ID injection
- Structured request/response logging
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    pass

_logger = structlog.get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token authentication middleware.

    Skips auth for the ``/health`` route so load-balancer probes always pass.
    Only active when ``AIFORGE_API_KEY`` is configured.
    """

    def __init__(self, app: "Any", api_key: str | None = None) -> None:  # noqa: ANN401
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Auth not configured — allow all requests
        if not self._api_key:
            return await call_next(request)

        # Health check is always public
        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )
        token = auth_header.removeprefix("Bearer ").strip()
        if token != self._api_key:
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Injects a request-id header and logs every request with latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        _logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 1),
            request_id=request_id,
        )
        response.headers["X-Request-Id"] = request_id
        return response
