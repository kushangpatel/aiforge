"""
AIForge FastAPI application.

Start with:
    aiforge serve                       # via CLI
    uvicorn aiforge.server.app:app      # direct
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aiforge.__version__ import __version__

_logger = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    from aiforge.infra.config import get_settings
    from aiforge.infra.logging import configure_logging
    from aiforge.infra.tracing import setup_tracing

    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)
    setup_tracing(settings)
    _logger.info("AIForge server starting", version=__version__, provider=settings.PROVIDER)
    yield
    _logger.info("AIForge server stopped")


def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI app."""
    from aiforge.infra.config import get_settings
    from aiforge.server.middleware import AuthMiddleware, RequestLoggingMiddleware
    from aiforge.server.routes import (
        chat,
        classify,
        extract,
        generate,
        health,
        jobs,
        moderate,
        pipeline,
        qa,
        search,
        summarize,
        translate,
        vision,
    )

    settings = get_settings()

    application = FastAPI(
        title="AIForge API",
        description="Plug-and-play AI toolkit REST API",
        version=__version__,
        lifespan=_lifespan,
    )

    # CORS — open by default for dev; tighten in production via AIFORGE_CORS_ORIGINS
    cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    allow_credentials = "*" not in cors_origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)
    application.add_middleware(AuthMiddleware, api_key=settings.API_KEY)

    # Register routes
    application.include_router(health.router)
    application.include_router(summarize.router)
    application.include_router(qa.router)
    application.include_router(extract.router)
    application.include_router(classify.router)
    application.include_router(search.router)
    application.include_router(generate.router)
    application.include_router(chat.router)
    application.include_router(moderate.router)
    application.include_router(translate.router)
    application.include_router(vision.router)
    application.include_router(pipeline.router)
    application.include_router(jobs.router)

    @application.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        _logger.error("Unhandled exception", exc_type=type(exc).__name__, detail=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


# Module-level app instance (used by uvicorn)
app: FastAPI = create_app()
