"""
Tracing setup: LangSmith, OpenTelemetry, console, or no-op modes.

Call ``setup_tracing(settings)`` once at startup.  A unique ``trace_id``
is generated per LLM call and included in every ``ModuleResponse``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings

_logger = structlog.get_logger(__name__)

_tracing_initialised = False


def setup_tracing(settings: AIForgeSettings) -> None:
    """Initialise tracing backend based on *settings*.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _tracing_initialised
    if _tracing_initialised or not settings.TRACING_ENABLED:
        return

    backend = settings.TRACING_BACKEND.lower()

    if backend == "langsmith":
        _setup_langsmith(settings)
    elif backend == "otel":
        _setup_otel(settings)
    elif backend == "console":
        _logger.info("Tracing: console mode (LangSmith / OTel disabled)")
    else:
        _logger.warning("Unknown tracing backend %r — tracing disabled", backend)

    _tracing_initialised = True


def _setup_langsmith(settings: AIForgeSettings) -> None:
    """Configure LangSmith tracing via environment variables."""
    import os

    if not settings.LANGSMITH_API_KEY:
        _logger.warning("AIFORGE_LANGSMITH_API_KEY not set — LangSmith tracing will not work")
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)

    _logger.info(
        "LangSmith tracing enabled",
        project=settings.LANGSMITH_PROJECT,
    )


def _setup_otel(settings: AIForgeSettings) -> None:
    """Configure OpenTelemetry tracing."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
    except ImportError:
        _logger.warning(
            "opentelemetry-sdk not installed. "
            'Run: pip install "aiforge-kit[observability]" to enable OTel tracing.'
        )
        return

    provider = TracerProvider()

    if settings.OTEL_EXPORTER_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            _logger.warning("OTLP exporter not available; using no-op exporter.")

    trace.set_tracer_provider(provider)
    _logger.info("OpenTelemetry tracing enabled", endpoint=settings.OTEL_EXPORTER_ENDPOINT)


def reset_tracing() -> None:
    """Reset the tracing initialisation flag (tests only)."""
    global _tracing_initialised
    _tracing_initialised = False


def new_trace_id() -> str:
    """Generate a new random trace ID."""
    return str(uuid.uuid4())


def get_tracer(name: str = "aiforge") -> Any:  # noqa: ANN401
    """Return the OpenTelemetry tracer (or a no-op if OTel not configured)."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpTracer:
    """Minimal no-op tracer used when OpenTelemetry is not installed."""

    class _NoOpSpan:
        def __enter__(self) -> _NoOpTracer._NoOpSpan:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def set_attribute(self, *args: Any) -> None:
            pass

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpTracer._NoOpSpan:  # noqa: ARG002
        return self._NoOpSpan()
