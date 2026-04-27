"""
Structured logging configuration using structlog.

Call ``configure_logging()`` once at startup (done automatically by
``get_settings()`` on first import of aiforge).

Key features:
- JSON output in production; colourised console format in dev
- API keys / secrets are stripped from log entries automatically
- Module and provider name injected via bound loggers
"""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

# Regex to detect values that look like API keys / secrets.
# Patterns:
#   sk-...  OpenAI-style keys starting with "sk-"
#   ant-... Anthropic-style keys starting with "ant-"
#   pa-...  Voyage AI keys starting with "pa-"
#   Typical long base64 tokens — must start with a letter/digit and be ≥40 chars
#   with no hyphens (UUIDs/IDs have hyphens; real secrets usually don't)
_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9\-_]{20,}"  # OpenAI / Cohere style
    r"|ant-[A-Za-z0-9\-_]{20,}"  # Anthropic style
    r"|pa-[A-Za-z0-9\-_]{20,}"  # Voyage AI style
    r"|[A-Za-z0-9+/]{40,}={0,2})",  # Long base64 without hyphens (≥40 chars)
    re.ASCII,
)

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "token",
        "password",
        "passwd",
        "authorization",
        "openai_api_key",
        "anthropic_api_key",
        "voyage_api_key",
        "langsmith_api_key",
        "azure_openai_api_key",
    }
)


def _strip_secrets(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor: replace secret-looking values with ``[REDACTED]``."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
            continue
        val = event_dict[key]
        if isinstance(val, str) and _SECRET_PATTERN.search(val) and len(val) > 20:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Initialise structlog for the whole process.

    Args:
        level: Python logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        fmt:   Output format — ``"json"`` for machine-parseable, ``"console"``
               for human-readable colourised output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging (captures 3rd-party lib logs)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _strip_secrets,
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "console":
        renderer: Any = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # Allow reconfiguration at runtime
    )

    # Route stdlib logs (3rd-party libraries) through a separate logging handler
    # so they use the same format, but don't interfere with the structlog pipeline.
    std_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[renderer],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(std_formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
