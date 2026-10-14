"""MCP server configuration schema."""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server connection.

    Args:
        name:      Display name (used as tool-name prefix to avoid collisions).
        transport: ``"stdio"`` for a local subprocess, ``"streamable_http"``
                   (alias ``"http"``) for a remote server, or the deprecated
                   ``"sse"`` for servers that only speak the legacy HTTP+SSE
                   transport.
        command:   Executable for stdio transport (e.g. ``npx``, ``python``).
        args:      CLI arguments for the stdio command.
        url:       Server endpoint URL for the HTTP transports.
        headers:   Extra HTTP headers for the HTTP transports (e.g.
                   ``{"Authorization": "Bearer …"}``).
        env:       Environment variables passed to the stdio subprocess.
        timeout:   Per-tool-call timeout in seconds.
        enabled:   Toggle individual servers on/off.
    """

    name: str
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = []
    url: str | None = None
    headers: dict[str, str] = {}
    env: dict[str, str] = {}
    timeout: float = 30.0
    enabled: bool = True

    @field_validator("transport", mode="before")
    @classmethod
    def _normalise_transport(cls, value: object) -> object:
        return "streamable_http" if value == "http" else value

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server {self.name!r}: stdio transport requires 'command'")
        if self.transport != "stdio" and not self.url:
            raise ValueError(f"MCP server {self.name!r}: {self.transport} transport requires 'url'")
        if self.transport == "sse":
            warnings.warn(
                f"MCP server {self.name!r}: the HTTP+SSE transport is deprecated in the "
                "MCP spec; use transport='streamable_http' unless the server only "
                "supports SSE.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self
