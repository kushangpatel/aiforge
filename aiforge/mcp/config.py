"""MCP server configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server connection.

    Args:
        name:      Display name (used as tool-name prefix to avoid collisions).
        transport: ``"stdio"`` for subprocess or ``"sse"`` for HTTP streaming.
        command:   Executable for stdio transport (e.g. ``npx``, ``python``).
        args:      CLI arguments for the stdio command.
        url:       SSE endpoint URL (for SSE transport).
        env:       Environment variables passed to the stdio subprocess.
        timeout:   Per-tool-call timeout in seconds.
        enabled:   Toggle individual servers on/off.
    """

    name: str
    transport: Literal["stdio", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = []
    url: str | None = None
    env: dict[str, str] = {}
    timeout: float = 30.0
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                f"MCP server {self.name!r}: stdio transport requires 'command'"
            )
        if self.transport == "sse" and not self.url:
            raise ValueError(
                f"MCP server {self.name!r}: sse transport requires 'url'"
            )
        return self
