"""MCP (Model Context Protocol) client integration for AIForge.

Allows ChatAgent to discover and invoke tools from external MCP servers
(Jira, Slack, GitHub, PostgreSQL, etc.) alongside local tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiforge.mcp.config import MCPServerConfig

if TYPE_CHECKING:
    from aiforge.mcp.client import MCPClientManager

__all__ = [
    "MCPClientManager",
    "MCPServerConfig",
]


def __getattr__(name: str) -> object:
    """Lazy-load MCPClientManager to avoid importing mcp at module level."""
    if name == "MCPClientManager":
        from aiforge.mcp.client import MCPClientManager

        return MCPClientManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
