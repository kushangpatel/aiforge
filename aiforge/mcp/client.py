"""MCPClientManager — connect to MCP servers, discover tools, wrap as LangChain StructuredTools."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import structlog

from aiforge.mcp.config import MCPServerConfig
from aiforge.mcp.tool_wrapper import _LoopThread, make_sync_tool_caller

logger = structlog.get_logger(__name__)


def _build_args_model(tool_name: str, input_schema: dict[str, Any]) -> type:
    """Dynamically build a Pydantic model from an MCP tool's JSON Schema ``inputSchema``."""
    from pydantic import create_model

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    field_defs: dict[str, Any] = {}
    _JSON_TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, Any)
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else prop_schema.get("default")

        from pydantic import Field

        field_defs[prop_name] = (py_type, Field(default, description=description))

    model_name = f"MCP_{tool_name}_Args"
    model: type = create_model(model_name, **field_defs)
    return model


class MCPClientManager:
    """Manages connections to one or more MCP servers.

    Lifecycle::

        manager = MCPClientManager(configs)
        await manager.connect()       # handshake + tool discovery
        tools = manager.get_langchain_tools()  # StructuredTool list
        ...
        await manager.disconnect()    # graceful cleanup

    Or use the sync helpers :meth:`connect_sync` / :meth:`disconnect_sync`
    for non-async callers.
    """

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self._configs = [s for s in servers if s.enabled]
        self._session_map: dict[str, Any] = {}  # server_name → session
        self._tools_meta: list[tuple[MCPServerConfig, Any]] = []  # (config, mcp_tool)
        self._exit_stack = AsyncExitStack()
        self._connected = False
        self._loop_thread = _LoopThread()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        if self._connected:
            return

        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install it with: pip install 'aiforge[mcp]'"
            ) from exc

        for cfg in self._configs:
            try:
                session = await self._connect_server(cfg)
                self._session_map[cfg.name] = session

                # Discover tools
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._tools_meta.append((cfg, tool))

                logger.info(
                    "mcp_server_connected",
                    server=cfg.name,
                    transport=cfg.transport,
                    tools_count=len(tools_result.tools),
                    tool_names=[t.name for t in tools_result.tools],
                )
            except Exception:
                logger.warning(
                    "mcp_server_connection_failed",
                    server=cfg.name,
                    transport=cfg.transport,
                    exc_info=True,
                )
                # Continue with remaining servers — graceful degradation
                continue

        self._connected = True
        logger.info(
            "mcp_client_ready",
            servers_connected=len(self._session_map),
            total_tools=len(self._tools_meta),
        )

    async def _connect_server(self, cfg: MCPServerConfig) -> Any:
        """Establish a connection to a single MCP server."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if cfg.transport == "stdio":
            server_params = StdioServerParameters(
                command=cfg.command,  # type: ignore[arg-type]
                args=cfg.args,
                env=cfg.env or None,
            )
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session

        elif cfg.transport == "sse":
            from mcp.client.sse import sse_client

            sse_transport = await self._exit_stack.enter_async_context(
                sse_client(cfg.url)  # type: ignore[arg-type]
            )
            read_stream, write_stream = sse_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session

        raise ValueError(f"Unknown transport: {cfg.transport!r}")

    async def disconnect(self) -> None:
        """Gracefully close all MCP sessions and transports."""
        if not self._connected:
            return
        try:
            await self._exit_stack.aclose()
        except Exception:
            logger.warning("mcp_disconnect_error", exc_info=True)
        finally:
            self._session_map.clear()
            self._tools_meta.clear()
            self._connected = False
            logger.info("mcp_client_disconnected")

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def connect_sync(self) -> None:
        """Synchronous wrapper around :meth:`connect`."""
        self._loop_thread.run(self.connect())

    def disconnect_sync(self) -> None:
        """Synchronous wrapper around :meth:`disconnect`.

        Also stops the background event loop thread.
        """
        self._loop_thread.run(self.disconnect())
        self._loop_thread.stop()

    # ------------------------------------------------------------------
    # Tool wrapping
    # ------------------------------------------------------------------

    def get_langchain_tools(self) -> list[Any]:
        """Return discovered MCP tools wrapped as LangChain ``StructuredTool`` objects.

        Each tool is namespaced as ``{server_name}_{tool_name}`` to avoid
        collisions between servers or with local tools.
        """
        from langchain_core.tools import StructuredTool

        wrapped: list[Any] = []

        for cfg, mcp_tool in self._tools_meta:
            session = self._session_map.get(cfg.name)
            if session is None:
                continue

            namespaced_name = f"{cfg.name}_{mcp_tool.name}"
            description = mcp_tool.description or f"MCP tool from {cfg.name}"

            # Build args schema from MCP tool's inputSchema
            input_schema = mcp_tool.inputSchema or {}
            args_model = (
                _build_args_model(namespaced_name, input_schema)
                if input_schema.get("properties")
                else None
            )

            # Create the sync callable that bridges to async MCP call
            sync_caller = make_sync_tool_caller(
                session=session,
                tool_name=mcp_tool.name,
                timeout=cfg.timeout,
                loop_thread=self._loop_thread,
            )

            tool_kwargs: dict[str, Any] = {
                "func": sync_caller,
                "name": namespaced_name,
                "description": description,
            }
            if args_model is not None:
                tool_kwargs["args_schema"] = args_model

            wrapped.append(StructuredTool(**tool_kwargs))

        logger.info("mcp_tools_wrapped", count=len(wrapped))
        return wrapped

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """Whether the manager has an active connection."""
        return self._connected

    @property
    def server_count(self) -> int:
        """Number of successfully connected servers."""
        return len(self._session_map)

    @property
    def tool_count(self) -> int:
        """Total number of discovered MCP tools."""
        return len(self._tools_meta)

    def tool_names(self) -> list[str]:
        """Return namespaced names of all discovered tools."""
        return [f"{cfg.name}_{t.name}" for cfg, t in self._tools_meta]
