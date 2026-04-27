"""Unit tests for the MCP client integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from aiforge.mcp.config import MCPServerConfig

# ---------------------------------------------------------------------------
# MCPServerConfig validation
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_stdio_valid(self):
        cfg = MCPServerConfig(name="test", transport="stdio", command="npx", args=["server"])
        assert cfg.name == "test"
        assert cfg.transport == "stdio"
        assert cfg.command == "npx"

    def test_sse_valid(self):
        cfg = MCPServerConfig(name="test", transport="sse", url="http://localhost:3000/mcp")
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:3000/mcp"

    def test_stdio_missing_command(self):
        with pytest.raises(ValueError, match="stdio transport requires 'command'"):
            MCPServerConfig(name="bad", transport="stdio")

    def test_sse_missing_url(self):
        with pytest.raises(ValueError, match="sse transport requires 'url'"):
            MCPServerConfig(name="bad", transport="sse")

    def test_disabled_server(self):
        cfg = MCPServerConfig(name="off", transport="stdio", command="npx", enabled=False)
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = MCPServerConfig(name="x", transport="stdio", command="node")
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.timeout == 30.0
        assert cfg.enabled is True

    def test_env_vars(self):
        cfg = MCPServerConfig(
            name="jira",
            transport="stdio",
            command="npx",
            args=["@mcp/server-jira"],
            env={"JIRA_URL": "https://example.atlassian.net"},
        )
        assert cfg.env["JIRA_URL"] == "https://example.atlassian.net"


# ---------------------------------------------------------------------------
# MCPClientManager
# ---------------------------------------------------------------------------


def _mock_mcp_tool(name: str = "list_issues", description: str = "List issues"):
    """Create a mock MCP tool metadata object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project key"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["project"],
    }
    return tool


class TestMCPClientManager:
    def test_disabled_servers_filtered(self):
        from aiforge.mcp.client import MCPClientManager

        configs = [
            MCPServerConfig(name="on", transport="stdio", command="npx"),
            MCPServerConfig(name="off", transport="stdio", command="npx", enabled=False),
        ]
        manager = MCPClientManager(configs)
        assert len(manager._configs) == 1
        assert manager._configs[0].name == "on"

    def test_properties_before_connect(self):
        from aiforge.mcp.client import MCPClientManager

        manager = MCPClientManager([])
        assert manager.connected is False
        assert manager.server_count == 0
        assert manager.tool_count == 0
        assert manager.tool_names() == []

    @pytest.mark.asyncio
    async def test_connect_and_discover_tools(self):
        """Test that connect() calls session.initialize() and list_tools()."""
        from aiforge.mcp.client import MCPClientManager

        mock_tool = _mock_mcp_tool()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        cfg = MCPServerConfig(name="jira", transport="stdio", command="npx", args=["server"])
        manager = MCPClientManager([cfg])

        with patch.object(manager, "_connect_server", return_value=mock_session):
            await manager.connect()

        assert manager.connected is True
        assert manager.server_count == 1
        assert manager.tool_count == 1
        assert manager.tool_names() == ["jira_list_issues"]

    @pytest.mark.asyncio
    async def test_get_langchain_tools(self):
        """Test that discovered tools are wrapped as StructuredTool."""
        from aiforge.mcp.client import MCPClientManager

        mock_tool = _mock_mcp_tool()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        cfg = MCPServerConfig(name="jira", transport="stdio", command="npx", args=["server"])
        manager = MCPClientManager([cfg])

        with patch.object(manager, "_connect_server", return_value=mock_session):
            await manager.connect()

        tools = manager.get_langchain_tools()
        assert len(tools) == 1
        assert tools[0].name == "jira_list_issues"
        assert tools[0].description == "List issues"
        # StructuredTool has .invoke()
        assert hasattr(tools[0], "invoke")

    @pytest.mark.asyncio
    async def test_graceful_server_failure(self):
        """If one server fails to connect, others still work."""
        from aiforge.mcp.client import MCPClientManager

        mock_tool = _mock_mcp_tool("search", "Search repos")
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        good_session = AsyncMock()
        good_session.initialize = AsyncMock()
        good_session.list_tools = AsyncMock(return_value=mock_tools_result)

        cfg_bad = MCPServerConfig(name="broken", transport="stdio", command="bad")
        cfg_good = MCPServerConfig(name="github", transport="stdio", command="npx")
        manager = MCPClientManager([cfg_bad, cfg_good])

        call_count = 0

        async def _side_effect(cfg: MCPServerConfig) -> Any:
            nonlocal call_count
            call_count += 1
            if cfg.name == "broken":
                raise ConnectionError("Server not found")
            return good_session

        with patch.object(manager, "_connect_server", side_effect=_side_effect):
            await manager.connect()

        assert manager.connected is True
        assert manager.server_count == 1
        assert manager.tool_count == 1

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect clears state."""
        from aiforge.mcp.client import MCPClientManager

        manager = MCPClientManager([])
        manager._connected = True
        manager._session_map = {"test": MagicMock()}
        manager._tools_meta = [(MagicMock(), MagicMock())]

        await manager.disconnect()

        assert manager.connected is False
        assert manager.server_count == 0
        assert manager.tool_count == 0

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        """Calling connect() twice should not re-connect."""
        from aiforge.mcp.client import MCPClientManager

        manager = MCPClientManager([])
        manager._connected = True

        # Should return immediately without doing anything
        await manager.connect()
        assert manager.connected is True


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class TestToolWrapper:
    def test_loop_thread_run(self):
        from aiforge.mcp.tool_wrapper import _LoopThread

        lt = _LoopThread()
        try:

            async def _coro() -> str:
                return "hello"

            result = lt.run(_coro())
            assert result == "hello"
        finally:
            lt.stop()

    def test_make_sync_tool_caller(self):
        from aiforge.mcp.tool_wrapper import _LoopThread, make_sync_tool_caller

        lt = _LoopThread()
        try:
            # Mock session with async call_tool
            mock_content = MagicMock()
            mock_content.text = "result text"
            mock_result = MagicMock()
            mock_result.content = [mock_content]

            mock_session = MagicMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)

            caller = make_sync_tool_caller(
                mock_session,
                "list_issues",
                timeout=10.0,
                loop_thread=lt,
            )
            result = caller(project="TEST")

            assert result == "result text"
            mock_session.call_tool.assert_called_once_with(
                "list_issues", arguments={"project": "TEST"}
            )
        finally:
            lt.stop()


# ---------------------------------------------------------------------------
# _build_args_model
# ---------------------------------------------------------------------------


class TestBuildArgsModel:
    def test_builds_model_from_schema(self):
        from aiforge.mcp.client import _build_args_model

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        }
        Model = _build_args_model("test_tool", schema)
        assert Model.__name__ == "MCP_test_tool_Args"

        # Required field
        with pytest.raises(ValidationError):
            Model(limit=10)  # missing required 'query'

        # Valid instance
        instance = Model(query="hello", limit=5)
        assert instance.query == "hello"
        assert instance.limit == 5

    def test_empty_properties(self):
        from aiforge.mcp.client import _build_args_model

        schema = {"type": "object", "properties": {}}
        Model = _build_args_model("empty", schema)
        instance = Model()
        assert instance is not None


# ---------------------------------------------------------------------------
# ChatAgent MCP integration
# ---------------------------------------------------------------------------


class TestChatAgentMCP:
    def _agent_with_mcp_tools(self, mcp_tools: list[Any]) -> Any:
        """Build a ChatAgent with mock MCP tools injected."""
        from aiforge.modules.chat.module import ChatAgent
        from aiforge.modules.chat.prompts import DEFAULT_SYSTEM_PROMPT

        agent = object.__new__(ChatAgent)
        agent.tools = list(mcp_tools)
        agent.memory_mode = "buffer"
        agent.memory_k = 20
        agent.max_tool_iterations = 5
        agent.conversation_id = "test-mcp"
        agent._system_prompt_override = None
        agent._mcp_manager = MagicMock()

        import structlog

        agent.logger = structlog.get_logger()
        agent._provider = None
        agent._embedding_provider = MagicMock()
        settings = MagicMock()
        settings.RATE_LIMIT_RPM = 500
        settings.RATE_LIMIT_TPM = 150_000
        settings.MODEL = "gpt-4o-mini"
        settings.CUSTOM_PROMPTS = {}
        settings.MCP_SERVERS = []
        agent.settings = settings
        from aiforge.infra.cost_tracker import CostTracker

        agent.cost_tracker = CostTracker.get_session()

        if mcp_tools:
            tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in agent.tools)
            from aiforge.modules.chat.prompts import TOOL_USE_SYSTEM

            agent.system_prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_USE_SYSTEM.format(tool_descriptions=tool_descs)}"
            )
        else:
            agent.system_prompt = DEFAULT_SYSTEM_PROMPT

        agent._messages = [{"role": "system", "content": agent.system_prompt}]
        return agent

    def test_mcp_tools_in_system_prompt(self):
        """MCP tools should appear in the system prompt tool descriptions."""
        from langchain_core.tools import StructuredTool

        mcp_tool = StructuredTool.from_function(
            func=lambda project: "issues",
            name="jira_list_issues",
            description="List Jira issues for a project",
        )

        agent = self._agent_with_mcp_tools([mcp_tool])
        assert "jira_list_issues" in agent.system_prompt
        assert "List Jira issues for a project" in agent.system_prompt

    def test_close_disconnects_mcp(self):
        """close() should call disconnect_sync on the MCP manager."""
        agent = self._agent_with_mcp_tools([])
        mock_manager = MagicMock()
        agent._mcp_manager = mock_manager

        agent.close()

        mock_manager.disconnect_sync.assert_called_once()
        assert agent._mcp_manager is None

    def test_close_without_mcp(self):
        """close() is safe when no MCP manager is set."""
        agent = self._agent_with_mcp_tools([])
        agent._mcp_manager = None
        agent.close()  # should not raise

    @pytest.mark.asyncio
    async def test_aclose(self):
        """aclose() should call async disconnect."""
        agent = self._agent_with_mcp_tools([])
        mock_manager = AsyncMock()
        agent._mcp_manager = mock_manager

        await agent.aclose()

        mock_manager.disconnect.assert_called_once()
        assert agent._mcp_manager is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """ChatAgent should work as an async context manager."""
        agent = self._agent_with_mcp_tools([])
        mock_manager = AsyncMock()
        agent._mcp_manager = mock_manager

        async with agent:
            pass

        mock_manager.disconnect.assert_called_once()

    def test_reset_preserves_mcp(self):
        """reset() should clear messages but keep MCP connections."""
        agent = self._agent_with_mcp_tools([])
        mock_manager = MagicMock()
        agent._mcp_manager = mock_manager

        agent.reset()

        # MCP manager should NOT be disconnected
        mock_manager.disconnect_sync.assert_not_called()
        assert agent._mcp_manager is mock_manager
