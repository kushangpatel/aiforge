"""Pydantic models for the Chat module."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiforge.core.types import ModuleResponse


class ToolCallRecord(ModuleResponse):
    """Record of a single tool invocation."""

    tool: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None

    def to_text(self) -> str:
        return f"{self.tool}({self.input}) → {self.output}"


class ChatResponse(ModuleResponse):
    """Response from a single chat turn.

    Attributes:
        message:          The assistant's text reply.
        conversation_id:  Identifies the ongoing conversation.
        tool_calls:       List of tool invocations made during this turn.
        iterations:       Number of tool-use iterations executed.
        memory_mode:      Memory strategy in effect.
    """

    message: str = ""
    conversation_id: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    iterations: int = 0
    memory_mode: str = "buffer"

    def to_text(self) -> str:
        return self.message
