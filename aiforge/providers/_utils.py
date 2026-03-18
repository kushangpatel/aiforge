"""
Shared utilities for all provider implementations.

Keeping these here avoids cross-provider imports that would force
unnecessary package dependencies (e.g. importing openai.py just to get
_to_lc_messages would pull in langchain-openai even on Anthropic-only installs).
"""

from __future__ import annotations

from typing import Any


def to_lc_messages(messages: list[dict[str, str]]) -> list[Any]:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    return lc
