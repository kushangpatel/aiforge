"""
Shared utilities for all provider implementations.

Keeping these here avoids cross-provider imports that would force
unnecessary package dependencies (e.g. importing openai.py just to get
_to_lc_messages would pull in langchain-openai even on Anthropic-only installs).
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Sampling-parameter support
# ---------------------------------------------------------------------------

_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

# claude-<family>-<major>[-<minor>] — dated snapshots (…-20250514) are ignored.
_CLAUDE_VERSION_RE = re.compile(r"^claude-[a-z]+-(\d+)(?:-(\d{1,2}))?(?:-|$)")


def supports_sampling_params(model: str) -> bool:
    """Return ``False`` for models that reject ``temperature`` / ``top_p`` / ``top_k``.

    * Anthropic Claude Opus 4.7 and every Claude 5+ model reject non-default
      sampling parameters (HTTP 400, or a client-side ``ValueError`` in
      ``langchain-anthropic``).
    * OpenAI reasoning models (o-series and the GPT-5+ family, except the
      ``-chat`` variants) only accept the default temperature.
    """
    name = model.lower().split("/")[-1]

    match = _CLAUDE_VERSION_RE.match(name)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        return (major, minor) < (4, 7)

    if re.match(r"^o\d", name):
        return False
    gpt = re.match(r"^gpt-(\d+)", name)
    if gpt and int(gpt.group(1)) >= 5:
        return "-chat" in name
    return True


def drop_unsupported_sampling(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return *params* without sampling parameters the model would reject."""
    if supports_sampling_params(model):
        return params
    return {k: v for k, v in params.items() if k not in _SAMPLING_PARAMS}
