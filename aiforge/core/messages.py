"""
Message and content-block helpers.

AIForge messages are plain dicts in the OpenAI chat shape, so existing code
that passes ``{"role": "user", "content": "..."}`` keeps working. These
helpers build the richer forms every built-in provider understands:

* multimodal user content (text, images, PDFs and other files),
* assistant turns that carry native tool calls,
* ``tool`` messages returning a tool's result to the model,
* a prompt-cache breakpoint on a text block (Anthropic).

Content blocks use LangChain's provider-neutral format, which each LangChain
chat model translates to its own API. Example::

    from aiforge.core import messages as m

    msgs = [
        m.system("You are a careful analyst."),
        m.user("What does this chart show?", m.image(path="chart.png")),
    ]
    result = provider.complete(msgs)
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from aiforge.core.types import ToolCall

Role = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict):
    """One chat message.

    ``content`` is either a string or a list of content blocks. ``tool_calls``
    appears only on assistant messages, ``tool_call_id`` / ``is_error`` only
    on ``tool`` messages.
    """

    role: Role
    content: str | list[dict[str, Any]]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]
    is_error: NotRequired[bool]


CACHE_BREAKPOINT: dict[str, str] = {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


def text(value: str, *, cache: bool = False) -> dict[str, Any]:
    """A text block. ``cache=True`` marks a prompt-cache breakpoint after it (Anthropic)."""
    block: dict[str, Any] = {"type": "text", "text": value}
    if cache:
        block["cache_control"] = dict(CACHE_BREAKPOINT)
    return block


def _binary_block(
    kind: Literal["image", "file"],
    *,
    url: str | None,
    data: bytes | str | None,
    path: str | Path | None,
    mime_type: str | None,
) -> dict[str, Any]:
    sources = [s for s in (url, data, path) if s is not None]
    if len(sources) != 1:
        raise ValueError(f"{kind}() takes exactly one of url=, data= or path=")

    if url is not None:
        block: dict[str, Any] = {"type": kind, "url": url}
        if mime_type:
            block["mime_type"] = mime_type
        return block

    if path is not None:
        p = Path(path)
        data = p.read_bytes()
        mime_type = mime_type or mimetypes.guess_type(p.name)[0]

    if mime_type is None:
        raise ValueError(f"{kind}(): mime_type= is required when it can't be inferred")
    encoded = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
    return {"type": kind, "base64": encoded, "mime_type": mime_type}


def image(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | Path | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """An image block from a URL, raw or base64 bytes, or a local file."""
    return _binary_block("image", url=url, data=data, path=path, mime_type=mime_type)


def file(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | Path | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """A document block (e.g. a PDF) from a URL, raw or base64 bytes, or a local file."""
    return _binary_block("file", url=url, data=data, path=path, mime_type=mime_type)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _content(parts: tuple[str | dict[str, Any], ...]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], str):
        return parts[0]
    return [text(p) if isinstance(p, str) else p for p in parts]


def system(*parts: str | dict[str, Any], cache: bool = False) -> Message:
    """A system message. ``cache=True`` puts a prompt-cache breakpoint at its end."""
    content = _content(parts)
    if cache:
        blocks = [text(content)] if isinstance(content, str) else list(content)
        blocks[-1] = {**blocks[-1], "cache_control": dict(CACHE_BREAKPOINT)}
        content = blocks
    return {"role": "system", "content": content}


def user(*parts: str | dict[str, Any]) -> Message:
    """A user message from strings and/or content blocks."""
    return {"role": "user", "content": _content(parts)}


def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
    """An assistant message, optionally carrying the tool calls it made."""
    msg: Message = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [tc.model_dump() for tc in tool_calls]
    return msg


def tool_result(tool_call_id: str, content: str, *, is_error: bool = False) -> Message:
    """The result of one tool call, sent back to the model."""
    msg: Message = {"role": "tool", "content": content, "tool_call_id": tool_call_id}
    if is_error:
        msg["is_error"] = True
    return msg
