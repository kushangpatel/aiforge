"""ChatAgent endpoints (/chat)."""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# In-memory conversation store (MVP; production would use Redis/DB)
# ---------------------------------------------------------------------------
_MAX_CONVERSATIONS = 200
_conversations: OrderedDict[str, Any] = OrderedDict()


def _store_conversation(conversation_id: str, agent: Any) -> None:
    """Store an agent and evict oldest when over capacity."""
    _conversations[conversation_id] = agent
    _conversations.move_to_end(conversation_id)
    while len(_conversations) > _MAX_CONVERSATIONS:
        _, evicted_agent = _conversations.popitem(last=False)
        # Close MCP connections on evicted agents to avoid leaking subprocesses
        if hasattr(evicted_agent, "close"):
            with contextlib.suppress(Exception):
                evicted_agent.close()


# ---------------------------------------------------------------------------
# Request / response schemas (route-level)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    system_prompt: str | None = None
    memory: str = "buffer"
    memory_k: int = 20
    stream: bool = False
    mcp_servers: list[dict[str, Any]] | None = None


class ChatResponseOut(BaseModel):
    message: str
    conversation_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    memory_mode: str = "buffer"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0


class ResetRequest(BaseModel):
    conversation_id: str


class ResetResponseOut(BaseModel):
    status: str
    conversation_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ChatResponseOut)
async def chat(req: ChatRequest) -> ChatResponseOut | StreamingResponse:
    """Send a chat message. Creates a new conversation if none exists."""
    from aiforge.modules.chat.module import ChatAgent

    try:
        agent: ChatAgent
        if req.conversation_id and req.conversation_id in _conversations:
            agent = _conversations[req.conversation_id]
        else:
            agent = ChatAgent(
                system_prompt=req.system_prompt,
                memory=req.memory,
                memory_k=req.memory_k,
                mcp_servers=req.mcp_servers,
            )
            _store_conversation(agent.conversation_id, agent)

        if req.stream:

            async def _stream() -> AsyncIterator[str]:
                async for chunk in agent.chat_stream(req.message):
                    yield chunk

            return StreamingResponse(_stream(), media_type="text/plain")

        result = agent.chat(req.message)
        return ChatResponseOut(
            message=result.message,
            conversation_id=result.conversation_id,
            tool_calls=[tc.model_dump() for tc in result.tool_calls],
            iterations=result.iterations,
            memory_mode=result.memory_mode,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=result.cost,
            model=result.model,
            trace_id=result.trace_id,
            latency_ms=result.latency_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/reset", response_model=ResetResponseOut)
async def reset_chat(req: ResetRequest) -> ResetResponseOut:
    """Reset a conversation (clear history, keep system prompt)."""
    if req.conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    agent = _conversations.pop(req.conversation_id)
    agent.reset()
    _store_conversation(agent.conversation_id, agent)
    return ResetResponseOut(
        status="reset",
        conversation_id=agent.conversation_id,
    )


@router.get("/conversations")
async def list_conversations() -> dict[str, Any]:
    """List active conversation IDs."""
    return {
        "conversations": list(_conversations.keys()),
        "count": len(_conversations),
    }
