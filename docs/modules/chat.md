# Module: Chat

`aiforge.chat` — Conversational Agent

---

## Purpose

Multi-turn conversational agent with memory, tool use, and autonomous reasoning. Powered by LangGraph.

---

## Public API

```python
from aiforge import ChatAgent
from aiforge.tools import calculator, web_search

agent = ChatAgent(
    system_prompt="You are a helpful assistant for our e-commerce platform.",
    tools=[calculator, web_search],     # optional tool use
    memory="buffer",                    # "buffer" | "summary" | "none"
    memory_k=20,                        # keep last 20 messages
)

# Chat
response = agent.chat("What's the return policy?")
print(response.message)        # "Our return policy allows..."
print(response.tool_calls)     # [] (no tools needed)

# Multi-turn (memory maintained)
response = agent.chat("What about international orders?")
# Agent remembers the context is about return policies

# With tool use
response = agent.chat("What's 15% off $249.99?")
print(response.message)        # "15% off $249.99 is $212.49"
print(response.tool_calls)     # [{"tool": "calculator", "input": "249.99 * 0.85"}]

# Streaming
async for chunk in agent.chat_stream("Explain our shipping options"):
    print(chunk, end="")

# Reset conversation
agent.reset()
```

### Async Support

```python
response = await agent.achat("What's the return policy?")

# Async streaming
async for chunk in agent.achat_stream("Explain shipping options"):
    print(chunk, end="")
```

---

## LangGraph Agent Architecture

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  agent   │  ← LLM decides: respond or use tool
                    └────┬─────┘
                         │
                   ┌─────┴──────┐
                   │            │
              needs tool    no tool needed
                   │            │
              ┌────▼─────┐     │
              │  tools   │     │
              │ executor │     │
              └────┬─────┘     │
                   │           │
                   └───────────┘
                         │
                    ┌────▼─────┐
                    │ respond  │  ← Format final response
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │   END    │
                    └──────────┘
```

This is a **LangGraph** stateful agent with:
- **Conditional edges:** Route between tool execution and direct response
- **Cycles:** Tool use can trigger additional tool calls
- **State persistence:** Conversation memory across turns
- **Tool abstraction:** Plug in any LangChain-compatible tool

---

## Agent State

```python
class ChatAgentState(TypedDict):
    messages: list[BaseMessage]         # conversation history
    tool_calls: list[dict]              # pending tool invocations
    current_tools: list[str]            # available tool names
    iteration: int                      # loop counter
    max_tool_iterations: int            # safety cap (default: 5)
```

---

## Schemas

### Request

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = False
```

### Response

```python
class ChatResponse(BaseModel):
    message: str
    tool_calls: list[dict]
    session_id: str
    input_tokens: int
    output_tokens: int
    cost: float
    trace_id: str
```

---

## Functional Requirements

| ID | Requirement | Priority | Description |
|----|-------------|----------|-------------|
| FR-CHT-001 | Multi-turn conversation | P0 | Maintain conversation history |
| FR-CHT-002 | Tool use | P1 | Agent can call registered tools |
| FR-CHT-003 | Memory management | P0 | Buffer memory with configurable window |
| FR-CHT-004 | Streaming | P0 | Stream messages token by token |
| FR-CHT-005 | Conversation reset | P0 | Clear history |

---

## File Structure

```
aiforge/modules/chat/
├── __init__.py
├── module.py          # ChatAgent class
├── agent.py           # LangGraph chat agent
├── prompts.py         # System prompt templates
├── schemas.py         # ChatRequest, ChatResponse
└── widget.py          # Streamlit widget
```

---

## See Also

- [Quick Start](../quickstart.md) — Basic chat example
- [API Server](../api-server.md) — `POST /chat` endpoint
- [Playground](../playground.md) — Interactive chat tab
