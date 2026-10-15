# Module: Chat

`aiforge.modules.chat` — Multi-turn conversational agent with memory, tool use, and autonomous reasoning. Powered by **LangGraph**.

---

## Installation

The chat module is included with the base `aiforge` install. No extras required.

```bash
pip install aiforge-kit
```

---

## Import

```python
from aiforge import ChatAgent
from aiforge import ChatResponse, ToolCallRecord  # response types
```

---

## Quick Start

```python
from aiforge import ChatAgent

agent = ChatAgent()

response = agent.chat("What is the capital of France?")
print(response.message)  # "The capital of France is Paris."
```

---

## Constructor

```python
ChatAgent(
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    memory: str = "buffer",
    memory_k: int = 20,
    max_tool_iterations: int = 5,
    # Inherited from BaseModule:
    quality: str | None = None,
    provider: str | None = None,
    **kwargs,  # AIForgeSettings overrides
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str \| None` | `None` | Custom system prompt. If `None`, uses the built-in default: *"You are a helpful AI assistant. Answer questions clearly and concisely."* Resolved via prompt precedence (see [Prompt Resolution](#prompt-resolution)). |
| `tools` | `list[Any] \| None` | `None` | List of LangChain-compatible tools. Each tool must have `.name`, `.description`, and `.invoke()` attributes. Built-in providers receive them through their native tool-calling API; for custom providers, tool-use instructions are appended to the system prompt instead. |
| `memory` | `str` | `"buffer"` | Memory management strategy. One of `"buffer"`, `"summary"`, or `"none"`. See [Memory Modes](#memory-modes). Raises `ValueError` if invalid. |
| `memory_k` | `int` | `20` | Number of messages to retain in the conversation buffer. Must be ≥ 1. Raises `ValueError` otherwise. |
| `max_tool_iterations` | `int` | `5` | Maximum number of tool-use loops per turn before the agent stops. Must be ≥ 1. Safety cap to prevent runaway tool calls. |
| `native_tools` | `bool \| None` | `None` | Use the provider's native tool-calling API. `None` picks it automatically when the provider supports it (all built-in providers do); `False` forces the prompted JSON protocol. |
| `quality` | `str \| None` | `None` | Quality tier for model routing: `"high"`, `"medium"`, or `"low"`. When set (and `MODEL_ROUTER_ENABLED=True`), the model router selects the cheapest model meeting this tier. Falls back to `DEFAULT_QUALITY` if omitted. |
| `provider` | `str \| None` | `None` | LLM provider name (`"openai"`, `"anthropic"`, `"internal_gateway"`, etc.). Falls back to `settings.PROVIDER` if omitted. |
| `**kwargs` | | | Any `AIForgeSettings` field as a keyword override, e.g. `MODEL="gpt-4o"`, `TEMPERATURE=0.5`. |

### Examples

```python
# Minimal — uses defaults (buffer memory, 20 messages, default model)
agent = ChatAgent()

# Custom system prompt
agent = ChatAgent(
    system_prompt="You are a customer service agent for Acme Corp."
)

# With tools
from langchain_core.tools import StructuredTool

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # simplified example

calc_tool = StructuredTool.from_function(calculate, name="calculator", description="Evaluate math")

agent = ChatAgent(tools=[calc_tool])

# Summary memory with GPT-4o
agent = ChatAgent(
    memory="summary",
    memory_k=10,
    provider="openai",
    MODEL="gpt-4o",
    TEMPERATURE=0.3,
)

# Quality-based model routing
agent = ChatAgent(quality="high")   # routes to the best model for your provider
agent = ChatAgent(quality="low")    # routes to the cheapest model
```

---

## Methods

### `chat(message, *, system_prompt=None) -> ChatResponse`

Send a message and get a response. Maintains conversation history across calls.

```python
def chat(
    self,
    message: str,
    *,
    system_prompt: str | None = None,
) -> ChatResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | *(required)* | The user message. Must not be empty or whitespace-only (raises `ValueError`). |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. Temporarily replaces the system prompt for this turn only. |

**Returns:** [`ChatResponse`](#chatresponse)

```python
agent = ChatAgent()

# Basic conversation
response = agent.chat("Hello!")
print(response.message)           # "Hello! How can I help you today?"

# Multi-turn — context is retained
response = agent.chat("What's the weather in NYC?")
response = agent.chat("What about tomorrow?")  # agent remembers "NYC"

# Per-call system prompt override
response = agent.chat(
    "Translate 'hello' to Spanish",
    system_prompt="You are a professional translator. Respond only with the translation.",
)
print(response.message)  # "Hola"
```

---

### `achat(message, *, system_prompt=None) -> ChatResponse`

Async version of `chat()`.

```python
async def achat(
    self,
    message: str,
    *,
    system_prompt: str | None = None,
) -> ChatResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | *(required)* | The user message. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Returns:** [`ChatResponse`](#chatresponse)

```python
response = await agent.achat("Explain quantum computing in one sentence.")
print(response.message)
```

---

### `chat_stream(message, *, system_prompt=None) -> AsyncGenerator[str, None]`

Stream the response token-by-token. Useful for real-time UI rendering.

```python
async def chat_stream(
    self,
    message: str,
    *,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | *(required)* | The user message. |
| `system_prompt` | `str \| None` | `None` | Per-call system prompt override. |

**Yields:** `str` — text chunks as they arrive from the LLM.

```python
async for chunk in agent.chat_stream("Write a haiku about Python"):
    print(chunk, end="", flush=True)
# Output streams token by token
```

---

### `achat_stream(message, *, system_prompt=None) -> AsyncGenerator[str, None]`

Alias for `chat_stream()`. Provided for naming symmetry with `achat()`.

```python
async for chunk in agent.achat_stream("Tell me a joke"):
    print(chunk, end="")
```

---

### `run(text, **kwargs) -> ChatResponse`

BaseModule interface method. Delegates to `chat()`.

```python
def run(self, text: str, **kwargs) -> ChatResponse
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The user message. |
| `**kwargs` | | Accepts `system_prompt` keyword argument. |

**Returns:** [`ChatResponse`](#chatresponse)

```python
# Equivalent to agent.chat("Hello")
response = agent.run("Hello")
```

---

### `arun(text, **kwargs) -> ChatResponse`

Async version of `run()`. Delegates to `achat()`.

```python
async def arun(self, text: str, **kwargs) -> ChatResponse
```

---

### `run_stream(text, **kwargs) -> Iterator[str]`

**Synchronous** streaming interface. Wraps `chat_stream()` for use outside async contexts.

```python
def run_stream(self, text: str, **kwargs) -> Iterator[str]
```

**Returns:** `Iterator[str]`

```python
for chunk in agent.run_stream("Tell me about black holes"):
    print(chunk, end="")
```

---

### `arun_stream(text, **kwargs) -> AsyncGenerator[str, None]`

Async streaming interface. Delegates to `chat_stream()`.

---

### `run_batch(items, concurrency=10, max_batch=100) -> BatchResult`

Run multiple messages concurrently. Inherited from `BaseModule`.

```python
def run_batch(
    self,
    items: list[str],
    concurrency: int = 10,
    max_batch: int = 100,
) -> BatchResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | `list[str]` | *(required)* | List of messages to process. |
| `concurrency` | `int` | `10` | Max concurrent requests. |
| `max_batch` | `int` | `100` | Max items to process. |

**Returns:** `BatchResult` with `.results`, `.total_cost`, `.total_tokens`, `.failed`.

> **Note:** Each item in the batch starts a fresh conversation (no shared memory between items).

```python
questions = ["What is AI?", "What is ML?", "What is DL?"]
batch = agent.run_batch(questions, concurrency=3)
for r in batch.results:
    print(r.message[:80])
print(f"Total cost: ${batch.total_cost:.4f}")
```

---

### `reset() -> None`

Clear conversation history and start a new conversation.

```python
def reset(self) -> None
```

- Clears all messages (keeps only the system prompt)
- Generates a new `conversation_id`

```python
agent.chat("Remember: my name is Alice")
agent.chat("What's my name?")  # "Your name is Alice"

agent.reset()

agent.chat("What's my name?")  # "I don't know your name yet"
```

---

## Response Types

### `ChatResponse`

Returned by `chat()`, `achat()`, `run()`, and `arun()`.

```python
from aiforge import ChatResponse
```

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | The assistant's text reply. |
| `conversation_id` | `str` | Unique ID for this conversation session (16-char hex). |
| `tool_calls` | `list[ToolCallRecord]` | List of tool invocations made during this turn. Empty if no tools were used. |
| `iterations` | `int` | Number of tool-use loop iterations executed (0 if no tools). |
| `memory_mode` | `str` | Active memory strategy (`"buffer"`, `"summary"`, or `"none"`). |
| `cost` | `float` | Total cost in USD for this turn. |
| `input_tokens` | `int` | Total input tokens consumed. |
| `output_tokens` | `int` | Total output tokens generated. |
| `model` | `str` | Model used (e.g. `"gpt-4o-mini"`). |
| `trace_id` | `str` | Unique trace ID for this request (16-char hex). |
| `latency_ms` | `float` | End-to-end latency in milliseconds. |

```python
response = agent.chat("What is 2+2?")

print(response.message)          # "2 + 2 = 4"
print(response.conversation_id)  # "a1b2c3d4e5f67890"
print(response.tool_calls)       # []
print(response.iterations)       # 0
print(response.memory_mode)      # "buffer"
print(response.cost)             # 0.00012
print(response.input_tokens)     # 42
print(response.output_tokens)    # 8
print(response.model)            # "gpt-4o-mini"
print(response.latency_ms)       # 234.5

# to_text() returns the message (useful for pipeline chaining)
print(response.to_text())        # "2 + 2 = 4"
```

---

### `ToolCallRecord`

Represents a single tool invocation within a turn.

```python
from aiforge import ToolCallRecord
```

| Field | Type | Description |
|-------|------|-------------|
| `tool` | `str` | The tool name that was called. |
| `input` | `dict[str, Any]` | The parameters passed to the tool. |
| `output` | `str \| None` | The tool's execution result. `None` if the tool hasn't returned yet. |

```python
response = agent.chat("What's 15% off $249.99?")

for tc in response.tool_calls:
    print(f"Tool: {tc.tool}")       # "calculator"
    print(f"Input: {tc.input}")     # {"expression": "249.99 * 0.85"}
    print(f"Output: {tc.output}")   # "212.4915"
    print(tc.to_text())             # "calculator({'expression': '249.99 * 0.85'}) → 212.4915"
```

---

## Memory Modes

The `memory` parameter controls how conversation history is managed between turns.

### `"buffer"` (default)

Keeps the last `memory_k` messages in the conversation window. Oldest messages are dropped when the limit is exceeded.

```python
agent = ChatAgent(memory="buffer", memory_k=10)
# Keeps only the system prompt + last 10 messages
```

**Behavior:**
- System prompt is always retained as the first message
- When `len(messages) > memory_k + 1`, the oldest non-system messages are trimmed

### `"summary"`

Summarizes older messages when the history exceeds `2 × memory_k`. The summary replaces older messages while keeping the most recent `memory_k` messages intact.

```python
agent = ChatAgent(memory="summary", memory_k=10)
# When history exceeds 20 messages:
#   - Messages older than the last 10 are summarized by the LLM
#   - Summary is injected as a system message
#   - Recent 10 messages are kept verbatim
```

**Behavior:**
- Triggers when `len(messages) > 2 × memory_k`
- Uses the LLM to generate a ≤500-word summary of older messages
- Inserts the summary after the system prompt
- Best for long-running conversations where context matters

### `"none"`

No memory — each turn is independent. Only the system prompt is kept.

```python
agent = ChatAgent(memory="none")
# Each message is treated as a fresh conversation
```

---

## Tool Use

The chat agent can use external tools via the LangGraph agent loop. Tools are LangChain-compatible objects with `.name`, `.description`, and `.invoke()`.

### Providing Tools

```python
from langchain_core.tools import StructuredTool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"72°F and sunny in {city}"

def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Found 3 results for '{query}'"

weather_tool = StructuredTool.from_function(
    get_weather,
    name="weather",
    description="Get current weather for a city",
)
search_tool = StructuredTool.from_function(
    search_docs,
    name="search",
    description="Search internal documentation",
)

agent = ChatAgent(
    system_prompt="You are a helpful assistant with access to weather and doc search.",
    tools=[weather_tool, search_tool],
    max_tool_iterations=3,
)
```

### How Tool Use Works

1. The LLM receives the user message along with the available tools
2. If the LLM decides a tool is needed, it requests a call through the provider's native tool-calling API (custom providers without one emit `{"tool": "weather", "input": {"city": "NYC"}}` JSON instead)
3. The agent executes the tool and feeds the result back to the LLM as a tool result tied to that call
4. The LLM can call more tools or produce a final response
5. This loop continues until the LLM responds directly or `max_tool_iterations` is reached

```python
response = agent.chat("What's the weather in San Francisco?")
print(response.message)      # "It's 72°F and sunny in San Francisco!"
print(response.iterations)   # 1
print(response.tool_calls)   # [ToolCallRecord(tool="weather", input={"city": "San Francisco"}, output="72°F and sunny in San Francisco")]
```

### Multi-Tool Chains

The agent can call multiple tools in sequence within a single turn:

```python
response = agent.chat("What's the weather in NYC and find docs about umbrellas")
# The agent may:
#   1. Call weather("NYC") → "55°F and rainy"
#   2. Call search("umbrellas") → "Found 3 results..."
#   3. Combine results into a final response
print(response.iterations)  # 2
print(len(response.tool_calls))  # 2
```

---

## Prompt Resolution

System prompts are resolved in this precedence order (highest to lowest):

1. **Per-call override:** `agent.chat(msg, system_prompt="...")` — applies to that turn only
2. **Constructor override:** `ChatAgent(system_prompt="...")` — applies to all turns
3. **Config file:** `system_prompt` field under `[chat]` in `aiforge.yaml`
4. **Built-in default:** `"You are a helpful AI assistant. Answer questions clearly and concisely. If you don't know something, say so honestly."`

When tools are registered, tool-use instructions are automatically appended to the resolved system prompt.

---

## LangGraph Agent Architecture

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  route   │  ← LLM decides: respond or use tool
                    └────┬─────┘
                         │
                   ┌─────┴──────┐
                   │            │
              needs tool    no tool needed
                   │            │
              ┌────▼─────┐     │
              │ execute  │     │
              │  tool    │     │
              └────┬─────┘     │
                   │           │
              ┌────▼─────┐    │
              │ process  │    │
              │ result   │    │
              └────┬─────┘    │
                   │          │
            ┌──────┴──────┐   │
            │             │   │
       more tools    done │   │
            │             │   │
            └──► execute  │   │
                 tool     │   │
                          │   │
                   ┌──────┘   │
                   └──────────┘
                         │
                    ┌────▼─────┐
                    │   END    │
                    └──────────┘
```

### Graph Nodes

| Node | Purpose |
|------|---------|
| **route** | Sends the full message history to the LLM. If the LLM outputs a tool call JSON, routes to `execute_tool`. Otherwise, appends the assistant response and routes to `END`. |
| **execute_tool** | Executes all pending tool calls by invoking `tool.invoke(input)`. Appends results to messages. Increments the iteration counter. |
| **process_result** | Sends the updated message history (with tool results) back to the LLM. If the LLM requests more tools, routes back to `execute_tool`. Otherwise, routes to `END`. |

### Routing Conditions

- **`should_use_tools`**: Routes to `execute_tool` if `state["tool_calls"]` is non-empty, otherwise `END`.
- **`should_continue_after_tool`**: Routes to `execute_tool` if `state["tool_calls"]` is non-empty AND `state["iteration"] < state["max_tool_iterations"]`, otherwise `END`.

---

## Agent State

The internal LangGraph state tracks conversation progress through each turn:

```python
class ChatAgentState(TypedDict):
    messages: list[dict[str, str]]                                  # conversation history
    current_input: str                                              # current user message
    tool_calls: list[dict[str, Any]]                                # pending tool invocations
    tool_results: Annotated[list[dict[str, Any]], operator.add]     # accumulated tool results
    iteration: int                                                  # tool-loop counter
    max_tool_iterations: int                                        # safety cap (default: 5)
    trace: Annotated[list[dict[str, Any]], operator.add]            # execution trace log
    total_input_tokens: Annotated[int, operator.add]                # accumulated input tokens
    total_output_tokens: Annotated[int, operator.add]               # accumulated output tokens
    model: str                                                      # model used
```

Fields annotated with `operator.add` use LangGraph reducer semantics — values are accumulated across nodes rather than replaced.

---

## Instance Properties

After constructing a `ChatAgent`, the following properties are available:

| Property | Type | Description |
|----------|------|-------------|
| `agent.system_prompt` | `str` | The fully resolved system prompt (including tool instructions if tools are set). |
| `agent.conversation_id` | `str` | Current conversation session ID (16-char hex). Changes on `reset()`. |
| `agent.memory_mode` | `str` | Active memory mode. |
| `agent.memory_k` | `int` | Buffer size. |
| `agent.max_tool_iterations` | `int` | Tool loop safety cap. |
| `agent.tools` | `list[Any]` | Registered tools. |
| `agent.settings` | `AIForgeSettings` | Resolved configuration. |
| `agent.cost_tracker` | `CostTracker` | Session-level cost tracker. |

---

## Full Examples

### Basic Conversation

```python
from aiforge import ChatAgent

agent = ChatAgent(
    system_prompt="You are a friendly coding tutor.",
    memory="buffer",
    memory_k=30,
)

r = agent.chat("What's a list comprehension in Python?")
print(r.message)
# "A list comprehension is a concise way to create lists..."

r = agent.chat("Show me an example with filtering")
# Agent remembers you're asking about list comprehensions

r = agent.chat("How is that different from a generator expression?")
# Conversation context flows naturally

print(f"Cost so far: ${agent.cost_tracker.session_total:.4f}")
```

### Tool-Augmented Agent

```python
from aiforge import ChatAgent
from langchain_core.tools import StructuredTool

def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    # Your database lookup here
    return f"Order {order_id}: shipped, arriving March 28"

def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"Order {order_id} has been cancelled. Refund in 3-5 days."

agent = ChatAgent(
    system_prompt="You are a customer service agent for ShopCo.",
    tools=[
        StructuredTool.from_function(lookup_order, name="lookup_order", description="Look up order status by ID"),
        StructuredTool.from_function(cancel_order, name="cancel_order", description="Cancel an order by ID"),
    ],
    max_tool_iterations=3,
)

r = agent.chat("What's the status of order #12345?")
print(r.message)        # "Order #12345 has been shipped and will arrive on March 28."
print(r.tool_calls[0])  # ToolCallRecord(tool="lookup_order", input={"order_id": "12345"}, output="Order 12345: shipped, arriving March 28")

r = agent.chat("Actually, please cancel it")
print(r.message)        # "Done! Order #12345 has been cancelled. Your refund will arrive in 3-5 days."
```

### Streaming in an Async App

```python
import asyncio
from aiforge import ChatAgent

async def main():
    agent = ChatAgent(quality="high")

    print("Assistant: ", end="")
    async for chunk in agent.chat_stream("Write a short poem about APIs"):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

### Synchronous Streaming

```python
from aiforge import ChatAgent

agent = ChatAgent()

for chunk in agent.run_stream("Explain REST vs GraphQL"):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
from aiforge import ChatAgent

agent = ChatAgent(quality="low")  # use cheapest model for bulk work

faqs = [
    "What are your business hours?",
    "How do I reset my password?",
    "Where is my order?",
    "Do you offer refunds?",
]

result = agent.run_batch(faqs, concurrency=4)

for r in result.results:
    print(f"Q: ... → A: {r.message[:60]}...")

print(f"Total cost: ${result.total_cost:.4f}")
print(f"Failed: {len(result.failed)}")
```

### Configuring via YAML

```yaml
# aiforge.yaml
PROVIDER: "openai"
MODEL: "gpt-4o-mini"
TEMPERATURE: 0.7
```

```python
# No constructor args needed — picks up aiforge.yaml automatically
agent = ChatAgent()
response = agent.chat("Hello!")
print(response.model)  # "gpt-4o-mini"
```

### Model Override per Agent

```python
# Different models for different use cases
support_agent = ChatAgent(MODEL="gpt-4o", TEMPERATURE=0.3)
creative_agent = ChatAgent(MODEL="gpt-4o", TEMPERATURE=0.9)
cheap_agent = ChatAgent(MODEL="gpt-4o-mini", TEMPERATURE=0.5)
```

---

## Error Handling

| Error | When | Example |
|-------|------|---------|
| `ValueError` | Empty message, invalid `memory` mode, `memory_k < 1`, `max_tool_iterations < 1` | `agent.chat("")` |
| `ProviderError` | LLM provider returns an error | API key invalid, model not found |
| `RateLimitError` | Provider rate limit exceeded | Too many requests |
| `APITimeoutError` | Provider request timed out | Slow network |
| `ConfigError` | Invalid configuration | Unknown provider name |

```python
from aiforge.core.types import ProviderError, RateLimitError

try:
    response = agent.chat("Hello")
except RateLimitError:
    print("Rate limited — retrying after backoff")
except ProviderError as e:
    print(f"Provider error: {e}")
```

---

## File Structure

```
aiforge/modules/chat/
├── __init__.py        # Exports: ChatAgent, ChatResponse, ToolCallRecord
├── module.py          # ChatAgent class (constructor, chat/achat, streaming, memory)
├── agent.py           # LangGraph graph definition (build_chat_graph, nodes, routing)
├── prompts.py         # Prompt templates (DEFAULT_SYSTEM_PROMPT, TOOL_USE_SYSTEM, SUMMARIZE_MEMORY_PROMPT)
├── schemas.py         # Pydantic models (ChatResponse, ToolCallRecord)
└── widget.py          # Streamlit playground widget (built-in tools: calculator, research, unit_converter, current_time)
```

---

## See Also

- [Quick Start](../quickstart.md) — Getting started with AIForge
- [API Server](../api-server.md) — `POST /chat` endpoint
- [Playground](../playground.md) — Interactive chat tab with built-in tools
- [Configuration](../configuration.md) — Full settings reference
- [Build & Publish](../BUILD_AND_PUBLISH.md) — Packaging and distribution
