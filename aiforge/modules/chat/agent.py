"""LangGraph StateGraph agent for multi-turn chat with tool use.

The graph implements a route → execute_tool → process_result pipeline with
a conditional loop: if the *process_result* node detects more tool calls,
control flows back to *execute_tool* (up to ``max_tool_iterations`` times).

Tool calls are read one of two ways:

* **native** (``native_tools=True``): the provider receives the tools via its
  tool-calling API and returns structured :class:`~aiforge.core.types.ToolCall`
  objects; results go back as ``tool`` messages tied to each call's id.
* **prompted** (the default, for custom providers without tool calling): the
  system prompt asks for ``{"tool": ..., "input": ...}`` JSON, which is parsed
  out of the reply text and results are returned as a user message.

Usage::

    graph = build_chat_graph(llm_fn=my_llm, tools=[calculator])
    result = graph.invoke(initial_state)
    print(result["messages"][-1]["content"])
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class ChatAgentState(TypedDict):
    """State passed through the chat agent graph.

    Fields annotated with ``operator.add`` use LangGraph's *reducer*
    semantics: node return values are **accumulated** rather than replaced.
    """

    messages: list[dict[str, Any]]
    current_input: str
    tool_calls: list[dict[str, Any]]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    iteration: int
    max_tool_iterations: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]
    total_cost: Annotated[float, operator.add]
    model: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_chat_graph(
    llm_fn: Callable[[list[dict[str, Any]]], Any],
    tools: list[Any] | None = None,
    *,
    native_tools: bool = False,
) -> Any:
    """Build and compile the chat agent graph.

    Args:
        llm_fn:  ``(messages: list[dict]) → CompletionResult`` — wraps the
                 module's ``_call_llm`` method. With ``native_tools`` it must
                 offer the tools to the provider itself.
        tools:   List of LangChain-compatible tools (each has ``.name``,
                 ``.description``, and ``.invoke()``).
        native_tools: Read tool calls from ``CompletionResult.tool_calls``
                 instead of parsing JSON out of the reply text.

    Returns:
        A compiled LangGraph ``Runnable``.
    """
    tool_map: dict[str, Any] = {}
    if tools:
        for t in tools:
            tool_map[t.name] = t

    # -- nodes ---------------------------------------------------------------

    def _decide(
        state: ChatAgentState, step: str, tool_decision: str, answer: str
    ) -> dict[str, Any]:
        """Call the LLM once and either queue tool calls or record the answer."""
        result = llm_fn(state["messages"])
        raw = result.text.strip()

        if native_tools:
            parsed_tools = [
                {"id": tc.id, "tool": tc.name, "input": tc.arguments} for tc in result.tool_calls
            ]
        else:
            parsed_tools = _parse_tool_calls(raw)

        updates: dict[str, Any] = {
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "total_cost": result.cost,
            "model": result.model,
        }

        if parsed_tools and state["iteration"] < state["max_tool_iterations"]:
            updates["tool_calls"] = parsed_tools
            if native_tools:
                # The assistant turn that made the calls must precede their results.
                updates["messages"] = state["messages"] + [
                    {
                        "role": "assistant",
                        "content": raw,
                        "tool_calls": [
                            {"id": tc["id"], "name": tc["tool"], "arguments": tc["input"]}
                            for tc in parsed_tools
                        ],
                    }
                ]
            updates["trace"] = [
                {
                    "step": step,
                    "decision": tool_decision,
                    "tool_calls": [tc["tool"] for tc in parsed_tools],
                }
            ]
        elif parsed_tools and not native_tools:
            # Prompted mode at the iteration cap: keep the legacy behaviour of
            # leaving the request pending (the caller sees the last marker).
            updates["tool_calls"] = parsed_tools
            updates["trace"] = [
                {
                    "step": step,
                    "decision": tool_decision,
                    "tool_calls": [tc["tool"] for tc in parsed_tools],
                }
            ]
        else:
            updates["tool_calls"] = []
            updates["messages"] = state["messages"] + [{"role": "assistant", "content": raw}]
            updates["trace"] = [{"step": step, "decision": answer}]

        return updates

    def route_node(state: ChatAgentState) -> dict[str, Any]:
        """Send the full message history to the LLM; collect tool calls."""
        return _decide(state, "route", "needs_tool", "direct_answer")

    def execute_tool_node(state: ChatAgentState) -> dict[str, Any]:
        """Execute all pending tool calls and record results."""
        new_results: list[dict[str, Any]] = []

        for tc in state["tool_calls"]:
            tool_name = tc["tool"]
            tool_input = tc.get("input", {})

            failed = True
            if tool_name in tool_map:
                try:
                    if isinstance(tool_input, dict):
                        # Auto-coerce: LLMs often produce dict/list values
                        # for parameters that tools expect as JSON strings.
                        tool_input = _coerce_tool_input(tool_map[tool_name], tool_input)
                        output = str(tool_map[tool_name].invoke(tool_input))
                    else:
                        output = str(tool_map[tool_name].invoke(str(tool_input)))
                    failed = False
                except Exception as exc:
                    output = f"Error: {exc}"
            else:
                output = f"Error: Unknown tool '{tool_name}'"

            new_results.append(
                {
                    "id": tc.get("id", ""),
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                    "is_error": failed,
                }
            )

        if native_tools:
            new_messages = state["messages"] + [
                {
                    "role": "tool",
                    "tool_call_id": r["id"],
                    "content": r["output"],
                    "is_error": r["is_error"],
                }
                for r in new_results
            ]
        else:
            # Append tool results as a message so the LLM can see them
            tool_summary = "\n".join(
                f"Tool '{r['tool']}' returned: {r['output']}" for r in new_results
            )
            new_messages = state["messages"] + [
                {
                    "role": "assistant",
                    "content": f"[Tool calls: {', '.join(tc['tool'] for tc in state['tool_calls'])}]",
                },
                {"role": "user", "content": f"Tool results:\n{tool_summary}"},
            ]

        return {
            "messages": new_messages,
            "tool_results": new_results,
            "tool_calls": [],
            "iteration": state["iteration"] + 1,
            "trace": [
                {
                    "step": "execute_tool",
                    "tools": [r["tool"] for r in new_results],
                    "iteration": state["iteration"] + 1,
                }
            ],
        }

    def process_result_node(state: ChatAgentState) -> dict[str, Any]:
        """Let the LLM process tool results and decide next action."""
        return _decide(state, "process_result", "more_tools", "respond")

    # -- routing -------------------------------------------------------------

    def should_use_tools(state: ChatAgentState) -> str:
        if state["tool_calls"]:
            return "execute_tool"
        return "end"

    def should_continue_after_tool(state: ChatAgentState) -> str:
        if state["tool_calls"] and state["iteration"] < state["max_tool_iterations"]:
            return "execute_tool"
        return "end"

    # -- assemble graph ------------------------------------------------------

    graph = StateGraph(ChatAgentState)

    graph.add_node("route", route_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("process_result", process_result_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        should_use_tools,
        {"execute_tool": "execute_tool", "end": END},
    )
    graph.add_edge("execute_tool", "process_result")
    graph.add_conditional_edges(
        "process_result",
        should_continue_after_tool,
        {"execute_tool": "execute_tool", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_tool_input(tool: Any, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Auto-serialize dict/list values to JSON strings when the tool expects str.

    LLMs frequently produce structured objects for parameters that MCP tools
    declare as ``"type": "string"`` (e.g.  a ``body`` parameter that should
    be a JSON-encoded string).  This helper checks the tool's ``args``
    schema and converts mismatched types so Pydantic validation succeeds.
    """
    try:
        args_schema = getattr(tool, "args", None)
        if not args_schema:
            return tool_input
        coerced = dict(tool_input)
        for key, value in coerced.items():
            if isinstance(value, (dict, list)):
                prop = args_schema.get(key, {})
                if isinstance(prop, dict) and prop.get("type") == "string":
                    coerced[key] = json.dumps(value)
        return coerced
    except Exception:
        return tool_input


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool call JSON objects from LLM output.

    Looks for ``{"tool": ..., "input": ...}`` patterns in the text.
    Handles multi-line JSON and JSON embedded in surrounding prose.
    Returns an empty list if no valid tool calls are found.
    """
    calls: list[dict[str, Any]] = []

    # Fast exit — no possible tool call in the text
    if '"tool"' not in text:
        return calls

    stripped = text.strip()

    # 1. Try the whole text as a single JSON object
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if "tool" in obj:
                calls.append(obj)
                return calls
        except json.JSONDecodeError:
            pass

    # 2. Bracket-matching: find top-level { } pairs that may span
    #    multiple lines and be surrounded by prose.
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                if '"tool"' in candidate:
                    try:
                        obj = json.loads(candidate)
                        if "tool" in obj:
                            calls.append(obj)
                    except json.JSONDecodeError:
                        pass
                start = None

    if calls:
        return calls

    # 3. Fallback — line-by-line (single-line JSON on its own line)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"tool"' in line:
            try:
                obj = json.loads(line)
                if "tool" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                continue

    return calls
