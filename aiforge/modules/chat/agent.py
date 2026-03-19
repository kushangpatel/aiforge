"""LangGraph StateGraph agent for multi-turn chat with tool use.

The graph implements a route → execute_tool → process_result pipeline with
a conditional loop: if the *process_result* node detects more tool calls,
control flows back to *execute_tool* (up to ``max_tool_iterations`` times).

Usage::

    graph = build_chat_graph(llm_fn=my_llm, tools=[calculator])
    result = graph.invoke(initial_state)
    print(result["messages"][-1]["content"])
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class ChatAgentState(TypedDict):
    """State passed through the chat agent graph.

    Fields annotated with ``operator.add`` use LangGraph's *reducer*
    semantics: node return values are **accumulated** rather than replaced.
    """

    messages: list[dict[str, str]]
    current_input: str
    tool_calls: list[dict[str, Any]]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    iteration: int
    max_tool_iterations: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]
    model: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_chat_graph(
    llm_fn: Callable[[list[dict[str, str]]], Any],
    tools: list[Any] | None = None,
) -> Any:
    """Build and compile the chat agent graph.

    Args:
        llm_fn:  ``(messages: list[dict]) → CompletionResult`` — wraps the
                 module's ``_call_llm`` method.
        tools:   List of LangChain-compatible tools (each has ``.name``,
                 ``.description``, and ``.invoke()``).

    Returns:
        A compiled LangGraph ``Runnable``.
    """
    tool_map: dict[str, Any] = {}
    if tools:
        for t in tools:
            tool_map[t.name] = t

    # -- nodes ---------------------------------------------------------------

    def route_node(state: ChatAgentState) -> dict[str, Any]:
        """Send the full message history to the LLM; parse tool calls."""
        result = llm_fn(state["messages"])
        raw = result.text.strip()

        parsed_tools = _parse_tool_calls(raw)

        updates: dict[str, Any] = {
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "model": result.model,
        }

        if parsed_tools:
            updates["tool_calls"] = parsed_tools
            updates["trace"] = [
                {
                    "step": "route",
                    "decision": "needs_tool",
                    "tool_calls": [tc["tool"] for tc in parsed_tools],
                }
            ]
        else:
            # Direct answer — append AI message
            updates["tool_calls"] = []
            updates["messages"] = state["messages"] + [
                {"role": "assistant", "content": raw}
            ]
            updates["trace"] = [{"step": "route", "decision": "direct_answer"}]

        return updates

    def execute_tool_node(state: ChatAgentState) -> dict[str, Any]:
        """Execute all pending tool calls and record results."""
        new_results: list[dict[str, Any]] = []

        for tc in state["tool_calls"]:
            tool_name = tc["tool"]
            tool_input = tc.get("input", {})

            if tool_name in tool_map:
                try:
                    if isinstance(tool_input, dict):
                        output = str(tool_map[tool_name].invoke(tool_input))
                    else:
                        output = str(tool_map[tool_name].invoke(str(tool_input)))
                except Exception as exc:
                    output = f"Error: {exc}"
            else:
                output = f"Error: Unknown tool '{tool_name}'"

            new_results.append(
                {"tool": tool_name, "input": tool_input, "output": output}
            )

        # Append tool results as a message so the LLM can see them
        tool_summary = "\n".join(
            f"Tool '{r['tool']}' returned: {r['output']}" for r in new_results
        )
        new_messages = state["messages"] + [
            {"role": "assistant", "content": f"[Tool calls: {', '.join(tc['tool'] for tc in state['tool_calls'])}]"},
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
        result = llm_fn(state["messages"])
        raw = result.text.strip()

        parsed_tools = _parse_tool_calls(raw)

        updates: dict[str, Any] = {
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "model": result.model,
        }

        if parsed_tools:
            updates["tool_calls"] = parsed_tools
            updates["trace"] = [
                {
                    "step": "process_result",
                    "decision": "more_tools",
                    "tool_calls": [tc["tool"] for tc in parsed_tools],
                }
            ]
        else:
            updates["tool_calls"] = []
            updates["messages"] = state["messages"] + [
                {"role": "assistant", "content": raw}
            ]
            updates["trace"] = [
                {"step": "process_result", "decision": "respond"}
            ]

        return updates

    # -- routing -------------------------------------------------------------

    def should_use_tools(state: ChatAgentState) -> str:
        if state["tool_calls"]:
            return "execute_tool"
        return "end"

    def should_continue_after_tool(state: ChatAgentState) -> str:
        if (
            state["tool_calls"]
            and state["iteration"] < state["max_tool_iterations"]
        ):
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


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool call JSON objects from LLM output.

    Looks for ``{"tool": ..., "input": ...}`` patterns in the text.
    Returns an empty list if no valid tool calls are found.
    """
    calls: list[dict[str, Any]] = []
    # Try the whole text as a single JSON object first
    stripped = text.strip()
    if stripped.startswith("{") and '"tool"' in stripped:
        try:
            obj = json.loads(stripped)
            if "tool" in obj:
                calls.append(obj)
                return calls
        except json.JSONDecodeError:
            pass

    # Try to find JSON objects line-by-line
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
