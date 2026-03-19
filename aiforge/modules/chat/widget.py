"""Streamlit widget for the ChatAgent module."""

from __future__ import annotations

import ast
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Built-in safe tools (available in the playground)
# ---------------------------------------------------------------------------


def _safe_calculator(expression: str) -> str:
    """Evaluate a mathematical expression using AST-based safe parsing."""
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        # operators
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    }
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Unsafe operation: {type(node).__name__}")
    result = eval(compile(tree, "<string>", "eval"))  # noqa: S307
    return str(result)


def _current_time() -> str:
    """Return the current date and time in ISO format."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()


def _get_builtin_tools() -> list[Any]:
    """Create LangChain-compatible tool wrappers for playground use."""
    from langchain_core.tools import StructuredTool

    calc = StructuredTool.from_function(
        func=_safe_calculator,
        name="calculator",
        description="Evaluate a mathematical expression safely (arithmetic only).",
    )
    clock = StructuredTool.from_function(
        func=_current_time,
        name="current_time",
        description="Get the current date and time (UTC).",
    )
    return [calc, clock]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the ChatAgent playground widget."""
    # ── Sidebar config ──
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful AI assistant.",
        height=80,
    )
    col1, col2 = st.columns(2)
    with col1:
        memory_mode = st.selectbox(
            "Memory mode", ["buffer", "summary", "none"], index=0,
        )
    with col2:
        memory_k = st.slider("Buffer size (K)", 2, 50, 20)

    enable_tools = st.checkbox("Enable built-in tools", value=True)

    if st.button("Reset conversation"):
        st.session_state.pop("chat_agent", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

    st.divider()

    # ── Initialise agent ──
    if "chat_agent" not in st.session_state:
        _init_agent(system_prompt, memory_mode, memory_k, enable_tools)

    agent = st.session_state.chat_agent

    # ── Message history ──
    for entry in st.session_state.get("chat_history", []):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("tool_calls"):
                with st.expander("Tool calls"):
                    for tc in entry["tool_calls"]:
                        st.code(
                            f"{tc['tool']}({tc['input']}) → {tc.get('output', '…')}",
                        )
            if entry.get("cost") is not None:
                st.caption(
                    f"Cost: ${entry['cost']:.4f} · "
                    f"Tokens: {entry.get('input_tokens', 0)}↑ "
                    f"{entry.get('output_tokens', 0)}↓ · "
                    f"Tools: {entry.get('iterations', 0)}"
                )

    # ── Chat input ──
    user_input = st.chat_input("Type a message…")
    if not user_input:
        return

    # Show user message
    st.session_state.setdefault("chat_history", []).append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.spinner("Thinking…"):
        try:
            result = agent.chat(user_input)
        except Exception as exc:
            st.error(f"Chat error: {exc}")
            return

    assistant_entry: dict[str, Any] = {
        "role": "assistant",
        "content": result.message,
        "tool_calls": [tc.model_dump() for tc in result.tool_calls],
        "cost": result.cost,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "iterations": result.iterations,
    }
    st.session_state.chat_history.append(assistant_entry)

    with st.chat_message("assistant"):
        st.markdown(result.message)
        if result.tool_calls:
            with st.expander("Tool calls"):
                for tc in result.tool_calls:
                    st.code(f"{tc.tool}({tc.input}) → {tc.output}")
        st.caption(
            f"Cost: ${result.cost:.4f} · "
            f"Tokens: {result.input_tokens}↑ {result.output_tokens}↓ · "
            f"Tools: {result.iterations}"
        )


def _init_agent(
    system_prompt: str,
    memory_mode: str,
    memory_k: int,
    enable_tools: bool,
) -> None:
    """Create a ChatAgent and store it in session state."""
    from aiforge.modules.chat.module import ChatAgent

    tools = _get_builtin_tools() if enable_tools else []
    st.session_state.chat_agent = ChatAgent(
        system_prompt=system_prompt,
        tools=tools,
        memory=memory_mode,
        memory_k=memory_k,
    )
    st.session_state.chat_history = []
