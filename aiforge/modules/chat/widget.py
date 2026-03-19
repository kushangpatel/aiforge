"""Streamlit widget for the ChatAgent module."""

from __future__ import annotations

import ast
import math
import operator as op
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Built-in safe tools (available in the playground)
# ---------------------------------------------------------------------------

# Allowed math functions — mapped from name → callable.
# Only pure, side-effect-free functions are included.
_SAFE_MATH_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    # math module
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
    "hypot": math.hypot,
}

# Allowed constants
_SAFE_MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe math operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsafe constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_MATH_CONSTANTS:
            return _SAFE_MATH_CONSTANTS[name]
        raise ValueError(f"Unknown variable: {name!r}")

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsafe unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        ops: dict[type, Any] = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Mod: op.mod,
            ast.Pow: op.pow,
        }
        fn = ops.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsafe binary op: {type(node.op).__name__}")
        return fn(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are allowed")
        func_name = node.func.id
        if func_name not in _SAFE_MATH_FUNCTIONS:
            raise ValueError(
                f"Unsafe function: {func_name!r}. "
                f"Allowed: {', '.join(sorted(_SAFE_MATH_FUNCTIONS))}"
            )
        args = [_safe_eval_node(a) for a in node.args]
        return _SAFE_MATH_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsafe operation: {type(node).__name__}")


def _safe_calculator(expression: str) -> str:
    """Evaluate a mathematical expression using AST-based safe parsing.

    Supports:
    - Basic arithmetic: ``+  -  *  /  //  %  **``
    - Constants: ``pi``, ``e``, ``tau``, ``inf``
    - Functions: ``sqrt``, ``cbrt``, ``sin``, ``cos``, ``tan``, ``asin``,
      ``acos``, ``atan``, ``atan2``, ``sinh``, ``cosh``, ``tanh``,
      ``log``, ``log2``, ``log10``, ``exp``, ``ceil``, ``floor``,
      ``factorial``, ``gcd``, ``abs``, ``round``, ``min``, ``max``,
      ``radians``, ``degrees``, ``hypot``
    """
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval_node(tree)
    return str(result)


def _current_time() -> str:
    """Return the current date and time in ISO format."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat()


def _research(query: str) -> str:
    """Research a topic using the AIForge search module.

    Performs a semantic search across previously-ingested knowledge sources
    (documents, codebases, etc.) using the AIForge Search module. Returns
    the top results as context the chat agent can use to answer questions.

    Falls back to a helpful error message if the search module or its
    dependencies are not available.
    """
    try:
        from aiforge.modules.search.module import SearchAgent

        agent = SearchAgent()
        result = agent.run(query)
        if not result.results:
            return f"No results found for: {query}"

        lines: list[str] = []
        for i, r in enumerate(result.results[:5], 1):
            source = r.metadata.get("source", "unknown")
            score = f" (score: {r.score:.2f})" if r.score else ""
            lines.append(f"[{i}] {source}{score}\n{r.text[:500]}")
        return "\n\n".join(lines)
    except ImportError:
        return (
            "Research tool unavailable — the search module or its "
            "dependencies (chromadb, sentence-transformers) are not installed."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Research error: {exc}"


def _unit_converter(expression: str) -> str:
    """Convert between common units.

    Format: ``<value> <from_unit> to <to_unit>``
    Examples: ``100 celsius to fahrenheit``, ``5 miles to km``,
    ``2.5 kg to lbs``, ``180 degrees to radians``
    """
    import re

    expression = expression.strip().lower()
    match = re.match(
        r"^(-?\d+\.?\d*)\s+(\w+)\s+to\s+(\w+)$", expression,
    )
    if not match:
        raise ValueError(
            "Format: '<value> <from_unit> to <to_unit>' "
            "(e.g., '100 celsius to fahrenheit')"
        )

    value = float(match.group(1))
    from_u = match.group(2)
    to_u = match.group(3)

    # Temperature
    temp_conversions: dict[tuple[str, str], Any] = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda v: v + 273.15,
        ("kelvin", "celsius"): lambda v: v - 273.15,
        ("fahrenheit", "kelvin"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("kelvin", "fahrenheit"): lambda v: (v - 273.15) * 9 / 5 + 32,
    }

    # Linear conversions: (from, to) → multiply factor
    linear: dict[tuple[str, str], float] = {
        # Length
        ("miles", "km"): 1.60934,
        ("km", "miles"): 0.621371,
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("inches", "cm"): 2.54,
        ("cm", "inches"): 0.393701,
        ("yards", "meters"): 0.9144,
        ("meters", "yards"): 1.09361,
        # Weight
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("oz", "grams"): 28.3495,
        ("grams", "oz"): 0.035274,
        # Volume
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
        # Angle
        ("degrees", "radians"): math.pi / 180,
        ("radians", "degrees"): 180 / math.pi,
    }

    key = (from_u, to_u)
    if key in temp_conversions:
        result = temp_conversions[key](value)
        return f"{value} {from_u} = {result:.4f} {to_u}"
    if key in linear:
        result = value * linear[key]
        return f"{value} {from_u} = {result:.4f} {to_u}"
    if from_u == to_u:
        return f"{value} {from_u} = {value} {to_u} (same unit)"

    available = sorted({f"{a} to {b}" for a, b in {**temp_conversions, **linear}})
    raise ValueError(
        f"Unknown conversion: {from_u} to {to_u}. "
        f"Supported: {', '.join(available)}"
    )


def _get_builtin_tools() -> list[Any]:
    """Create LangChain-compatible tool wrappers for playground use."""
    from langchain_core.tools import StructuredTool

    calc = StructuredTool.from_function(
        func=_safe_calculator,
        name="calculator",
        description=(
            "Evaluate a mathematical expression. Supports arithmetic "
            "(+ - * / // % **), constants (pi, e, tau), and functions "
            "(sqrt, cbrt, sin, cos, tan, asin, acos, atan, log, log2, "
            "log10, exp, ceil, floor, factorial, gcd, abs, round, min, "
            "max, radians, degrees, hypot). "
            "Example: 'sqrt(2) * pi', 'log(100, 10)', 'factorial(10)'"
        ),
    )
    clock = StructuredTool.from_function(
        func=_current_time,
        name="current_time",
        description="Get the current date and time (UTC).",
    )
    research = StructuredTool.from_function(
        func=_research,
        name="research",
        description=(
            "Search previously-ingested knowledge sources (documents, "
            "codebases) for information about a topic. Returns the top "
            "matching passages. Use when you need factual information "
            "that might be in the knowledge base."
        ),
    )
    converter = StructuredTool.from_function(
        func=_unit_converter,
        name="unit_converter",
        description=(
            "Convert between common units. Format: '<value> <from_unit> "
            "to <to_unit>'. Supports temperature (celsius/fahrenheit/"
            "kelvin), length (miles/km/meters/feet/inches/cm/yards), "
            "weight (kg/lbs/oz/grams), volume (liters/gallons), and "
            "angle (degrees/radians). Example: '100 celsius to fahrenheit'"
        ),
    )
    return [calc, clock, research, converter]


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
