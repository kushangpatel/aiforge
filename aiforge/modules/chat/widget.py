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
    """Research a topic using web search and/or ingested knowledge sources.

    Strategy:
    1. Search the web via DuckDuckGo Lite (no API key needed).
    2. Also search ingested knowledge sources via AIForge SearchAgent.
    3. Return combined results — web first, then knowledge base.

    Falls back gracefully if either source is unavailable.
    """
    sections: list[str] = []

    # ── Web search via DuckDuckGo Lite ──
    web_results = _web_search(query)
    if web_results:
        sections.append("=== Web Results ===\n" + web_results)

    # ── Knowledge base search ──
    kb_results = _knowledge_search(query)
    if kb_results:
        sections.append("=== Knowledge Base ===\n" + kb_results)

    if not sections:
        return f"No results found for: {query}"

    return "\n\n".join(sections)


def _web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo Lite (HTML). No API key required.

    Returns a formatted string of search results, or empty string on failure.

    Note:
        Results are parsed via regex against DuckDuckGo Lite's HTML structure.
        If DuckDuckGo changes its markup, parsing may silently return no
        results.  The function degrades gracefully (returns "").
    """
    import re
    import urllib.parse
    import urllib.request

    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AIForge/1.0 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")

        # Parse the DuckDuckGo Lite results page.
        # Each result is in a table row with class "result-link" (title)
        # and a subsequent snippet row.
        titles = re.findall(
            r'class="result-link"[^>]*>(.*?)</a>', html, re.DOTALL,
        )
        snippets = re.findall(
            r'class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL,
        )
        links = re.findall(
            r'class="result-link"\s+href="([^"]+)"', html,
        )

        if not titles:
            # Fallback: try a simpler pattern for result links
            titles = re.findall(r'<a[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
            links = re.findall(r'<a[^>]*class="[^"]*result[^"]*"\s+href="([^"]+)"', html)

        if not titles:
            return ""

        lines: list[str] = []
        for i, title in enumerate(titles[:max_results]):
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            link = links[i] if i < len(links) else ""
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            entry = f"[{i + 1}] {clean_title}"
            if link:
                entry += f"\n    {link}"
            if snippet:
                entry += f"\n    {snippet}"
            lines.append(entry)

        return "\n\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def _knowledge_search(query: str) -> str:
    """Search ingested knowledge sources via AIForge SemanticSearch."""
    try:
        from aiforge.modules.search.module import SemanticSearch

        ss = SemanticSearch()
        result = ss.query(query)
        if not result.hits:
            return ""

        lines: list[str] = []
        for i, hit in enumerate(result.hits[:5], 1):
            source = hit.metadata.get("source", "unknown")
            score = f" (score: {hit.score:.2f})" if hit.score else ""
            lines.append(f"[{i}] {source}{score}\n{hit.text[:500]}")
        return "\n\n".join(lines)
    except ImportError:
        return ""
    except Exception:  # noqa: BLE001
        return ""


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
            "Research a topic by searching the web (DuckDuckGo) and any "
            "previously-ingested knowledge sources. Returns web results "
            "and knowledge base matches. Use for factual questions, "
            "current events, technical lookups, or anything requiring "
            "external information."
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
