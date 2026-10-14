"""Streamlit widget for the Summarizer module."""

from __future__ import annotations

from typing import Any, cast

import streamlit as st


def render() -> None:
    """Render the Summarizer playground widget."""
    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox(
            "Style",
            ["concise", "detailed", "bullet_points", "executive"],
            index=0,
        )
    with col2:
        max_length = st.slider("Max length (words)", 50, 2000, 500)

    strategy = st.selectbox(
        "Strategy",
        [None, "stuff", "map_reduce"],
        format_func=lambda x: "Auto-detect" if x is None else x,
    )

    st.subheader("Input")
    text = st.text_area(
        "Text to summarize",
        height=300,
        placeholder="Paste the text you want to summarize...",
    )

    if not st.button("Summarize", type="primary", disabled=not text.strip()):
        return

    with st.spinner("Summarizing..."):
        try:
            from aiforge.modules.summarize.module import Summarizer

            s = Summarizer(style=cast(Any, style), max_length=max_length)
            result = s.summarize(text, strategy=strategy)
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            return
        except Exception as exc:
            st.error(f"Summarization failed: {exc}")
            return

    st.subheader("Summary")
    st.write(result.summary)

    with st.expander("Metrics"):
        st.json(
            {
                "strategy_used": result.strategy_used,
                "chunks_processed": result.chunks_processed,
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
