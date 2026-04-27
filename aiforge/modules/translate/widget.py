"""Streamlit widget for the Translator module."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Translator playground widget."""
    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        source_language = st.text_input(
            "Source language",
            value="auto",
            help="Language of the input text, or 'auto' to detect.",
        )
    with col2:
        target_language = st.text_input(
            "Target language",
            value="Spanish",
            help="Language to translate into.",
        )

    col3, col4 = st.columns(2)
    with col3:
        formality = st.selectbox("Formality", ["neutral", "formal", "informal"])
    with col4:
        domain = st.text_input(
            "Domain (optional)",
            placeholder="e.g. medical, legal, technical",
        )

    st.subheader("Input")
    text = st.text_area(
        "Text to translate",
        height=200,
        placeholder="Enter text in any language…",
    )

    if not st.button("Translate", type="primary", disabled=not text.strip()):
        return

    with st.spinner("Translating..."):
        try:
            from aiforge.modules.translate.module import Translator

            t = Translator(
                target_language=target_language,
                source_language=source_language,
                formality=formality,
                domain=domain if domain.strip() else None,
            )
            result = t.translate(text)
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            return
        except Exception as exc:
            st.error(f"Translation failed: {exc}")
            return

    st.subheader("Translation")
    st.write(result.text)

    st.caption(
        f"Detected source: **{result.source_language}** → **{result.target_language}** ({result.formality})"
    )

    with st.expander("Metrics"):
        st.json(
            {
                "source_language": result.source_language,
                "target_language": result.target_language,
                "formality": result.formality,
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
