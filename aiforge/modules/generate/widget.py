"""Streamlit widget for the Generator module."""

from __future__ import annotations

import re

import streamlit as st


def render() -> None:
    """Render the Generator playground widget."""
    st.subheader("Configuration")

    mode = st.radio("Mode", ["Template", "Freeform"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max tokens", 64, 4096, 1024)
    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

    guardrails = st.multiselect(
        "Guardrails",
        ["no_pii", "no_competitor_mentions", "no_pricing", "professional_tone"],
    )

    st.subheader("Input")

    if mode == "Template":
        template = st.text_area(
            "Template (use {variable_name} for placeholders)",
            value="Write a {tone} description for {product}.",
            height=120,
        )
        detected = re.findall(r"\{(\w+)\}", template)
        if detected:
            st.caption(f"Detected variables: {detected}")

        variables: dict[str, str] = {}
        for var in detected:
            variables[var] = st.text_input(f"Value for {{{var}}}")

        ready = bool(template.strip())
        if not st.button("Generate", type="primary", disabled=not ready):
            return

        with st.spinner("Generating..."):
            try:
                from aiforge.modules.generate.module import Generator

                g = Generator(
                    template=template,
                    guardrails=guardrails,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = g.generate(**variables)
            except ValueError as exc:
                st.error(f"Configuration error: {exc}")
                return
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

    else:
        prompt = st.text_area(
            "Prompt",
            height=200,
            placeholder="Describe what you want to generate…",
        )

        if not st.button("Generate", type="primary", disabled=not prompt.strip()):
            return

        with st.spinner("Generating..."):
            try:
                from aiforge.modules.generate.module import Generator

                g = Generator(
                    guardrails=guardrails,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = g.generate(prompt=prompt)
            except ValueError as exc:
                st.error(f"Configuration error: {exc}")
                return
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

    st.subheader("Generated Content")
    st.write(result.text)

    if result.guardrail_flags:
        st.warning(f"Guardrail violations: {result.guardrail_flags}")

    with st.expander("Metrics"):
        st.json(
            {
                "retry_count": result.retry_count,
                "guardrail_flags": result.guardrail_flags,
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
