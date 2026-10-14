"""Streamlit widget for the Vision module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import streamlit as st


def render() -> None:
    """Render the Vision playground widget."""
    st.subheader("Configuration")

    task = st.selectbox("Task", ["describe", "qa", "extract", "classify"])

    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max tokens", 64, 4096, 1024)
    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.3, step=0.1)

    st.subheader("Image")

    source_type = st.radio("Source", ["Upload", "URL"], horizontal=True)

    image_path: str | None = None
    if source_type == "Upload":
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "gif", "webp"],
        )
        if uploaded:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
    else:
        url = st.text_input("Image URL", placeholder="https://example.com/photo.jpg")
        if url.strip():
            image_path = url.strip()
            st.image(url, caption="Image from URL", use_container_width=True)

    # Task-specific inputs
    question = None
    focus = None
    labels = None

    if task == "qa":
        question = st.text_input("Question about the image")
    elif task == "describe":
        focus = st.text_input("Focus area (optional)", placeholder="e.g. text, colors, people")
    elif task == "classify":
        labels_input = st.text_input(
            "Labels (comma-separated)",
            placeholder="e.g. landscape, portrait, product, document",
        )
        if labels_input.strip():
            labels = [lb.strip() for lb in labels_input.split(",") if lb.strip()]

    ready = bool((source_type == "Upload" and uploaded) or (source_type == "URL" and image_path))
    if task == "qa" and not question:
        ready = False
    if task == "classify" and not labels:
        ready = False

    if not st.button("Analyze", type="primary", disabled=not ready):
        return

    # Write uploaded file to temp path
    if source_type == "Upload" and uploaded:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
        image_path = tmp.name

    if not image_path:
        return

    with st.spinner("Analyzing image..."):
        try:
            from aiforge.modules.vision.module import Vision

            v = Vision(
                task=cast(Any, task),  # selectbox options match the Literal
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = v.analyze(
                image_path,
                question=question,
                focus=focus if focus and focus.strip() else None,
                labels=labels,
            )
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            return
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    st.subheader("Result")
    st.write(result.text)

    with st.expander("Metrics"):
        st.json(
            {
                "task": result.task,
                "image_source": result.image_source,
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
