"""Streamlit widget for the Classifier module."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Classifier playground widget."""
    st.subheader("Configuration")
    raw_labels = st.text_area(
        "Labels (one per line)",
        value="positive\nnegative\nneutral",
        height=110,
        help="Enter each classification label on its own line.",
    )
    labels = [lbl.strip() for lbl in raw_labels.splitlines() if lbl.strip()]

    col1, col2 = st.columns(2)
    with col1:
        multi_label = st.checkbox("Multi-label mode", value=False)
    with col2:
        show_examples = st.checkbox("Add few-shot examples", value=False)

    examples_raw = ""
    if show_examples:
        examples_raw = st.text_area(
            "Few-shot examples (format: `text | label`, one per line)",
            height=120,
            placeholder="The match was thrilling. | sports\nThe vote passed. | politics",
        )

    st.subheader("Input")
    text = st.text_area(
        "Text to classify",
        height=140,
        placeholder="Enter the text you want to classify...",
    )

    if not st.button("Classify", type="primary", disabled=not text.strip() or not labels):
        return

    from aiforge.modules.classify.module import Classifier
    from aiforge.modules.classify.schemas import ClassifyExample

    examples: list[ClassifyExample] = []
    malformed_lines = 0
    dropped_labels = 0
    if show_examples and examples_raw.strip():
        for line in examples_raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                malformed_lines += 1
                continue
            ex_text, ex_label = parts[0].strip(), parts[1].strip()
            if not ex_text or not ex_label:
                malformed_lines += 1
                continue
            if ex_label in labels:
                examples.append(ClassifyExample(text=ex_text, label=ex_label))
            else:
                dropped_labels += 1
    if malformed_lines:
        st.warning(
            f"{malformed_lines} example line(s) skipped — use format: `text | label`"
        )
    if dropped_labels:
        st.warning(
            f"{dropped_labels} example(s) ignored — label not found in taxonomy"
        )

    with st.spinner("Classifying..."):
        try:
            c = Classifier(
                labels=labels,
                examples=examples or None,
                multi_label=multi_label,
            )
            result = c.classify(text)
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            return

    st.subheader("Result")
    if result.label:
        st.success(f"**Label:** `{result.label}`")
    else:
        st.warning("Could not determine a label from the provided taxonomy.")

    if result.rationale:
        st.caption(f"Rationale: {result.rationale}")

    if result.scores:
        st.subheader("Scores")
        st.table(
            [
                {"Label": s.label, "Confidence": f"{s.confidence:.0%}"}
                for s in result.scores
            ]
        )

    with st.expander("Metrics"):
        st.json(
            {
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
