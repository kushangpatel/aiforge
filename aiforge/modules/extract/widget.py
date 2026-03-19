"""Streamlit widget for the Extractor module."""

from __future__ import annotations

import json

import streamlit as st


def render() -> None:
    """Render the Extractor playground widget."""
    st.subheader("Configuration")

    schema_json = st.text_area(
        "Output Schema (JSON Schema)",
        value='{\n  "type": "object",\n  "properties": {\n'
        '    "name": {"type": "string"},\n'
        '    "age": {"type": "integer"}\n'
        "  },\n"
        '  "required": ["name", "age"]\n}',
        height=180,
        help="Paste a JSON Schema describing the structure you want to extract.",
    )

    show_examples = st.checkbox("Add few-shot examples", value=False)
    examples_raw = ""
    if show_examples:
        examples_raw = st.text_area(
            "Few-shot examples (JSON array of {input, output})",
            height=140,
            placeholder='[\n  {\n    "input": "John is 30",\n    "output": {"name": "John", "age": 30}\n  }\n]',
        )

    st.subheader("Input")
    text = st.text_area(
        "Text to extract from",
        height=180,
        placeholder="Paste the text you want to extract structured data from...",
    )

    if not st.button("Extract", type="primary", disabled=not text.strip()):
        return

    # Parse the schema
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON Schema: {exc}")
        return

    # Parse examples
    from aiforge.modules.extract.schemas import ExtractExample

    examples: list[ExtractExample] = []
    if show_examples and examples_raw.strip():
        try:
            raw_list = json.loads(examples_raw)
            if not isinstance(raw_list, list):
                st.error("Examples must be a JSON array.")
                return
            for item in raw_list:
                examples.append(
                    ExtractExample(input=item["input"], output=item["output"])
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            st.error(f"Invalid examples format: {exc}")
            return

    with st.spinner("Extracting..."):
        try:
            from aiforge.modules.extract.module import Extractor

            e = Extractor(output_schema=schema, examples=examples or None)
            result = e.extract(text)
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
            return

    st.subheader("Extracted Data")
    if result.validation_passed:
        st.success("Validation passed")
        st.json(result.data)
    else:
        st.warning("Validation failed — raw LLM output shown below")
        st.code(result.raw_json, language="json")

    with st.expander("Metrics"):
        st.json(
            {
                "validation_passed": result.validation_passed,
                "retry_count": result.retry_count,
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
            }
        )
