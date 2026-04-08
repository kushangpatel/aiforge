"""Pipeline page."""

import streamlit as st

st.title("🔗 Pipeline")
st.caption("Build and run multi-step AI pipelines by chaining modules together.")

try:
    from aiforge.pipeline.widget import render  # noqa: E402

    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
