"""Generate playground page."""

import streamlit as st

st.set_page_config(page_title="Generate — AIForge", page_icon="✨")
st.title("✨ Generate")
st.caption("Template-based and freeform content generation with guardrails")

try:
    from aiforge.modules.generate.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
