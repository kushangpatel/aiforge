"""Extract playground page."""

import streamlit as st

st.set_page_config(page_title="Extract — AIForge", page_icon="🧩")
st.title("🧩 Extract")
st.caption("Schema-driven structured data extraction with validation and retry")

try:
    from aiforge.modules.extract.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
