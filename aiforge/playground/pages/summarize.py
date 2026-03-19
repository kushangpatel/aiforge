"""Summarize playground page."""

import streamlit as st

st.set_page_config(page_title="Summarize — AIForge", page_icon="📝")
st.title("📝 Summarize")
st.caption("Text summarisation with automatic stuff / map-reduce strategy selection")

try:
    from aiforge.modules.summarize.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
