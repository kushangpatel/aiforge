"""Moderator playground page."""

import streamlit as st

st.set_page_config(page_title="Moderate — AIForge", page_icon="🛡️")
st.title("🛡️ Moderate")
st.caption("Content safety, PII detection, and prompt injection scanning")

try:
    from aiforge.modules.moderate.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")

