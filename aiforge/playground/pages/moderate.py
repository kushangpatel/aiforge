"""Moderator playground page."""

import streamlit as st

st.title("🛡️ Moderate")
st.caption("Content safety, PII detection, and prompt injection scanning")

try:
    from aiforge.modules.moderate.widget import render

    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
