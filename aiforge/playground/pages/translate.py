"""Translate playground page."""

import streamlit as st

st.title("🌐 Translate")
st.caption("Translate text between languages with formality and domain controls")

try:
    from aiforge.modules.translate.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
