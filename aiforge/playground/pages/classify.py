"""Classify playground page."""

import streamlit as st

st.title("🏷️ Classify")
st.caption("Zero/few-shot text classification against custom label taxonomies")

try:
    from aiforge.modules.classify.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
