"""Vision playground page."""

import streamlit as st

st.title("👁️ Vision")
st.caption("Analyze images — describe, answer questions, extract data, or classify")

try:
    from aiforge.modules.vision.widget import render

    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
