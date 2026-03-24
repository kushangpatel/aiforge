"""Semantic Search playground page."""

import streamlit as st

st.title("🔍 Semantic Search")
st.caption("Embed, index, and semantically query any text corpus")

try:
    from aiforge.modules.search.widget import render

    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")

