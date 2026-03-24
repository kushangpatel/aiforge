"""DocQA playground page."""

import streamlit as st

st.title("❓ DocQA")
st.caption("Document Q&A with RAG pipeline and agentic refinement")

try:
    from aiforge.modules.qa.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")
