"""ChatAgent playground page."""

import streamlit as st

st.set_page_config(page_title="Chat — AIForge", page_icon="💬")
st.title("💬 Chat")
st.caption("Multi-turn conversational agent with memory and tool use")

try:
    from aiforge.modules.chat.widget import render
    render()
except ImportError as exc:
    st.error(f"Module unavailable: {exc}")

