"""
AIForge Streamlit Playground — main entry point.

Run with:
    aiforge playground          # via CLI
    streamlit run aiforge/playground/app.py   # direct
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AIForge Playground",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "🏠 Home": None,
    "📝 Summarize": "pages/summarize.py",
    "❓ DocQA": "pages/qa.py",
    "🔍 Semantic Search": "pages/search.py",
    "🏷️ Classify": "pages/classify.py",
    "🧩 Extract": "pages/extract.py",
    "✍️ Generate": "pages/generate.py",
    "💬 Chat": "pages/chat.py",
    "🛡️ Moderate": "pages/moderate.py",
    "🔗 Pipeline": "pages/pipeline.py",
    "⚙️ Settings": "pages/settings.py",
}

with st.sidebar:
    st.title("⚡ AIForge")
    st.caption("Plug-and-play AI toolkit")
    st.divider()
    selected = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

if selected == "🏠 Home" or PAGES.get(selected) is None:
    st.title("⚡ AIForge Playground")
    st.write(
        "Welcome to the AIForge interactive playground. "
        "Select a module from the sidebar to get started."
    )

    from aiforge.__version__ import __version__
    from aiforge.infra.config import get_settings

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Version", __version__)
    try:
        settings = get_settings()
        with col2:
            st.metric("Provider", settings.PROVIDER)
        with col3:
            st.metric("Model", settings.MODEL)
    except Exception:  # noqa: BLE001
        with col2:
            st.metric("Provider", "not configured")

    st.divider()
    st.info(
        "**Available modules:** Summarize · DocQA · Semantic Search · "
        "Classify · Extract · Generate · Chat · Moderate · Pipeline"
    )
