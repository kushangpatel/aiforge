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
# Home page (rendered via st.navigation)
# ---------------------------------------------------------------------------


def _home_page() -> None:
    """Render the home / landing page."""
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
        "Classify · Extract · Generate · Chat · Moderate · Translate · "
        "Vision · Pipeline"
    )


# ---------------------------------------------------------------------------
# Sidebar navigation (st.navigation replaces the auto-generated MPA sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ AIForge")
    st.caption("Plug-and-play AI toolkit")

pg = st.navigation(
    {
        "": [
            st.Page(_home_page, title="Home", icon="🏠", default=True),
        ],
        "Modules": [
            st.Page("pages/summarize.py", title="Summarize", icon="📝"),
            st.Page("pages/qa.py", title="DocQA", icon="❓"),
            st.Page("pages/search.py", title="Semantic Search", icon="🔍"),
            st.Page("pages/classify.py", title="Classify", icon="🏷️"),
            st.Page("pages/extract.py", title="Extract", icon="🧩"),
            st.Page("pages/generate.py", title="Generate", icon="✍️"),
            st.Page("pages/chat.py", title="Chat", icon="💬"),
            st.Page("pages/moderate.py", title="Moderate", icon="🛡️"),
            st.Page("pages/translate.py", title="Translate", icon="🌐"),
            st.Page("pages/vision.py", title="Vision", icon="👁️"),
            st.Page("pages/pipeline.py", title="Pipeline", icon="🔗"),
        ],
        "System": [
            st.Page("pages/settings.py", title="Settings", icon="⚙️"),
        ],
    }
)

pg.run()
