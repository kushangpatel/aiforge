"""Settings page."""
import streamlit as st

st.title("⚙️ Settings")

try:
    from aiforge.infra.config import get_settings
    s = get_settings()
    st.json({
        "PROVIDER": s.PROVIDER,
        "MODEL": s.MODEL,
        "EMBEDDING_MODEL": s.EMBEDDING_MODEL,
        "TEMPERATURE": s.TEMPERATURE,
        "MAX_TOKENS": s.MAX_TOKENS,
        "RATE_LIMIT_RPM": s.RATE_LIMIT_RPM,
        "RATE_LIMIT_TPM": s.RATE_LIMIT_TPM,
        "TRACING_ENABLED": s.TRACING_ENABLED,
    })
except Exception as exc:
    st.error(f"Could not load settings: {exc}")
