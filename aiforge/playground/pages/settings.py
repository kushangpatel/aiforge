"""Settings page — full configuration dashboard."""
import streamlit as st

st.title("⚙️ Settings")
st.caption("Live view of the active AIForge configuration (read-only)")

try:
    from aiforge.infra.config import get_settings
    from aiforge.infra.cost_tracker import CostTracker

    s = get_settings()

    # ------------------------------------------------------------------
    # Provider & Model
    # ------------------------------------------------------------------
    st.header("Provider & Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provider", s.PROVIDER)
    col2.metric("Model", s.MODEL)
    col3.metric("Temperature", s.TEMPERATURE)
    col4.metric("Max Tokens", s.MAX_TOKENS or "default")

    st.text_input("Embedding Model", value=s.EMBEDDING_MODEL, disabled=True)

    # ------------------------------------------------------------------
    # Model Router
    # ------------------------------------------------------------------
    st.header("Model Router")
    col1, col2, col3 = st.columns(3)
    col1.metric("Enabled", "✅" if s.MODEL_ROUTER_ENABLED else "❌")
    col2.metric("Default Quality", s.DEFAULT_QUALITY)
    col3.metric("Prefer Local", "✅" if s.PREFER_LOCAL else "❌")

    if s.CUSTOM_MODELS:
        with st.expander("Custom Models"):
            st.json(s.CUSTOM_MODELS)

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------
    st.header("Vector Store")
    col1, col2 = st.columns(2)
    col1.metric("Backend", s.VECTOR_STORE)
    if s.VECTOR_STORE == "chroma":
        col2.metric("Persist Dir", s.CHROMA_PERSIST_DIR)
    elif s.VECTOR_STORE == "pgvector":
        col2.metric("Collection", s.PGVECTOR_COLLECTION)
    elif s.VECTOR_STORE == "pinecone":
        col2.metric("Index", s.PINECONE_INDEX or "—")
    elif s.VECTOR_STORE == "qdrant":
        col2.metric("Collection", s.QDRANT_COLLECTION or "—")

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------
    st.header("Rate Limiting")
    col1, col2 = st.columns(2)
    col1.metric("Requests / min", f"{s.RATE_LIMIT_RPM:,}")
    col2.metric("Tokens / min", f"{s.RATE_LIMIT_TPM:,}")

    # ------------------------------------------------------------------
    # Cost Tracking
    # ------------------------------------------------------------------
    st.header("Cost Tracking")
    col1, col2 = st.columns(2)
    col1.metric("Enabled", "✅" if s.COST_TRACKING_ENABLED else "❌")
    col2.metric("Warning Threshold", f"${s.COST_WARN_THRESHOLD:.2f} / call")

    tracker = CostTracker.get_session()
    if tracker.call_count > 0:
        st.subheader("Session Usage")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"${tracker.total_cost:.4f}")
        col2.metric("Total Tokens", f"{tracker.total_tokens:,}")
        col3.metric("API Calls", tracker.call_count)

        if tracker.by_module:
            with st.expander("Cost by Module"):
                st.json(dict(tracker.by_module))
        if tracker.by_model:
            with st.expander("Cost by Model"):
                st.json(dict(tracker.by_model))

    # ------------------------------------------------------------------
    # Request Cache
    # ------------------------------------------------------------------
    st.header("Request Cache")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Enabled", "✅" if s.CACHE_ENABLED else "❌")
    col2.metric("Backend", s.CACHE_BACKEND)
    col3.metric("TTL", f"{s.CACHE_TTL}s")
    col4.metric("Max Entries", f"{s.CACHE_MAX_ENTRIES:,}")

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------
    st.header("Webhooks")
    col1, col2, col3 = st.columns(3)
    col1.metric("Enabled", "✅" if s.WEBHOOK_ENABLED else "❌")
    col2.metric("Timeout", f"{s.WEBHOOK_TIMEOUT}s")
    col3.metric("Allow Insecure", "⚠️ Yes" if s.WEBHOOK_ALLOW_INSECURE else "No")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    st.header("Observability")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tracing", "✅" if s.TRACING_ENABLED else "❌")
    col2.metric("Backend", s.TRACING_BACKEND)
    col3.metric("Log Level", s.LOG_LEVEL)

    col1, col2 = st.columns(2)
    col1.metric("Log Format", s.LOG_FORMAT)
    if s.TRACING_BACKEND == "langsmith":
        col2.metric("LangSmith Project", s.LANGSMITH_PROJECT)

    # ------------------------------------------------------------------
    # API Server
    # ------------------------------------------------------------------
    st.header("API Server")
    col1, col2, col3 = st.columns(3)
    col1.metric("Host", s.API_HOST)
    col2.metric("Port", s.API_PORT)
    col3.metric("Auth", "🔒 Key set" if s.API_KEY else "🔓 Open")

    # ------------------------------------------------------------------
    # API Keys (presence only — never show values)
    # ------------------------------------------------------------------
    st.header("API Keys")
    st.caption("Shows whether each key is configured — values are never displayed.")
    keys = {
        "OpenAI": s.OPENAI_API_KEY,
        "Anthropic": s.ANTHROPIC_API_KEY,
        "Voyage AI": s.VOYAGE_API_KEY,
        "Azure OpenAI": s.AZURE_OPENAI_API_KEY,
        "Pinecone": s.PINECONE_API_KEY,
        "Qdrant": s.QDRANT_API_KEY,
        "LangSmith": s.LANGSMITH_API_KEY,
    }
    cols = st.columns(4)
    for i, (name, val) in enumerate(keys.items()):
        cols[i % 4].metric(name, "✅ Set" if val else "—")

    # ------------------------------------------------------------------
    # Custom Pricing Overrides
    # ------------------------------------------------------------------
    if s.CUSTOM_PRICING or s.CUSTOM_CONTEXT_WINDOWS:
        st.header("Custom Overrides")
        if s.CUSTOM_PRICING:
            with st.expander("Pricing Overrides"):
                st.json(s.CUSTOM_PRICING)
        if s.CUSTOM_CONTEXT_WINDOWS:
            with st.expander("Context Window Overrides"):
                st.json(s.CUSTOM_CONTEXT_WINDOWS)

    # ------------------------------------------------------------------
    # Custom Prompts
    # ------------------------------------------------------------------
    if s.CUSTOM_PROMPTS:
        st.header("Custom Prompts")
        for module, prompts in s.CUSTOM_PROMPTS.items():
            with st.expander(f"Module: {module}"):
                st.json(prompts)

except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load settings: {exc}")
