"""Settings page — full configuration dashboard."""

import streamlit as st

st.title("⚙️ Settings")
st.caption("Live view of the active AIForge configuration (read-only)")

# Tighten spacing: smaller metric labels, reduce expander/section gaps
st.markdown(
    """<style>
    [data-testid="stExpander"] { margin-bottom: 0.25rem; }
    div[data-testid="stMarkdownContainer"] table { font-size: 0.85rem; }
    </style>""",
    unsafe_allow_html=True,
)


def _row(label: str, value: object) -> str:
    """Build a single markdown table row."""
    return f"| {label} | {value} |"


def _table(rows: list[tuple[str, object]]) -> str:
    """Build a compact two-column markdown table."""
    lines = ["| Setting | Value |", "|:---|:---|"]
    lines.extend(_row(label, val) for label, val in rows)
    return "\n".join(lines)


def _flag(val: bool) -> str:  # noqa: FBT001
    return "✅" if val else "❌"


try:
    from aiforge.infra.config import get_settings
    from aiforge.infra.cost_tracker import CostTracker

    s = get_settings()

    # ------------------------------------------------------------------
    # Provider & Model
    # ------------------------------------------------------------------
    with st.expander("**Provider & Model**", expanded=True):
        st.markdown(
            _table(
                [
                    ("Provider", s.PROVIDER),
                    ("Model", s.MODEL),
                    ("Embedding Model", f"`{s.EMBEDDING_MODEL}`"),
                    ("Temperature", s.TEMPERATURE),
                    ("Max Tokens", s.MAX_TOKENS or "default"),
                ]
            )
        )

    # ------------------------------------------------------------------
    # Model Router
    # ------------------------------------------------------------------
    with st.expander("**Model Router**"):
        st.markdown(
            _table(
                [
                    ("Enabled", _flag(s.MODEL_ROUTER_ENABLED)),
                    ("Default Quality", s.DEFAULT_QUALITY),
                    ("Prefer Local", _flag(s.PREFER_LOCAL)),
                ]
            )
        )
        if s.CUSTOM_MODELS:
            st.json(s.CUSTOM_MODELS)

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------
    with st.expander("**Vector Store**"):
        rows: list[tuple[str, object]] = [("Backend", s.VECTOR_STORE)]
        if s.VECTOR_STORE == "chroma":
            rows.append(("Persist Dir", f"`{s.CHROMA_PERSIST_DIR}`"))
        elif s.VECTOR_STORE == "pgvector":
            rows.append(("Collection", s.PGVECTOR_COLLECTION))
        elif s.VECTOR_STORE == "pinecone":
            rows.append(("Index", s.PINECONE_INDEX or "—"))
        elif s.VECTOR_STORE == "qdrant":
            rows.append(("Collection", s.QDRANT_COLLECTION or "—"))
        st.markdown(_table(rows))

    # ------------------------------------------------------------------
    # Rate Limiting & Cost Tracking
    # ------------------------------------------------------------------
    with st.expander("**Rate Limiting & Cost Tracking**"):
        st.markdown(
            _table(
                [
                    ("Requests / min", f"{s.RATE_LIMIT_RPM:,}"),
                    ("Tokens / min", f"{s.RATE_LIMIT_TPM:,}"),
                    ("Cost Tracking", _flag(s.COST_TRACKING_ENABLED)),
                    ("Warning Threshold", f"${s.COST_WARN_THRESHOLD:.2f} / call"),
                ]
            )
        )

        tracker = CostTracker.get_session()
        if tracker.call_count > 0:
            st.markdown("---")
            st.markdown("**Session Usage**")
            st.markdown(
                _table(
                    [
                        ("Total Cost", f"${tracker.total_cost:.4f}"),
                        ("Total Tokens", f"{tracker.total_tokens:,}"),
                        ("API Calls", tracker.call_count),
                    ]
                )
            )
            if tracker.by_module:
                st.markdown("**By Module**")
                st.json(dict(tracker.by_module))
            if tracker.by_model:
                st.markdown("**By Model**")
                st.json(dict(tracker.by_model))

    # ------------------------------------------------------------------
    # Request Cache
    # ------------------------------------------------------------------
    with st.expander("**Request Cache**"):
        st.markdown(
            _table(
                [
                    ("Enabled", _flag(s.CACHE_ENABLED)),
                    ("Backend", s.CACHE_BACKEND),
                    ("TTL", f"{s.CACHE_TTL}s"),
                    ("Max Entries", f"{s.CACHE_MAX_ENTRIES:,}"),
                ]
            )
        )

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------
    with st.expander("**Webhooks**"):
        st.markdown(
            _table(
                [
                    ("Enabled", _flag(s.WEBHOOK_ENABLED)),
                    ("Timeout", f"{s.WEBHOOK_TIMEOUT}s"),
                    ("Allow Insecure", "⚠️ Yes" if s.WEBHOOK_ALLOW_INSECURE else "No"),
                ]
            )
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    with st.expander("**Observability**"):
        rows = [
            ("Tracing", _flag(s.TRACING_ENABLED)),
            ("Tracing Backend", s.TRACING_BACKEND),
            ("Log Level", s.LOG_LEVEL),
            ("Log Format", s.LOG_FORMAT),
        ]
        if s.TRACING_BACKEND == "langsmith":
            rows.append(("LangSmith Project", s.LANGSMITH_PROJECT))
        st.markdown(_table(rows))

    # ------------------------------------------------------------------
    # API Server
    # ------------------------------------------------------------------
    with st.expander("**API Server**"):
        st.markdown(
            _table(
                [
                    ("Host", f"`{s.API_HOST}`"),
                    ("Port", s.API_PORT),
                    ("Auth", "🔒 Key set" if s.API_KEY else "🔓 Open"),
                ]
            )
        )

    # ------------------------------------------------------------------
    # API Keys (presence only — never show values)
    # ------------------------------------------------------------------
    with st.expander("**API Keys**"):
        st.caption("Values are never displayed — only whether each key is set.")
        keys = {
            "OpenAI": s.OPENAI_API_KEY,
            "Anthropic": s.ANTHROPIC_API_KEY,
            "Voyage AI": s.VOYAGE_API_KEY,
            "Azure OpenAI": s.AZURE_OPENAI_API_KEY,
            "Google Gemini": s.GOOGLE_API_KEY,
            "Mistral": s.MISTRAL_API_KEY,
            "Pinecone": s.PINECONE_API_KEY,
            "Qdrant": s.QDRANT_API_KEY,
            "LangSmith": s.LANGSMITH_API_KEY,
        }
        st.markdown(_table([(k, "✅ Set" if v else "—") for k, v in keys.items()]))

    # ------------------------------------------------------------------
    # Custom Overrides (only shown when configured)
    # ------------------------------------------------------------------
    if s.CUSTOM_PRICING or s.CUSTOM_CONTEXT_WINDOWS or s.CUSTOM_PROMPTS:
        with st.expander("**Custom Overrides**"):
            if s.CUSTOM_PRICING:
                st.markdown("**Pricing**")
                st.json(s.CUSTOM_PRICING)
            if s.CUSTOM_CONTEXT_WINDOWS:
                st.markdown("**Context Windows**")
                st.json(s.CUSTOM_CONTEXT_WINDOWS)
            if s.CUSTOM_PROMPTS:
                st.markdown("**Prompts**")
                for module, prompts in s.CUSTOM_PROMPTS.items():
                    st.markdown(f"*{module}*")
                    st.json(prompts)

except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load settings: {exc}")
