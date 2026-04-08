"""Streamlit widget for the DocQA (RAG) module."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the DocQA playground widget."""
    collection = st.text_input(
        "Collection name",
        value="playground",
        help="ChromaDB collection to use for ingestion and queries.",
    )
    mode = st.radio("Mode", ["Ask", "Ingest", "Collections"], horizontal=True)

    if mode == "Ingest":
        _render_ingest(collection)
    elif mode == "Collections":
        _render_collections(collection)
    else:
        _render_ask(collection)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _render_ingest(collection: str) -> None:
    st.subheader("Ingest Documents")
    source = st.text_input(
        "Source (file path, directory, or git URL)",
        placeholder="/path/to/codebase",
    )
    ext_filter = st.text_input(
        "Extension filter (comma-separated, optional)",
        placeholder=".py, .md, .ts",
    )

    if not st.button(
        "Ingest", type="primary", disabled=not source.strip() or not collection.strip()
    ):
        return

    from aiforge.modules.qa.module import DocQA

    file_extensions: list[str] | None = None
    if ext_filter.strip():
        file_extensions = [e.strip() for e in ext_filter.split(",") if e.strip()]

    with st.spinner("Ingesting documents…"):
        try:
            qa = DocQA(collection=collection)
            result = qa.ingest(source, file_extensions=file_extensions)
        except Exception as exc:
            st.error(f"Ingestion error: {exc}")
            return

    st.success(
        f"Indexed **{result.chunks_indexed}** chunks from "
        f"**{result.files_processed}** files into `{result.collection}`."
    )
    if result.skipped_files:
        with st.expander(f"Skipped files ({len(result.skipped_files)})"):
            st.write(result.skipped_files)
    with st.expander("Metrics"):
        st.json(
            {
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "model": result.model,
                "languages": result.languages,
            }
        )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def _render_ask(collection: str) -> None:
    st.subheader("Ask a Question")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    query = st.text_input("Question", placeholder="How does the auth middleware work?")
    col1, col2 = st.columns(2)
    with col1:
        filter_repo = st.text_input("Filter by source path (optional)")
    with col2:
        show_trace = st.checkbox("Show agent trace")

    if not st.button("Ask", type="primary", disabled=not query.strip() or not collection.strip()):
        # still render history
        _render_history()
        return

    from aiforge.modules.qa.module import DocQA

    with st.spinner("Thinking…"):
        try:
            qa = DocQA(collection=collection)
            result = qa.ask(
                query,
                filter_repo=filter_repo or None,
                include_trace=show_trace,
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
            _render_history()
            return

    st.session_state.qa_history.append({"query": query, "result": result})
    _render_history()


def _render_history() -> None:
    for entry in reversed(st.session_state.get("qa_history", [])):
        r = entry["result"]
        st.markdown(f"**Q:** {entry['query']}")
        st.markdown(f"**A:** {r.text}")
        st.caption(
            f"Confidence: {r.confidence:.0%} · "
            f"Sources: {', '.join(r.sources) if r.sources else '—'} · "
            f"Iterations: {r.iterations} · "
            f"Cost: ${r.cost:.4f}"
        )
        if r.trace:
            with st.expander("Agent Trace"):
                for step in r.trace:
                    st.json(step)
        st.divider()


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def _render_collections(collection: str) -> None:
    st.subheader("Collection Management")

    from aiforge.modules.qa.module import DocQA

    try:
        qa = DocQA(collection=collection)
        collections = qa.list_collections()
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    if not collections:
        st.info("No collections found.")
        return

    st.write(f"**{len(collections)}** collection(s) found:")
    for name in collections:
        with st.expander(name):
            try:
                stats = qa.collection_stats(name)
                st.metric("Chunks", stats.get("chunks", 0))
                st.json(stats)
            except Exception as exc:
                st.error(f"Could not load stats: {exc}")

            if st.button(f"Delete {name}", key=f"del_{name}"):
                qa.delete_collection(name)
                st.success(f"Deleted collection `{name}`.")
                st.rerun()
