"""Streamlit widget for the SemanticSearch module."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the SemanticSearch playground widget."""
    st.subheader("Configuration")
    collection = st.text_input(
        "Collection name",
        value="playground",
        help="ChromaDB collection to index into or query from.",
    )
    mode = st.radio("Mode", ["Search", "Index"], horizontal=True)

    if mode == "Index":
        _render_index(collection)
    else:
        _render_search(collection)


def _render_index(collection: str) -> None:
    st.subheader("Documents to Index")
    st.caption("One document per line — format: `id | text`")
    docs_raw = st.text_area(
        "Documents",
        height=160,
        placeholder=(
            "prod-1 | Wireless noise-cancelling headphones\n"
            "prod-2 | Portable Bluetooth speaker with bass boost"
        ),
    )

    if not st.button(
        "Index", type="primary", disabled=not docs_raw.strip() or not collection.strip()
    ):
        return

    from aiforge.modules.search.module import SemanticSearch
    from aiforge.modules.search.schemas import SearchDocument

    documents: list[SearchDocument] = []
    malformed = 0
    for line in docs_raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            malformed += 1
            continue
        doc_id, doc_text = parts[0].strip(), parts[1].strip()
        if not doc_id or not doc_text:
            malformed += 1
            continue
        documents.append(SearchDocument(id=doc_id, text=doc_text))

    if malformed:
        st.warning(f"{malformed} line(s) skipped — use format: `id | text`")
    if not documents:
        st.error("No valid documents to index.")
        return

    with st.spinner(f"Indexing {len(documents)} document(s)..."):
        try:
            ss = SemanticSearch(collection=collection)
            result = ss.index(documents)
        except Exception as exc:
            st.error(f"Indexing error: {exc}")
            return

    st.success(
        f"Indexed **{result.indexed_count}** document(s) into "
        f"`{result.collection}` ({result.total_documents} total)."
    )
    with st.expander("Metrics"):
        st.json(
            {
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "model": result.model,
            }
        )


def _render_search(collection: str) -> None:
    st.subheader("Search")
    query = st.text_input(
        "Search query",
        placeholder="something to listen to music on the go",
    )
    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

    if not st.button(
        "Search", type="primary", disabled=not query.strip() or not collection.strip()
    ):
        return

    from aiforge.modules.search.module import SemanticSearch

    with st.spinner("Searching..."):
        try:
            ss = SemanticSearch(collection=collection)
            result = ss.query(query, top_k=top_k)
        except Exception as exc:
            st.error(f"Search error: {exc}")
            return

    if not result.hits:
        st.info("No results found. Try indexing some documents first.")
        return

    st.subheader(f"Results ({len(result.hits)} hits)")
    for i, hit in enumerate(result.hits, 1):
        with st.expander(f"#{i} — Score: {hit.score:.3f} | ID: `{hit.id}`"):
            st.write(hit.text)
            if hit.metadata:
                st.json(hit.metadata)

    with st.expander("Metrics"):
        st.json(
            {
                "latency_ms": round(result.latency_ms, 1),
                "cost_usd": result.cost,
                "input_tokens": result.input_tokens,
                "model": result.model,
                "total_searched": result.total_searched,
            }
        )
