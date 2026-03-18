"""
SemanticSearch module — embed, index, and semantically query text via ChromaDB.

No LLM calls are made; only embedding calls.  The module is fully usable
inside an AIForge :class:`~aiforge.core.pipeline.Pipeline` via ``run()``
and ``arun()``, which delegate to :meth:`query` and :meth:`aquery`.

Usage::

    from aiforge import SemanticSearch

    ss = SemanticSearch(collection="products")

    # Index documents (dicts or SearchDocument instances)
    resp = ss.index([
        {"id": "1", "text": "Wireless noise-cancelling headphones", "metadata": {"price": 299}},
        {"id": "2", "text": "Portable Bluetooth speaker", "metadata": {"price": 79}},
        {"id": "3", "text": "USB-C fast charging cable", "metadata": {"price": 12}},
    ])
    print(resp.indexed_count)   # 3
    print(resp.total_documents) # 3

    # Semantic search
    results = ss.query("something to listen to music on the go", top_k=3)
    for hit in results.hits:
        print(f"{hit.score:.3f} | {hit.text}")
    # 0.923 | Portable Bluetooth speaker ...
    # 0.891 | Wireless noise-cancelling headphones ...

    # Collection management
    print(ss.list_collections())            # ["products"]
    stats = ss.collection_stats()           # {"name": "products", "chunks": 3, ...}
    ss.delete_collection()                  # drop the collection
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from aiforge.core.pricing import calculate_cost
from aiforge.core.types import BaseModule
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.modules.search.schemas import (
    IndexResponse,
    SearchDocument,
    SearchHit,
    SearchResponse,
)

_TOP_K_MAX = 100


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _coerce_document(raw: dict | SearchDocument) -> SearchDocument:
    """Coerce a raw dict into a :class:`SearchDocument`."""
    if isinstance(raw, SearchDocument):
        return raw
    return SearchDocument(**raw)


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, sum(len(t) for t in texts) // 4)


# ---------------------------------------------------------------------------
# SemanticSearch class
# ---------------------------------------------------------------------------


class SemanticSearch(BaseModule):
    """Semantic search module backed by ChromaDB (default) or any VectorStoreAdapter.

    Embed documents once with :meth:`index`, then retrieve similar ones with
    :meth:`query`.  All embedding costs are tracked and returned in every
    response.

    Args:
        collection:   ChromaDB collection name.  Leading/trailing whitespace
                      is stripped.
        persist_dir:  Directory for ChromaDB's SQLite backing store.
        top_k:        Default number of results returned per query.  Can be
                      overridden per-call.
        store:        Optional pre-built :class:`VectorStoreAdapter`.
                      Primarily useful for dependency injection in tests —
                      pass a mock here instead of hitting a real DB.
        **kwargs:     Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        collection: str = "default",
        persist_dir: str = "./aiforge_data",
        top_k: int = 5,
        store: VectorStoreAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        stripped = (collection or "").strip()
        if not stripped:
            raise ValueError("collection must not be empty or whitespace-only")
        if top_k < 1 or top_k > _TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {_TOP_K_MAX}")

        self.collection_name: str = stripped
        self.persist_dir: str = persist_dir
        self.default_top_k: int = top_k
        self._store: VectorStoreAdapter = store or self._build_store()

    # ------------------------------------------------------------------
    # Public API — indexing
    # ------------------------------------------------------------------

    def index(
        self,
        documents: list[dict | SearchDocument],
        *,
        batch_size: int = 100,
    ) -> IndexResponse:
        """Embed and upsert *documents* into the collection.

        Documents with the same ``id`` are updated (upsert semantics).

        Args:
            documents:  :class:`SearchDocument` instances or plain dicts with
                        ``id``, ``text``, and optional ``metadata`` keys.
            batch_size: Documents are embedded in batches; adjust if provider
                        rate limits are hit.

        Returns:
            :class:`IndexResponse` with counts and cost.

        Raises:
            ValueError: if *documents* is empty.
        """
        if not documents:
            raise ValueError("documents must not be empty")

        docs = [_coerce_document(d) for d in documents]
        t0 = time.monotonic()
        total_tokens = 0
        embedding_model = ""

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.text for d in batch]
            try:
                embeddings = self._call_embedding(texts)
            except Exception as exc:
                self.logger.error(
                    "search.index.embedding_failed",
                    error=str(exc),
                    num_texts=len(texts),
                    batch_index=i // batch_size,
                )
                raise
            total_tokens += _estimate_tokens(texts)
            embedding_model = self.embedding_provider.model_name
            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=[d.metadata for d in batch],
                ids=[d.id for d in batch],
            )

        stats = self._store.collection_stats(self.collection_name)
        cost = calculate_cost(
            embedding_model,
            total_tokens,
            0,  # Embeddings have no output token cost
        )

        return IndexResponse(
            indexed_count=len(docs),
            collection=self.collection_name,
            total_documents=stats.get("chunks", 0),
            cost=cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aindex(
        self,
        documents: list[dict | SearchDocument],
        *,
        batch_size: int = 100,
    ) -> IndexResponse:
        """Async version of :meth:`index`."""
        if not documents:
            raise ValueError("documents must not be empty")

        docs = [_coerce_document(d) for d in documents]
        t0 = time.monotonic()
        total_tokens = 0
        embedding_model = ""

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.text for d in batch]
            try:
                embeddings = await self._acall_embedding(texts)
            except Exception as exc:
                self.logger.error(
                    "search.aindex.embedding_failed",
                    error=str(exc),
                    num_texts=len(texts),
                    batch_index=i // batch_size,
                )
                raise
            total_tokens += _estimate_tokens(texts)
            embedding_model = self.embedding_provider.model_name
            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=[d.metadata for d in batch],
                ids=[d.id for d in batch],
            )

        stats = self._store.collection_stats(self.collection_name)
        cost = calculate_cost(
            embedding_model,
            total_tokens,
            0,  # Embeddings have no output token cost
        )

        return IndexResponse(
            indexed_count=len(docs),
            collection=self.collection_name,
            total_documents=stats.get("chunks", 0),
            cost=cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def index_files(
        self,
        source: str | list[str],
        file_extensions: list[str] | None = None,
    ) -> IndexResponse:
        """Load files from *source* and index them.

        Each file becomes one :class:`SearchDocument`.  IDs are generated
        deterministically from the file path.  Files that cannot be loaded
        are skipped with a warning.

        Args:
            source:          File path, directory path, or list of file paths.
            file_extensions: Extension whitelist (e.g. ``[".txt", ".md"]``).
                             ``None`` → all supported types.

        Returns:
            :class:`IndexResponse` (``indexed_count=0`` if nothing was loaded).
        """
        from aiforge.core.doc_loader import DocLoader

        loader = DocLoader(file_extensions=file_extensions)
        lang_docs, _ = loader.load(source)

        if not lang_docs:
            self.logger.warning(
                "search.index_files.no_documents_loaded", source=str(source)
            )
            stats = self._store.collection_stats(self.collection_name)
            return IndexResponse(
                indexed_count=0,
                collection=self.collection_name,
                total_documents=stats.get("chunks", 0),
            )

        search_docs = [
            SearchDocument(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        doc.metadata.get("source", "") + doc.page_content[:64],
                    )
                ),
                text=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in lang_docs
            if doc.page_content.strip()
        ]

        if not search_docs:
            stats = self._store.collection_stats(self.collection_name)
            return IndexResponse(
                indexed_count=0,
                collection=self.collection_name,
                total_documents=stats.get("chunks", 0),
            )

        return self.index(search_docs)

    # ------------------------------------------------------------------
    # Public API — querying
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict | None = None,
        include_embeddings: bool = False,
    ) -> SearchResponse:
        """Embed *query* and retrieve the most similar documents.

        Args:
            query:              Search query text.  Must be non-empty.
            top_k:              Number of results (overrides instance default).
            filter:             ChromaDB ``where`` dict for metadata filtering,
                                e.g. ``{"price": {"$lt": 100}}``.
            include_embeddings: When ``True``, each :class:`SearchHit` in the
                                response will have its ``embeddings`` field
                                populated with the raw vector returned by the
                                vector store.

        Returns:
            :class:`SearchResponse` with ranked hits.

        Raises:
            ValueError: if *query* is empty or whitespace-only, or if
                        *top_k* is out of range.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        k = top_k if top_k is not None else self.default_top_k
        if k < 1 or k > _TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {_TOP_K_MAX}")

        t0 = time.monotonic()
        try:
            embedding = self.embedding_provider.embed_query(query)
        except Exception as exc:
            self.logger.error(
                "search.query.embedding_failed",
                error=str(exc),
                query_head=query[:100],
            )
            raise
        embedding_model = self.embedding_provider.model_name
        token_count = _estimate_tokens([query])
        cost = calculate_cost(
            embedding_model,
            token_count,
            0,  # Embeddings have no output token cost
        )

        raw_hits = self._store.query(
            embedding=embedding,
            top_k=k,
            filter=filter,
            include_embeddings=include_embeddings,
        )
        total = self._store.collection_stats(self.collection_name).get("chunks", 0)

        hits = [
            SearchHit(
                id=h["id"],
                text=h["text"],
                score=max(0.0, min(1.0, 1.0 - h["distance"])),
                metadata=h.get("metadata", {}),
                embeddings=h.get("embeddings") if include_embeddings else None,
            )
            for h in raw_hits
        ]

        return SearchResponse(
            hits=hits,
            query=query,
            total_searched=total,
            cost=cost,
            input_tokens=token_count,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aquery(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict | None = None,
        include_embeddings: bool = False,
    ) -> SearchResponse:
        """Async version of :meth:`query`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        k = top_k if top_k is not None else self.default_top_k
        if k < 1 or k > _TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {_TOP_K_MAX}")

        t0 = time.monotonic()
        try:
            embedding = await self.embedding_provider.embed_query_async(query)
        except Exception as exc:
            self.logger.error(
                "search.aquery.embedding_failed",
                error=str(exc),
                query_head=query[:100],
            )
            raise
        embedding_model = self.embedding_provider.model_name
        token_count = _estimate_tokens([query])
        cost = calculate_cost(
            embedding_model,
            token_count,
            0,  # Embeddings have no output token cost
        )

        raw_hits = self._store.query(
            embedding=embedding,
            top_k=k,
            filter=filter,
            include_embeddings=include_embeddings,
        )
        total = self._store.collection_stats(self.collection_name).get("chunks", 0)

        hits = [
            SearchHit(
                id=h["id"],
                text=h["text"],
                score=max(0.0, min(1.0, 1.0 - h["distance"])),
                metadata=h.get("metadata", {}),
                embeddings=h.get("embeddings") if include_embeddings else None,
            )
            for h in raw_hits
        ]

        return SearchResponse(
            hits=hits,
            query=query,
            total_searched=total,
            cost=cost,
            input_tokens=token_count,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Pipeline interface (BaseModule)
    # ------------------------------------------------------------------

    def run(self, text: str, **kwargs: Any) -> SearchResponse:
        """Pipeline entry point — delegates to :meth:`query`."""
        return self.query(text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> SearchResponse:
        """Async pipeline entry point — delegates to :meth:`aquery`."""
        return await self.aquery(text, **kwargs)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return names of all collections in the vector store."""
        return self._store.list_collections()

    def collection_stats(self, collection: str | None = None) -> dict:
        """Return stats for *collection* (defaults to this instance's collection)."""
        return self._store.collection_stats(
            collection if collection is not None else self.collection_name
        )

    def delete_collection(self, collection: str | None = None) -> None:
        """Delete *collection* (defaults to this instance's collection)."""
        self._store.delete_collection(
            collection if collection is not None else self.collection_name
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_store(self) -> VectorStoreAdapter:
        from aiforge.core.vectorstore import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
