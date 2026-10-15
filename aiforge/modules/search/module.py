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

import asyncio
import time
import uuid
from collections.abc import Sequence
from typing import Any

from aiforge.core.pricing import calculate_cost
from aiforge.core.rerank import LLMReranker, Reranker, apply_reranker
from aiforge.core.retrieval import (
    RetrievalMode,
    Retriever,
    retrieval_settings,
    validate_retrieval_mode,
)
from aiforge.core.types import BaseModule, CompletionResult
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


def _coerce_document(raw: dict[str, Any] | SearchDocument) -> SearchDocument:
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
    """Semantic search module backed by a configurable vector store.

    The backend is selected via ``AIFORGE_VECTOR_STORE`` (default: ``chroma``).
    Any registered :class:`VectorStoreAdapter` can be used — see
    :func:`~aiforge.registry.register_vectorstore`.

    Embed documents once with :meth:`index`, then retrieve similar ones with
    :meth:`query`.  All embedding costs are tracked and returned in every
    response.

    Args:
        collection:   Vector store collection name.  Leading/trailing
                      whitespace is stripped.
        persist_dir:  Data directory (used by ChromaDB; ignored by other backends).
        top_k:        Default number of results returned per query.  Can be
                      overridden per-call.
        store:        Optional pre-built :class:`VectorStoreAdapter`.
                      Primarily useful for dependency injection in tests —
                      pass a mock here instead of hitting a real DB.
        retrieval_mode: ``"vector"``, ``"hybrid"`` or ``"keyword"``
                      (default: ``AIFORGE_RETRIEVAL_MODE``, ``"vector"``).
        rerank:       Rerank candidates before returning them (default:
                      ``AIFORGE_RERANK``; ``True`` when *reranker* is given).
        rerank_top_n: Hits kept after reranking when a query passes no
                      ``top_k`` (default: *top_k*).
        fetch_k:      Candidates fetched before fusion / reranking
                      (default: ``AIFORGE_FETCH_K``, 20).
        reranker:     Custom :class:`~aiforge.core.rerank.Reranker`; defaults
                      to an :class:`~aiforge.core.rerank.LLMReranker`.
        **kwargs:     Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    # Class-level defaults keep instances built without __init__ (tests) working.
    retrieval_mode: str = "vector"
    rerank: bool = False
    rerank_top_n: int | None = None
    fetch_k: int = 20
    rrf_k: int = 60
    reranker: Reranker | None = None
    _retriever: Retriever | None = None

    def __init__(
        self,
        collection: str = "default",
        persist_dir: str = "./aiforge_data",
        top_k: int = 5,
        store: VectorStoreAdapter | None = None,
        *,
        retrieval_mode: str | None = None,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
        fetch_k: int | None = None,
        reranker: Reranker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        stripped = (collection or "").strip()
        if not stripped:
            raise ValueError("collection must not be empty or whitespace-only")
        if top_k < 1 or top_k > _TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {_TOP_K_MAX}")

        cfg = retrieval_settings(self)
        self.collection_name: str = stripped
        self.persist_dir: str = persist_dir
        self.default_top_k: int = top_k
        self.retrieval_mode = validate_retrieval_mode(retrieval_mode or cfg.RETRIEVAL_MODE)
        self.reranker = reranker
        self.rerank = rerank if rerank is not None else (reranker is not None or cfg.RERANK)
        self.rerank_top_n = rerank_top_n if rerank_top_n is not None else cfg.RERANK_TOP_N
        self.fetch_k = fetch_k if fetch_k is not None else cfg.FETCH_K
        self.rrf_k = cfg.RRF_K
        if self.rerank_top_n is not None and not 1 <= self.rerank_top_n <= _TOP_K_MAX:
            raise ValueError(f"rerank_top_n must be between 1 and {_TOP_K_MAX}")
        if self.fetch_k < 1:
            raise ValueError("fetch_k must be >= 1")
        self._store: VectorStoreAdapter = store or self._build_store()

    # ------------------------------------------------------------------
    # Public API — indexing
    # ------------------------------------------------------------------

    def index(
        self,
        documents: Sequence[dict[str, Any] | SearchDocument],
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
        t0 = time.perf_counter()
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

        self._invalidate_keyword_index()
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
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def aindex(
        self,
        documents: Sequence[dict[str, Any] | SearchDocument],
        *,
        batch_size: int = 100,
    ) -> IndexResponse:
        """Async version of :meth:`index`."""
        if not documents:
            raise ValueError("documents must not be empty")

        docs = [_coerce_document(d) for d in documents]
        t0 = time.perf_counter()
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

        self._invalidate_keyword_index()
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
            latency_ms=(time.perf_counter() - t0) * 1000,
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
            self.logger.warning("search.index_files.no_documents_loaded", source=str(source))
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
        filter: dict[str, Any] | None = None,
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
        k = self._validate_query(query, top_k)
        t0 = time.perf_counter()
        mode = self._get_retriever().effective_mode(filter)

        embedding: list[float] | None = None
        embedding_model = ""
        token_count = 0
        if mode != "keyword":
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

        raw_hits, rerank_result = self._retrieve_hits(
            query, k, mode, embedding, filter, include_embeddings
        )
        total = self._store.collection_stats(self.collection_name).get("chunks", 0)
        return self._build_response(
            query,
            raw_hits,
            total,
            include_embeddings,
            embedding_model,
            token_count,
            rerank_result,
            t0,
        )

    async def aquery(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> SearchResponse:
        """Async version of :meth:`query`."""
        k = self._validate_query(query, top_k)
        t0 = time.perf_counter()
        # Run sync store operations in a thread pool to avoid blocking the loop.
        loop = asyncio.get_running_loop()
        retriever = self._get_retriever()
        mode = await loop.run_in_executor(None, retriever.effective_mode, filter)

        embedding: list[float] | None = None
        embedding_model = ""
        token_count = 0
        if mode != "keyword":
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

        raw_hits, rerank_result = await loop.run_in_executor(
            None,
            lambda: self._retrieve_hits(query, k, mode, embedding, filter, include_embeddings),
        )
        total = await loop.run_in_executor(
            None,
            lambda: self._store.collection_stats(self.collection_name).get("chunks", 0),
        )
        return self._build_response(
            query,
            raw_hits,
            total,
            include_embeddings,
            embedding_model,
            token_count,
            rerank_result,
            t0,
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

    def collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Return stats for *collection* (defaults to this instance's collection)."""
        return self._store.collection_stats(
            collection if collection is not None else self.collection_name
        )

    def delete_collection(self, collection: str | None = None) -> None:
        """Delete *collection* (defaults to this instance's collection)."""
        self._store.delete_collection(
            collection if collection is not None else self.collection_name
        )
        self._invalidate_keyword_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_query(self, query: str, top_k: int | None) -> int:
        """Validate *query* / *top_k* and return the number of hits to return."""
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if top_k is not None:
            k = top_k
        elif self.rerank and self.rerank_top_n is not None:
            k = self.rerank_top_n
        else:
            k = self.default_top_k
        if k < 1 or k > _TOP_K_MAX:
            raise ValueError(f"top_k must be between 1 and {_TOP_K_MAX}")
        return k

    def _retrieve_hits(
        self,
        query: str,
        k: int,
        mode: RetrievalMode,
        embedding: list[float] | None,
        filter: dict[str, Any] | None,
        include_embeddings: bool,
    ) -> tuple[list[dict[str, Any]], CompletionResult | None]:
        """Run retrieval (and reranking) synchronously; return raw hits + reranker call."""
        reranker = self._active_reranker()
        if mode == "vector" and reranker is None:
            # Unchanged default path: a single similarity query.
            assert embedding is not None
            hits = self._store.query(
                embedding=embedding,
                top_k=k,
                filter=filter,
                include_embeddings=include_embeddings,
            )
            return hits, None

        candidate_k = max(k, self.fetch_k)
        hits = self._get_retriever().retrieve(
            query,
            top_k=candidate_k if reranker is not None else k,
            embedding=embedding,
            filter=filter,
            include_embeddings=include_embeddings,
            candidate_k=candidate_k,
            mode=mode,
        )
        if reranker is None:
            return hits, None
        return apply_reranker(reranker, query, hits, k)

    @staticmethod
    def _build_response(
        query: str,
        raw_hits: list[dict[str, Any]],
        total: int,
        include_embeddings: bool,
        embedding_model: str,
        token_count: int,
        rerank_result: CompletionResult | None,
        t0: float,
    ) -> SearchResponse:
        # NOTE: in vector mode ``1.0 - distance`` assumes cosine distance (range
        # 0-2, typically 0-1); hybrid / keyword hits carry a ready-made ``score``.
        hits = [
            SearchHit(
                id=h["id"],
                text=h["text"],
                score=(h["score"] if "score" in h else max(0.0, min(1.0, 1.0 - h["distance"]))),
                metadata=h.get("metadata", {}),
                embeddings=h.get("embeddings") if include_embeddings else None,
                rerank_score=h.get("rerank_score"),
            )
            for h in raw_hits
        ]
        cost = calculate_cost(embedding_model, token_count, 0) if embedding_model else 0.0
        output_tokens = 0
        if rerank_result is not None:
            cost += rerank_result.cost
            token_count += rerank_result.input_tokens
            output_tokens = rerank_result.output_tokens

        return SearchResponse(
            hits=hits,
            query=query,
            total_searched=total,
            cost=cost,
            input_tokens=token_count,
            output_tokens=output_tokens,
            model=embedding_model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _active_reranker(self) -> Reranker | None:
        if not self.rerank:
            return None
        if self.reranker is None:
            self.reranker = LLMReranker(self)
        return self.reranker

    def _get_retriever(self) -> Retriever:
        if self._retriever is None or self._retriever.store is not self._store:
            self._retriever = Retriever(
                self._store,
                self.collection_name,
                mode=self.retrieval_mode,
                rrf_k=self.rrf_k,
            )
        return self._retriever

    def _invalidate_keyword_index(self) -> None:
        if self._retriever is not None:
            self._retriever.invalidate()

    def _build_store(self) -> VectorStoreAdapter:
        from aiforge.core.vectorstore_factory import build_vectorstore

        return build_vectorstore(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir,
            settings=self.settings,
        )
