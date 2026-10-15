"""
DocQA module — document question answering with RAG and agentic refinement.

Full pipeline: ingest documents → chunk → embed → store in ChromaDB →
ask questions with a LangGraph agent that retrieves context, self-assesses
sufficiency, and optionally refines the search before generating an answer.

Usage::

    from aiforge import DocQA

    qa = DocQA(collection="my-project")

    # Ingest a directory
    stats = qa.ingest("/path/to/codebase")
    print(f"Indexed {stats.chunks_indexed} chunks from {stats.files_processed} files")

    # Ask a question
    answer = qa.ask("How does the auth middleware work?", include_trace=True)
    print(answer.text)
    print(f"Sources: {answer.sources}")
    print(f"Confidence: {answer.confidence:.0%}")
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import uuid
from collections.abc import Callable
from typing import Any

from aiforge.core.pricing import calculate_cost
from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.rerank import LLMReranker, Reranker, apply_reranker
from aiforge.core.retrieval import Retriever, retrieval_settings, validate_retrieval_mode
from aiforge.core.text_splitter import validate_chunking
from aiforge.core.types import AIForgeError, BaseModule, CompletionResult
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.modules.qa.schemas import Citation, IngestResponse, QAResponse


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, sum(len(t) for t in texts) // 4)


class DocQA(BaseModule):
    """Document Q&A module with RAG pipeline and agentic refinement.

    Args:
        collection:    Vector store collection name.
        persist_dir:   Data directory (used by ChromaDB; ignored by other backends).
        chunk_size:    Override default chunk size for the text splitter.
        chunk_overlap: Override default chunk overlap for the text splitter.
        retrieval_k:   Number of chunks to retrieve per agent iteration.
        max_iterations: Maximum agentic refinement loops before responding.
        store:         Optional pre-built :class:`VectorStoreAdapter` (for DI/testing).
        chunking:      ``"recursive"``, ``"markdown"`` or ``"token"``
                       (default: ``AIFORGE_CHUNKING``, ``"recursive"``).
        retrieval_mode: ``"vector"``, ``"hybrid"`` or ``"keyword"``
                       (default: ``AIFORGE_RETRIEVAL_MODE``, ``"vector"``).
        rerank:        Rerank retrieved candidates (default: ``AIFORGE_RERANK``;
                       ``True`` when *reranker* is given).
        rerank_top_n:  Chunks kept after reranking (default: *retrieval_k*).
        fetch_k:       Candidates fetched before fusion / reranking
                       (default: ``AIFORGE_FETCH_K``, 20).
        reranker:      Custom :class:`~aiforge.core.rerank.Reranker`; defaults
                       to an :class:`~aiforge.core.rerank.LLMReranker` on this
                       module's provider.
        **kwargs:      Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    # Class-level defaults keep instances built without __init__ (tests) working.
    chunking: str = "recursive"
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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        retrieval_k: int = 6,
        max_iterations: int = 3,
        store: VectorStoreAdapter | None = None,
        system_prompt: str | None = None,
        *,
        chunking: str | None = None,
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
        if retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        cfg = retrieval_settings(self)
        self.chunking = validate_chunking(chunking or cfg.CHUNKING)
        self.retrieval_mode = validate_retrieval_mode(retrieval_mode or cfg.RETRIEVAL_MODE)
        self.reranker = reranker
        self.rerank = rerank if rerank is not None else (reranker is not None or cfg.RERANK)
        self.rerank_top_n = rerank_top_n if rerank_top_n is not None else cfg.RERANK_TOP_N
        self.fetch_k = fetch_k if fetch_k is not None else cfg.FETCH_K
        self.rrf_k = cfg.RRF_K
        if self.rerank_top_n is not None and self.rerank_top_n < 1:
            raise ValueError("rerank_top_n must be >= 1")
        if self.fetch_k < 1:
            raise ValueError("fetch_k must be >= 1")

        self.collection: str = stripped
        self.persist_dir: str = persist_dir
        self.chunk_size: int | None = chunk_size
        self.chunk_overlap: int | None = chunk_overlap
        self.retrieval_k: int = retrieval_k
        self.max_iterations: int = max_iterations
        self._store: VectorStoreAdapter = store or self._build_store()
        self._system_prompt_override: str | None = system_prompt

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | list[str],
        *,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
    ) -> IngestResponse:
        """Ingest documents from path, directory, or list of files.

        Documents are loaded, chunked with language-aware splitting, embedded
        in batches, and stored.

        Args:
            source:          File path, directory path, or list of file paths.
            file_extensions: Extension whitelist (e.g. ``[".py", ".md"]``).
            batch_size:      Embedding batch size.

        Returns:
            :class:`IngestResponse` with ingestion statistics.
        """
        from aiforge.core.doc_loader import DocLoader
        from aiforge.core.text_splitter import TextSplitter

        t0 = time.perf_counter()

        loader = DocLoader(file_extensions=file_extensions)
        documents, stats = loader.load(source)

        if not documents:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=self._elapsed_ms(t0),
            )

        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=self.chunking,
        )
        chunks = splitter.split(documents)

        if not chunks:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=self._elapsed_ms(t0),
            )

        chunks_indexed, cost, embedding_model = self._embed_and_store(chunks, batch_size)

        return IngestResponse(
            chunks_indexed=chunks_indexed,
            files_processed=stats.files_processed,
            languages=stats.languages,
            skipped_files=stats.skipped_files,
            collection=self.collection,
            cost=cost,
            input_tokens=_estimate_tokens([c.page_content for c in chunks]),
            model=embedding_model,
            latency_ms=self._elapsed_ms(t0),
        )

    async def aingest(
        self,
        source: str | list[str],
        *,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
    ) -> IngestResponse:
        """Async version of :meth:`ingest`."""
        import asyncio

        from aiforge.core.doc_loader import DocLoader
        from aiforge.core.text_splitter import TextSplitter

        t0 = time.perf_counter()

        loader = DocLoader(file_extensions=file_extensions)
        loop = asyncio.get_running_loop()
        documents, stats = await loop.run_in_executor(
            None,
            loader.load,
            source,
        )

        if not documents:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=self._elapsed_ms(t0),
            )

        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=self.chunking,
        )
        chunks = splitter.split(documents)

        if not chunks:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=self._elapsed_ms(t0),
            )

        chunks_indexed, cost, embedding_model = await self._aembed_and_store(chunks, batch_size)
        total_tokens = _estimate_tokens([c.page_content for c in chunks])

        return IngestResponse(
            chunks_indexed=chunks_indexed,
            files_processed=stats.files_processed,
            languages=stats.languages,
            skipped_files=stats.skipped_files,
            collection=self.collection,
            cost=cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=self._elapsed_ms(t0),
        )

    # ------------------------------------------------------------------
    # Incremental ingestion
    # ------------------------------------------------------------------

    def ingest_incremental(
        self,
        source: str | list[str],
        *,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
    ) -> IngestResponse:
        """Re-ingest only changed or new files; purge chunks for deleted files.

        Compares the current file mtime against the mtime stored in chunk
        metadata at last ingest time.  Only files whose mtime has changed
        (or that were not previously indexed) are re-embedded.

        Algorithm:

        1. Walk *source* directory and record current mtime for every file.
        2. Query the vector store for the set of already-indexed
           ``(source_path, mtime)`` pairs.
        3. **Unchanged** files — same mtime — are skipped entirely.
        4. **Modified** files — different mtime — old chunks are deleted,
           then the file is re-embedded.
        5. **New** files — not in index — are embedded fresh.
        6. **Deleted** files — in index but not on disk — chunks are purged.

        Args:
            source:          File path, directory path, list of files, or git URL.
            file_extensions: Extension whitelist (e.g. ``[".py", ".md"]``).
            batch_size:      Embedding batch size.

        Returns:
            :class:`IngestResponse` with full incremental statistics.
        """
        from aiforge.core.doc_loader import DocLoader
        from aiforge.core.text_splitter import TextSplitter

        t0 = time.perf_counter()

        # 1. Load source through DocLoader (same source semantics as ingest).
        loader = DocLoader(file_extensions=file_extensions)
        documents, stats = loader.load(source)
        disk_files = self._snapshot_loaded_sources(documents)

        # 2. Snapshot of files already in the index: path -> mtime (float | None)
        indexed = self._get_indexed_sources()
        scoped_indexed = self._scope_indexed_sources(source, indexed)
        if file_extensions:
            ext_set = {e.lower() for e in file_extensions}
            scoped_indexed = {
                k: v
                for k, v in scoped_indexed.items()
                if any(k.lower().endswith(e) for e in ext_set)
            }

        # 3. Categorise
        added, updated, unchanged, deleted = self._diff_sources(disk_files, scoped_indexed)

        # 4. Purge stale chunks (modified + deleted files)
        chunks_deleted = 0
        for path in updated | deleted:
            chunks_deleted += self._store.delete_by_source(path)
        if updated or deleted:
            self._invalidate_keyword_index()

        # 5. Re-embed modified + new files
        files_to_index = added | updated
        chunks_indexed = 0
        total_cost = 0.0
        total_tokens = 0
        embedding_model = ""
        skipped = stats.skipped_files
        docs_to_index = [
            d for d in documents if self._normpath(d.metadata.get("source", "")) in files_to_index
        ]
        languages = dict(stats.languages)

        if docs_to_index:
            splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=self.chunking,
            )
            chunks = splitter.split(docs_to_index)

            # Attach mtime to metadata so future incremental runs can detect changes.
            for chunk in chunks:
                src = chunk.metadata.get("source", "")
                norm = self._normpath(src)
                mtime = disk_files.get(norm)
                if mtime is not None:
                    chunk.metadata["mtime"] = mtime

            if chunks:
                total_tokens = _estimate_tokens([c.page_content for c in chunks])
                chunks_indexed, total_cost, embedding_model = self._embed_and_store(
                    chunks, batch_size
                )

        return IngestResponse(
            chunks_indexed=chunks_indexed,
            files_processed=len(files_to_index),
            languages=languages,
            skipped_files=skipped,
            collection=self.collection,
            cost=total_cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=self._elapsed_ms(t0),
            # Incremental stats
            files_added=len(added),
            files_updated=len(updated),
            files_unchanged=len(unchanged),
            files_deleted=len(deleted),
            chunks_deleted=chunks_deleted,
        )

    async def aingest_incremental(
        self,
        source: str | list[str],
        *,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
    ) -> IngestResponse:
        """Async version of :meth:`ingest_incremental`."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ingest_incremental(
                source,
                file_extensions=file_extensions,
                batch_size=batch_size,
            ),
        )

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        *,
        filter_repo: str | None = None,
        include_trace: bool = False,
        system_prompt: str | None = None,
    ) -> QAResponse:
        """Ask a question against ingested documents using a LangGraph agent.

        The agent classifies the query, retrieves relevant chunks, self-assesses
        sufficiency, and may refine the search before responding.

        Args:
            query:         The question to answer.
            filter_repo:   Optional source-path substring to narrow retrieval.
            include_trace: Include step-by-step agent trace in the response.

        Returns:
            :class:`QAResponse` with answer, sources, and confidence.

        Raises:
            ValueError: if *query* is empty or whitespace-only.
            AIForgeError: If the collection contains no documents.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        from aiforge.modules.qa.agent import build_qa_graph

        t0 = time.perf_counter()

        coll_stats = self._store.collection_stats(self.collection)
        if coll_stats.get("chunks", 0) == 0:
            raise AIForgeError(f"No documents ingested in collection '{self.collection}'")

        rerank_usage: list[CompletionResult] = []
        graph = build_qa_graph(
            llm_fn=self._make_llm_fn(system_prompt=system_prompt),
            retrieve_fn=self._make_retrieve_fn(rerank_usage),
            retrieval_k=self._final_k(),
        )

        final = graph.invoke(self._initial_state(query, filter_repo))
        self._add_usage(final, rerank_usage)
        return self._build_response(final, include_trace, t0)

    async def aask(
        self,
        query: str,
        *,
        filter_repo: str | None = None,
        include_trace: bool = False,
        system_prompt: str | None = None,
    ) -> QAResponse:
        """Async version of :meth:`ask`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        from aiforge.modules.qa.agent import build_qa_graph

        t0 = time.perf_counter()

        coll_stats = self._store.collection_stats(self.collection)
        if coll_stats.get("chunks", 0) == 0:
            raise AIForgeError(f"No documents ingested in collection '{self.collection}'")

        rerank_usage: list[CompletionResult] = []
        graph = build_qa_graph(
            llm_fn=self._make_llm_fn(system_prompt=system_prompt),
            retrieve_fn=self._make_retrieve_fn(rerank_usage),
            retrieval_k=self._final_k(),
        )

        final = await graph.ainvoke(self._initial_state(query, filter_repo))
        self._add_usage(final, rerank_usage)
        return self._build_response(final, include_trace, t0)

    # ------------------------------------------------------------------
    # Pipeline interface (BaseModule)
    # ------------------------------------------------------------------

    def run(self, text: str, **kwargs: Any) -> QAResponse:
        """Pipeline entry point — delegates to :meth:`ask`."""
        sp = kwargs.pop("system_prompt", None)
        return self.ask(text, system_prompt=sp)

    async def arun(self, text: str, **kwargs: Any) -> QAResponse:
        """Async pipeline entry point — delegates to :meth:`aask`."""
        sp = kwargs.pop("system_prompt", None)
        return await self.aask(text, system_prompt=sp)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return names of all collections in the vector store."""
        return self._store.list_collections()

    def collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Return stats for *collection* (defaults to this instance's collection)."""
        return self._store.collection_stats(
            collection if collection is not None else self.collection
        )

    def delete_collection(self, collection: str | None = None) -> None:
        """Delete *collection* (defaults to this instance's collection)."""
        self._store.delete_collection(collection if collection is not None else self.collection)
        self._invalidate_keyword_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_store(self) -> VectorStoreAdapter:
        from aiforge.core.vectorstore_factory import build_vectorstore

        return build_vectorstore(
            collection_name=self.collection,
            persist_dir=self.persist_dir,
            settings=self.settings,
        )

    def _make_llm_fn(
        self, *, system_prompt: str | None = None
    ) -> Callable[[str], CompletionResult]:
        """Return a ``(prompt: str) → CompletionResult`` closure."""
        resolved = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("qa", {}).get("system"),
            default="",
        )

        def _llm(prompt: str) -> CompletionResult:
            messages: list[dict[str, str]] = []
            if resolved:
                messages.append({"role": "system", "content": resolved})
            messages.append({"role": "user", "content": prompt})
            return self._call_llm(messages, temperature=0.0)

        return _llm

    def _make_retrieve_fn(
        self, rerank_usage: list[CompletionResult] | None = None
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a ``(query, top_k, filter) → list[dict]`` closure.

        LLM reranker calls made by the closure are appended to *rerank_usage*
        so their tokens can be added to the response totals.
        """

        def _retrieve(
            query_text: str,
            top_k: int = 6,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            return self.retrieve(query_text, top_k=top_k, filter=filter, _usage=rerank_usage)

        return _retrieve

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        _usage: list[CompletionResult] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the chunks most relevant to *query*, without calling the agent.

        Runs the configured retrieval mode (vector / hybrid / keyword) and,
        when enabled, the reranker.  Each hit is a dict with ``id``, ``text``,
        ``metadata`` and ``score`` (plus ``rerank_score`` when reranked).

        Args:
            query:  Query text.
            top_k:  Hits to return (default: ``rerank_top_n`` when reranking,
                    else ``retrieval_k``).
            filter: Metadata filter (Chroma ``where`` syntax).
        """
        k = top_k if top_k is not None else self._final_k()
        retriever = self._get_retriever()
        mode = retriever.effective_mode(filter)
        reranker = self._active_reranker()

        if mode == "vector" and reranker is None:
            # Unchanged default path: a single similarity query.
            hits = self._store.query(
                embedding=self.embedding_provider.embed_query(query), top_k=k, filter=filter
            )
            for h in hits:
                h.setdefault("score", max(0.0, min(1.0, 1.0 - float(h.get("distance", 1.0)))))
            return hits

        candidate_k = max(k, self.fetch_k)
        hits = retriever.retrieve(
            query,
            top_k=candidate_k if reranker is not None else k,
            embedding=self.embedding_provider.embed_query(query) if mode != "keyword" else None,
            filter=filter,
            candidate_k=candidate_k,
            mode=mode,
        )
        if reranker is None:
            return hits
        reranked, result = apply_reranker(reranker, query, hits, k)
        if result is not None and _usage is not None:
            _usage.append(result)
        return reranked

    def _final_k(self) -> int:
        """Chunks used per retrieval: ``rerank_top_n`` when reranking, else ``retrieval_k``."""
        if self.rerank and self.rerank_top_n is not None:
            return self.rerank_top_n
        return self.retrieval_k

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
                self.collection,
                mode=self.retrieval_mode,
                rrf_k=self.rrf_k,
            )
        return self._retriever

    def _invalidate_keyword_index(self) -> None:
        if self._retriever is not None:
            self._retriever.invalidate()

    @staticmethod
    def _add_usage(final: dict[str, Any], usage: list[CompletionResult]) -> None:
        """Fold reranker LLM usage into the agent's token totals."""
        for r in usage:
            final["total_input_tokens"] += r.input_tokens
            final["total_output_tokens"] += r.output_tokens

    def _initial_state(
        self,
        query: str,
        filter_repo: str | None,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "repo_filter": filter_repo,
            "intent": "",
            "retrieved_chunks": [],
            "current_query": query,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "trace": [],
            "answer": "",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
            "decision": "",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "model": "",
        }

    @staticmethod
    def _build_response(
        final: dict[str, Any],
        include_trace: bool,
        t0: float,
    ) -> QAResponse:
        model = final["model"] or ""
        cost = calculate_cost(
            model,
            final["total_input_tokens"],
            final["total_output_tokens"],
        )
        return QAResponse(
            text=final["answer"],
            sources=final["sources"],
            citations=[Citation(**c) for c in final.get("citations") or []],
            confidence=final["confidence"],
            iterations=final["iteration"],
            trace=final["trace"] if include_trace else None,
            input_tokens=final["total_input_tokens"],
            output_tokens=final["total_output_tokens"],
            cost=cost,
            model=model,
            trace_id=uuid.uuid4().hex[:16],
            latency_ms=DocQA._elapsed_ms(t0),
        )

    # ------------------------------------------------------------------
    # Shared embed-and-store helpers (sync + async)
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
        """Build a deterministic, collision-safe chunk ID.

        Uses source path + chunk index + SHA256(content) so IDs are stable
        across re-ingest, while avoiding collisions when different chunks
        share the same leading prefix.
        """
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        name = f"{source}\n{chunk_index}\n{digest}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

    def _embed_and_store(
        self,
        chunks: list[Any],
        batch_size: int,
    ) -> tuple[int, float, str]:
        """Embed *chunks* and upsert them into the vector store.

        Returns:
            ``(chunks_indexed, total_cost, embedding_model)``
        """
        total_tokens = 0
        embedding_model = ""

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            try:
                embeddings = self._call_embedding(texts)
            except Exception as exc:
                self.logger.error(
                    "qa.ingest.embedding_failed",
                    error=str(exc),
                    batch_index=i // batch_size,
                )
                raise
            total_tokens += _estimate_tokens(texts)
            embedding_model = self.embedding_provider.model_name

            metadatas = [self._chunk_metadata(c, i + j) for j, c in enumerate(batch)]
            ids = [
                self._stable_chunk_id(
                    c.metadata.get("source", ""),
                    i + j,
                    c.page_content,
                )
                for j, c in enumerate(batch)
            ]

            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        self._invalidate_keyword_index()
        cost = calculate_cost(embedding_model, total_tokens, 0)
        return len(chunks), cost, embedding_model

    async def _aembed_and_store(
        self,
        chunks: list[Any],
        batch_size: int,
    ) -> tuple[int, float, str]:
        """Async version of :meth:`_embed_and_store`."""
        total_tokens = 0
        embedding_model = ""

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            try:
                embeddings = await self._acall_embedding(texts)
            except Exception as exc:
                self.logger.error(
                    "qa.aingest.embedding_failed",
                    error=str(exc),
                    batch_index=i // batch_size,
                )
                raise
            total_tokens += _estimate_tokens(texts)
            embedding_model = self.embedding_provider.model_name

            metadatas = [self._chunk_metadata(c, i + j) for j, c in enumerate(batch)]
            ids = [
                self._stable_chunk_id(
                    c.metadata.get("source", ""),
                    i + j,
                    c.page_content,
                )
                for j, c in enumerate(batch)
            ]

            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        self._invalidate_keyword_index()
        cost = calculate_cost(embedding_model, total_tokens, 0)
        return len(chunks), cost, embedding_model

    # Chunk metadata keys copied into the store when present on a chunk.
    _OPTIONAL_METADATA = ("mtime", "page", "heading_path")

    @classmethod
    def _chunk_metadata(cls, chunk: Any, chunk_index: int) -> dict[str, Any]:
        """Store metadata for one chunk: source, type, index, plus page / heading / mtime."""
        meta: dict[str, Any] = {
            "source": chunk.metadata.get("source", ""),
            "file_type": chunk.metadata.get("file_type", ""),
            "chunk_index": chunk_index,
        }
        for key in cls._OPTIONAL_METADATA:
            if chunk.metadata.get(key) is not None:
                meta[key] = chunk.metadata[key]
        return meta

    # ------------------------------------------------------------------
    # Incremental ingest helpers
    # ------------------------------------------------------------------

    def _scan_disk(
        self,
        source: str,
        file_extensions: list[str] | None,
    ) -> dict[str, float]:
        """Walk *source* directory and return ``{normpath: mtime}``.

        Skips the same directories that :class:`~aiforge.core.doc_loader.DocLoader`
        ignores (``__pycache__``, ``.git``, ``venv``, etc.) so spurious "new file"
        entries are not generated for files that DocLoader would skip anyway.
        Paths are normalised with :func:`_normpath` so they match whatever
        ``str(pathlib.Path(...))`` produces on the current OS.
        """
        import os

        # Mirror DocLoader._IGNORE_DIR_NAMES to avoid scanning dirs it skips.
        _IGNORE = frozenset(
            {
                ".git",
                ".svn",
                ".hg",
                "__pycache__",
                ".mypy_cache",
                ".ruff_cache",
                ".pytest_cache",
                "node_modules",
                ".next",
                ".nuxt",
                "venv",
                ".venv",
                "env",
                ".env",
                "dist",
                "build",
                ".build",
                "aiforge_data",
            }
        )

        ext_set = {e.lower() for e in file_extensions} if file_extensions else None
        result: dict[str, float] = {}

        for dirpath, dirnames, filenames in os.walk(source):
            # Prune ignored directories in-place so os.walk skips them.
            dirnames[:] = [d for d in dirnames if d not in _IGNORE]

            for fname in filenames:
                if ext_set is not None and not any(fname.lower().endswith(e) for e in ext_set):
                    continue
                full = self._normpath(os.path.join(dirpath, fname))
                # race condition — file may have disappeared; skip it
                with contextlib.suppress(OSError):
                    result[full] = os.path.getmtime(full)
        return result

    def _snapshot_loaded_sources(self, documents: list[Any]) -> dict[str, float | None]:
        """Build ``{normpath: mtime_or_none}`` for loaded documents."""
        import os

        snapshot: dict[str, float | None] = {}
        for doc in documents:
            src = doc.metadata.get("source", "")
            if not src:
                continue
            norm = self._normpath(src)
            if norm in snapshot:
                continue
            try:
                snapshot[norm] = os.path.getmtime(src) if os.path.exists(src) else None
            except OSError:
                snapshot[norm] = None
        return snapshot

    def _get_indexed_sources(self) -> dict[str, float | None]:
        """Return ``{source_path: mtime_or_None}`` for every chunk in the store.

        Uses :meth:`~aiforge.core.vectorstore.VectorStoreAdapter.list_all_metadata`
        so no query vector is required and all chunks are enumerated correctly.
        """
        try:
            all_meta = self._store.list_all_metadata()
        except Exception:
            return {}

        indexed: dict[str, float | None] = {}
        for meta in all_meta:
            src = meta.get("source", "")
            if not src:
                continue
            norm = self._normpath(src)
            if norm not in indexed:  # keep first occurrence (mtime may vary per chunk)
                raw_mtime = meta.get("mtime")
                indexed[norm] = float(raw_mtime) if raw_mtime is not None else None
        return indexed

    def _scope_indexed_sources(
        self,
        source: str | list[str],
        indexed: dict[str, float | None],
    ) -> dict[str, float | None]:
        """Return indexed sources relevant to the given ingest source scope."""
        import os
        from pathlib import Path

        if isinstance(source, list):
            source_set = {self._normpath(s) for s in source}
            return {k: v for k, v in indexed.items() if k in source_set}

        if source.endswith(".git"):
            prefix = self._normpath(source).rstrip("/\\") + "/"
            return {k: v for k, v in indexed.items() if k.startswith(prefix)}

        p = Path(source)
        if p.is_dir():
            root = self._normpath(str(p)).rstrip("/\\") + os.sep
            return {k: v for k, v in indexed.items() if k.startswith(root)}
        if p.is_file():
            file_key = self._normpath(str(p))
            if file_key in indexed:
                return {file_key: indexed[file_key]}
            return {}

        return {}

    @staticmethod
    def _normpath(path: str) -> str:
        """Normalise a path for cross-platform comparison.

        ``os.path.normcase`` lower-cases and converts slashes on Windows so
        that paths stored by ``str(pathlib.Path(...))`` match those produced
        by ``os.path.join`` in ``_scan_disk``.
        """
        import os

        if "://" in path or path.startswith("git@"):
            return path.strip().rstrip("/")
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _diff_sources(
        disk: dict[str, float | None],
        indexed: dict[str, float | None],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Categorise files into added / updated / unchanged / deleted.

        Returns:
            ``(added, updated, unchanged, deleted)`` — four disjoint sets of
            absolute file paths.
        """
        disk_set = set(disk)
        indexed_set = set(indexed)

        added = disk_set - indexed_set  # on disk, not in index
        deleted = indexed_set - disk_set  # in index, not on disk
        common = disk_set & indexed_set

        updated: set[str] = set()
        unchanged: set[str] = set()
        for path in common:
            disk_mtime = disk[path]
            stored_mtime = indexed[path]
            if disk_mtime is None or stored_mtime is None or abs(disk_mtime - stored_mtime) > 1e-3:
                updated.add(path)
            else:
                unchanged.add(path)

        return added, updated, unchanged, deleted

    @staticmethod
    def _elapsed_ms(t0: float) -> float:
        """Return non-zero elapsed ms for very fast code paths."""
        return max((time.perf_counter() - t0) * 1000, 0.001)
