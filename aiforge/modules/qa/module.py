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

import time
import uuid
from typing import Any

from aiforge.core.pricing import calculate_cost
from aiforge.core.types import AIForgeError, BaseModule
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.modules.qa.schemas import IngestResponse, QAResponse


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, sum(len(t) for t in texts) // 4)


class DocQA(BaseModule):
    """Document Q&A module with RAG pipeline and agentic refinement.

    Args:
        collection:    ChromaDB collection name.
        persist_dir:   Directory for ChromaDB's SQLite backing store.
        chunk_size:    Override default chunk size for the text splitter.
        chunk_overlap: Override default chunk overlap for the text splitter.
        retrieval_k:   Number of chunks to retrieve per agent iteration.
        max_iterations: Maximum agentic refinement loops before responding.
        store:         Optional pre-built :class:`VectorStoreAdapter` (for DI/testing).
        **kwargs:      Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        collection: str = "default",
        persist_dir: str = "./aiforge_data",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        retrieval_k: int = 6,
        max_iterations: int = 3,
        store: VectorStoreAdapter | None = None,
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

        self.collection: str = stripped
        self.persist_dir: str = persist_dir
        self.chunk_size: int | None = chunk_size
        self.chunk_overlap: int | None = chunk_overlap
        self.retrieval_k: int = retrieval_k
        self.max_iterations: int = max_iterations
        self._store: VectorStoreAdapter = store or self._build_store()

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
        in batches, and stored in ChromaDB.

        Args:
            source:          File path, directory path, or list of file paths.
            file_extensions: Extension whitelist (e.g. ``[".py", ".md"]``).
            batch_size:      Embedding batch size.

        Returns:
            :class:`IngestResponse` with ingestion statistics.
        """
        from aiforge.core.doc_loader import DocLoader
        from aiforge.core.text_splitter import TextSplitter

        t0 = time.monotonic()

        loader = DocLoader(file_extensions=file_extensions)
        documents, stats = loader.load(source)

        if not documents:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split(documents)

        if not chunks:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

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

            metadatas = [
                {
                    "source": c.metadata.get("source", ""),
                    "file_type": c.metadata.get("file_type", ""),
                    "chunk_index": i + j,
                }
                for j, c in enumerate(batch)
            ]
            ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        c.metadata.get("source", "") + c.page_content[:64],
                    )
                )
                for c in batch
            ]

            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        cost = calculate_cost(embedding_model, total_tokens, 0)

        return IngestResponse(
            chunks_indexed=len(chunks),
            files_processed=stats.files_processed,
            languages=stats.languages,
            skipped_files=stats.skipped_files,
            collection=self.collection,
            cost=cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aingest(
        self,
        source: str | list[str],
        *,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
    ) -> IngestResponse:
        """Async version of :meth:`ingest`."""
        from aiforge.core.doc_loader import DocLoader
        from aiforge.core.text_splitter import TextSplitter

        t0 = time.monotonic()

        loader = DocLoader(file_extensions=file_extensions)
        documents, stats = loader.load(source)

        if not documents:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split(documents)

        if not chunks:
            return IngestResponse(
                chunks_indexed=0,
                files_processed=stats.files_processed,
                languages=stats.languages,
                skipped_files=stats.skipped_files,
                collection=self.collection,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

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

            metadatas = [
                {
                    "source": c.metadata.get("source", ""),
                    "file_type": c.metadata.get("file_type", ""),
                    "chunk_index": i + j,
                }
                for j, c in enumerate(batch)
            ]
            ids = [
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        c.metadata.get("source", "") + c.page_content[:64],
                    )
                )
                for c in batch
            ]

            self._store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        cost = calculate_cost(embedding_model, total_tokens, 0)

        return IngestResponse(
            chunks_indexed=len(chunks),
            files_processed=stats.files_processed,
            languages=stats.languages,
            skipped_files=stats.skipped_files,
            collection=self.collection,
            cost=cost,
            input_tokens=total_tokens,
            model=embedding_model,
            latency_ms=(time.monotonic() - t0) * 1000,
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

        t0 = time.monotonic()

        coll_stats = self._store.collection_stats(self.collection)
        if coll_stats.get("chunks", 0) == 0:
            raise AIForgeError(
                f"No documents ingested in collection '{self.collection}'"
            )

        graph = build_qa_graph(
            llm_fn=self._make_llm_fn(),
            retrieve_fn=self._make_retrieve_fn(),
            retrieval_k=self.retrieval_k,
        )

        final = graph.invoke(self._initial_state(query, filter_repo))
        return self._build_response(final, include_trace, t0)

    async def aask(
        self,
        query: str,
        *,
        filter_repo: str | None = None,
        include_trace: bool = False,
    ) -> QAResponse:
        """Async version of :meth:`ask`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        from aiforge.modules.qa.agent import build_qa_graph

        t0 = time.monotonic()

        coll_stats = self._store.collection_stats(self.collection)
        if coll_stats.get("chunks", 0) == 0:
            raise AIForgeError(
                f"No documents ingested in collection '{self.collection}'"
            )

        graph = build_qa_graph(
            llm_fn=self._make_llm_fn(),
            retrieve_fn=self._make_retrieve_fn(),
            retrieval_k=self.retrieval_k,
        )

        final = await graph.ainvoke(self._initial_state(query, filter_repo))
        return self._build_response(final, include_trace, t0)

    # ------------------------------------------------------------------
    # Pipeline interface (BaseModule)
    # ------------------------------------------------------------------

    def run(self, text: str, **kwargs: Any) -> QAResponse:
        """Pipeline entry point — delegates to :meth:`ask`."""
        return self.ask(text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> QAResponse:
        """Async pipeline entry point — delegates to :meth:`aask`."""
        return await self.aask(text, **kwargs)

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
        self._store.delete_collection(
            collection if collection is not None else self.collection
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_store(self) -> VectorStoreAdapter:
        from aiforge.core.vectorstore import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection,
        )

    def _make_llm_fn(self):
        """Return a ``(prompt: str) → CompletionResult`` closure."""

        def _llm(prompt: str):
            return self._call_llm([{"role": "user", "content": prompt}])

        return _llm

    def _make_retrieve_fn(self):
        """Return a ``(query, top_k, filter) → list[dict]`` closure."""

        def _retrieve(
            query_text: str,
            top_k: int = 6,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            embedding = self.embedding_provider.embed_query(query_text)
            return self._store.query(
                embedding=embedding, top_k=top_k, filter=filter,
            )

        return _retrieve

    def _initial_state(
        self, query: str, filter_repo: str | None,
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
            confidence=final["confidence"],
            iterations=final["iteration"],
            trace=final["trace"] if include_trace else None,
            input_tokens=final["total_input_tokens"],
            output_tokens=final["total_output_tokens"],
            cost=cost,
            model=model,
            trace_id=uuid.uuid4().hex[:16],
            latency_ms=(time.monotonic() - t0) * 1000,
        )
