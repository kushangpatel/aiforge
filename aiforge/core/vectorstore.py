"""
VectorStoreAdapter ABC and ChromaDB default implementation.

The ABC provides a provider-agnostic interface; the ChromaDB implementation
uses a persistent SQLite-backed store (zero extra infra required).

Concurrent writes to the same collection are serialised via a per-collection
threading lock.  Concurrent reads are always safe.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any


class VectorStoreAdapter(ABC):
    """Abstract interface for a vector database collection."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Upsert documents into the collection."""

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar documents.

        Each returned dict has keys: ``id``, ``text``, ``metadata``,
        ``distance`` (lower is more similar), and optionally
        ``embeddings`` (list of floats, only when
        ``include_embeddings=True``).
        """

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return names of all existing collections."""

    @abstractmethod
    def collection_stats(self, name: str) -> dict[str, Any]:
        """Return metadata about a collection (count, size, etc.)."""


# ---------------------------------------------------------------------------
# ChromaDB implementation
# ---------------------------------------------------------------------------

# Per-collection write locks (shared across all ChromaVectorStore instances
# that reference the same collection, keyed by "persist_dir::collection_name")
_write_locks: dict[str, threading.Lock] = {}
_locks_meta: threading.Lock = threading.Lock()


def _get_write_lock(key: str) -> threading.Lock:
    with _locks_meta:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]


class ChromaVectorStore(VectorStoreAdapter):
    """Default vector store implementation using ChromaDB.

    Uses a PersistentClient so data survives process restarts.
    Backed by SQLite — no extra infrastructure needed.
    """

    def __init__(
        self,
        persist_dir: str = "./aiforge_data",
        collection_name: str = "default",
    ) -> None:
        import chromadb

        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._write_lock = _get_write_lock(f"{persist_dir}::{collection_name}")

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        with self._write_lock:
            self._collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": include,
        }
        if filter:
            kwargs["where"] = filter

        results = self._collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        if not results["documents"]:
            return output

        raw_embeds = (
            results["embeddings"][0]
            if include_embeddings and results.get("embeddings")
            else None
        )

        for i, (doc, meta, dist, doc_id) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            )
        ):
            hit: dict[str, Any] = {
                "id": doc_id,
                "text": doc,
                "metadata": meta or {},
                "distance": dist,
            }
            if raw_embeds is not None:
                hit["embeddings"] = raw_embeds[i]
            output.append(hit)
        return output

    def delete_collection(self, name: str) -> None:
        with self._write_lock:
            self._client.delete_collection(name)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def collection_stats(self, name: str) -> dict[str, Any]:
        col = self._client.get_collection(name)
        count = col.count()
        return {
            "name": name,
            "chunks": count,
            "persist_dir": self._persist_dir,
        }
