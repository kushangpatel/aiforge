"""Pinecone vector store adapter.

Each AIForge "collection" maps to a Pinecone **namespace** within the
configured index.  Text is stored in ``metadata["_text"]`` because
Pinecone is a pure-vector store with no native text column.
"""

from __future__ import annotations

import structlog
from typing import Any

from aiforge.core.vectorstore import VectorStoreAdapter

logger = structlog.get_logger(__name__)

_UPSERT_BATCH = 100  # Pinecone upsert limit per call


class PineconeVectorStore(VectorStoreAdapter):
    """Pinecone-backed vector store.

    Args:
        api_key:    Pinecone API key.
        index_name: Name of the Pinecone index.
        namespace:  Namespace within the index (maps to AIForge "collection").
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "default",
    ) -> None:
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self._namespace = namespace

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        vectors = []
        for doc_id, emb, text, meta in zip(ids, embeddings, texts, metadatas):
            if "_text" in meta:
                logger.warning(
                    "vectorstore.pinecone.reserved_key",
                    key="_text",
                    id=doc_id,
                    msg="Metadata key '_text' is reserved and will be overwritten",
                )
            meta_with_text = {**meta, "_text": text}
            vectors.append({"id": doc_id, "values": emb, "metadata": meta_with_text})

        for i in range(0, len(vectors), _UPSERT_BATCH):
            self._index.upsert(
                vectors=vectors[i : i + _UPSERT_BATCH],
                namespace=self._namespace,
            )

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        from pinecone import QueryResponse

        raw = self._index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self._namespace,
            filter=filter,
            include_metadata=True,
            include_values=include_embeddings,
        )
        results: QueryResponse = raw  # type: ignore[assignment]

        hits: list[dict[str, Any]] = []
        for match in results.matches:
            meta = dict(match.metadata or {})
            text = meta.pop("_text", "")
            hit: dict[str, Any] = {
                "id": match.id,
                "text": text,
                "metadata": meta,
                "distance": 1.0 - match.score,  # similarity → distance
            }
            if include_embeddings and match.values:
                hit["embeddings"] = match["values"]
            hits.append(hit)
        return hits

    def delete_collection(self, name: str) -> None:
        self._index.delete(delete_all=True, namespace=name)

    def list_collections(self) -> list[str]:
        stats = self._index.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())

    def collection_stats(self, name: str) -> dict[str, Any]:
        stats = self._index.describe_index_stats()
        ns = stats.get("namespaces", {}).get(name, {})
        return {
            "name": name,
            "chunks": ns.get("vector_count", 0),
        }
