"""Qdrant vector store adapter.

Each AIForge "collection" maps to a Qdrant collection.  The collection is
created automatically on first use if it does not already exist.
"""

from __future__ import annotations

import structlog
from typing import Any

from aiforge.core.vectorstore import VectorStoreAdapter

logger = structlog.get_logger(__name__)

_UPSERT_BATCH = 100


class QdrantVectorStore(VectorStoreAdapter):
    """Qdrant-backed vector store.

    Args:
        url:             Qdrant server URL (e.g. ``http://localhost:6333``).
        api_key:         API key for Qdrant Cloud deployments.
        collection_name: Qdrant collection name.
        vector_size:     Dimensionality of the embedding vectors (must match
                         the embedding model output).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "default",
        vector_size: int = 1536,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name
        self._vector_size = vector_size

        # Create collection if it doesn't exist
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

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
        from qdrant_client.models import PointStruct

        points = []
        for doc_id, emb, text, meta in zip(ids, embeddings, texts, metadatas):
            if "_text" in meta:
                logger.warning(
                    "vectorstore.qdrant.reserved_key",
                    key="_text",
                    id=doc_id,
                    msg="Metadata key '_text' is reserved and will be overwritten",
                )
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload={**meta, "_text": text},
                )
            )

        for i in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[i : i + _UPSERT_BATCH],
            )

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=filter,
            with_payload=True,
            with_vectors=include_embeddings,
        )

        hits: list[dict[str, Any]] = []
        for point in results:
            payload = dict(point.payload or {})
            text = payload.pop("_text", "")
            hit: dict[str, Any] = {
                "id": str(point.id),
                "text": text,
                "metadata": payload,
                "distance": 1.0 - point.score,  # cosine similarity → distance
            }
            if include_embeddings and point.vector:
                hit["embeddings"] = point.vector
            hits.append(hit)
        return hits

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(collection_name=name)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    def collection_stats(self, name: str) -> dict[str, Any]:
        info = self._client.get_collection(collection_name=name)
        return {
            "name": name,
            "chunks": info.points_count or 0,
        }
