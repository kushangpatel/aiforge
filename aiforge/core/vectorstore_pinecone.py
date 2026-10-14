"""Pinecone vector store adapter.

Each AIForge "collection" maps to a Pinecone **namespace** within the
configured index.  Text is stored in ``metadata["_text"]`` because
Pinecone is a pure-vector store with no native text column.
"""

from __future__ import annotations

from typing import Any

import structlog

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
        for doc_id, emb, text, meta in zip(ids, embeddings, texts, metadatas, strict=False):
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
        raw = self._index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self._namespace,
            filter=filter,
            include_metadata=True,
            include_values=include_embeddings,
        )
        matches = raw.get("matches", []) if isinstance(raw, dict) else getattr(raw, "matches", [])

        hits: list[dict[str, Any]] = []
        for match in matches:
            meta_raw = match.get("metadata", {}) if isinstance(match, dict) else (match.metadata or {})
            meta = dict(meta_raw)
            text = meta.pop("_text", "")
            match_id = match.get("id") if isinstance(match, dict) else match.id
            score = match.get("score", 0.0) if isinstance(match, dict) else match.score
            hit: dict[str, Any] = {
                "id": match_id,
                "text": text,
                "metadata": meta,
                "distance": 1.0 - score,  # similarity -> distance
            }
            values = match.get("values") if isinstance(match, dict) else getattr(match, "values", None)
            if include_embeddings and values is not None:
                hit["embeddings"] = values
            hits.append(hit)
        return hits

    def delete_collection(self, name: str) -> None:
        self._index.delete(delete_all=True, namespace=name)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks whose metadata['source'] == *source*.

        Uses Pinecone's native filter-based delete (serverless indexes) which
        avoids needing to know vector dimensionality.  Returns 0 because
        Pinecone's delete API does not report a count; callers should treat
        the return value as informational only for Pinecone.
        """
        self._index.delete(
            filter={"source": {"$eq": source}},
            namespace=self._namespace,
        )
        # Pinecone delete does not return the number of deleted vectors.
        return 0

    def list_all_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for every vector in this namespace.

        Uses ``list()`` to page all IDs, then ``fetch()`` in batches to
        retrieve their metadata payloads.
        """
        all_metadata: list[dict[str, Any]] = []
        batch_size = 1000  # Pinecone fetch limit per call

        # list() returns an iterator of ID pages: plain lists of str IDs in older
        # SDKs, ListResponse pages whose ``vectors`` hold ListItem(id=...) in newer ones.
        for id_page in self._index.list(namespace=self._namespace):
            items: Any = getattr(id_page, "vectors", id_page)
            ids = [item if isinstance(item, str) else item.id for item in items]
            ids = [i for i in ids if i]
            if not ids:
                continue
            for i in range(0, len(ids), batch_size):
                chunk_ids = ids[i : i + batch_size]
                fetch_result = self._index.fetch(
                    ids=chunk_ids, namespace=self._namespace
                )
                for vec in fetch_result.vectors.values():
                    meta = dict(vec.metadata or {})
                    meta.pop("_text", None)  # strip internal text field
                    all_metadata.append(meta)
        return all_metadata

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
