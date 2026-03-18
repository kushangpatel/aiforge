"""Unit tests for ChromaVectorStore."""

from __future__ import annotations

import pytest

from aiforge.core.vectorstore import ChromaVectorStore


class TestChromaVectorStore:
    def test_add_and_query(self, temp_vector_store: ChromaVectorStore) -> None:
        """Documents added to the store can be retrieved by similarity search."""
        temp_vector_store.add_documents(
            texts=["The sky is blue.", "Water is wet.", "Fire is hot."],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadatas=[{"src": "a"}, {"src": "b"}, {"src": "c"}],
            ids=["id-a", "id-b", "id-c"],
        )
        results = temp_vector_store.query(
            embedding=[1.0, 0.0, 0.0],
            top_k=1,
        )
        assert len(results) == 1
        assert results[0]["text"] == "The sky is blue."

    def test_collection_stats(self, temp_vector_store: ChromaVectorStore) -> None:
        """collection_stats returns chunk count after adding documents."""
        temp_vector_store.add_documents(
            texts=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "1"}, {"k": "2"}],
            ids=["s1", "s2"],
        )
        stats = temp_vector_store.collection_stats("test_collection")
        assert stats["chunks"] == 2
        assert stats["name"] == "test_collection"

    def test_list_collections(self, temp_vector_store: ChromaVectorStore) -> None:
        """list_collections includes the collection created at init."""
        collections = temp_vector_store.list_collections()
        assert "test_collection" in collections

    def test_delete_collection(self, tmp_path: object) -> None:
        """delete_collection removes it from the list."""
        import tempfile
        import os

        store_a = ChromaVectorStore(
            persist_dir=str(tmp_path) + "/del_test",  # type: ignore[operator]
            collection_name="to_delete",
        )
        store_b = ChromaVectorStore(
            persist_dir=str(tmp_path) + "/del_test",  # type: ignore[operator]
            collection_name="keeper",
        )
        assert "to_delete" in store_a.list_collections()
        store_a.delete_collection("to_delete")
        assert "to_delete" not in store_a.list_collections()
        assert "keeper" in store_b.list_collections()

    def test_query_empty_store_returns_nothing(
        self, temp_vector_store: ChromaVectorStore
    ) -> None:
        """Querying an empty collection returns an empty list."""
        results = temp_vector_store.query(
            embedding=[1.0, 0.0],
            top_k=5,
        )
        assert results == []

    def test_upsert_deduplicates(self, temp_vector_store: ChromaVectorStore) -> None:
        """Adding the same id twice updates rather than duplicates."""
        for _ in range(2):
            temp_vector_store.add_documents(
                texts=["repeated text"],
                embeddings=[[0.5, 0.5, 0.0]],
                metadatas=[{"k": "v"}],
                ids=["doc-1"],
            )
        stats = temp_vector_store.collection_stats("test_collection")
        assert stats["chunks"] == 1
