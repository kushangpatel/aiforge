"""Unit tests for PineconeVectorStore (mocked — no real Pinecone calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: mock the pinecone import so tests don't need pinecone-client
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pinecone():
    """Patch ``pinecone.Pinecone`` and return (mock_pc_cls, mock_index)."""
    mock_index = MagicMock()
    mock_pc_cls = MagicMock()
    mock_pc_cls.return_value.Index.return_value = mock_index

    with patch.dict("sys.modules", {"pinecone": MagicMock()}):
        import sys

        sys.modules["pinecone"].Pinecone = mock_pc_cls
        yield mock_pc_cls, mock_index


@pytest.fixture()
def store(mock_pinecone):
    """Create a PineconeVectorStore with a mocked backend."""
    from aiforge.core.vectorstore_pinecone import PineconeVectorStore

    _, mock_index = mock_pinecone
    s = PineconeVectorStore(
        api_key="pk-test",
        index_name="my-index",
        namespace="test-ns",
    )
    return s, mock_index


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_calls_upsert_with_correct_vectors(self, store) -> None:
        s, mock_index = store
        s.add_documents(
            texts=["hello", "world"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "1"}, {"k": "2"}],
            ids=["id-1", "id-2"],
        )
        mock_index.upsert.assert_called_once()
        args = mock_index.upsert.call_args
        vectors = args[1]["vectors"] if "vectors" in args[1] else args[0][0]
        assert len(vectors) == 2
        assert vectors[0]["id"] == "id-1"
        assert vectors[0]["metadata"]["_text"] == "hello"
        assert vectors[0]["metadata"]["k"] == "1"
        assert args[1]["namespace"] == "test-ns"

    def test_batches_at_100(self, store) -> None:
        s, mock_index = store
        n = 250
        s.add_documents(
            texts=[f"t{i}" for i in range(n)],
            embeddings=[[float(i)] for i in range(n)],
            metadatas=[{"i": i} for i in range(n)],
            ids=[f"id-{i}" for i in range(n)],
        )
        assert mock_index.upsert.call_count == 3  # 100 + 100 + 50


class TestQuery:
    def _mock_response(self, include_values: bool = False):
        match = {
            "id": "doc-1",
            "score": 0.95,
            "metadata": {"_text": "hello world", "source": "a.txt"},
        }
        if include_values:
            match["values"] = [0.1, 0.2]
        return {"matches": [match]}

    def test_returns_hits(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = self._mock_response()
        hits = s.query(embedding=[0.1, 0.2], top_k=3)

        assert len(hits) == 1
        assert hits[0]["id"] == "doc-1"
        assert hits[0]["text"] == "hello world"
        assert hits[0]["metadata"] == {"source": "a.txt"}

    def test_converts_similarity_to_distance(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = self._mock_response()
        hits = s.query(embedding=[0.1])
        assert abs(hits[0]["distance"] - 0.05) < 1e-9  # 1 - 0.95

    def test_include_embeddings(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = self._mock_response(include_values=True)
        hits = s.query(embedding=[0.1], include_embeddings=True)
        assert hits[0]["embeddings"] == [0.1, 0.2]

    def test_no_embeddings_by_default(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = self._mock_response()
        hits = s.query(embedding=[0.1])
        assert "embeddings" not in hits[0]

    def test_filter_passed_through(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = {"matches": []}
        s.query(embedding=[0.1], filter={"genre": "sci-fi"})
        _, kwargs = mock_index.query.call_args
        assert kwargs["filter"] == {"genre": "sci-fi"}

    def test_empty_matches(self, store) -> None:
        s, mock_index = store
        mock_index.query.return_value = {"matches": []}
        hits = s.query(embedding=[0.1])
        assert hits == []


class TestDeleteCollection:
    def test_calls_delete_all(self, store) -> None:
        s, mock_index = store
        s.delete_collection("my-ns")
        mock_index.delete.assert_called_once_with(
            delete_all=True,
            namespace="my-ns",
        )


class TestListCollections:
    def test_returns_namespaces(self, store) -> None:
        s, mock_index = store
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"ns1": {}, "ns2": {}},
        }
        result = s.list_collections()
        assert set(result) == {"ns1", "ns2"}


class TestCollectionStats:
    def test_returns_count(self, store) -> None:
        s, mock_index = store
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"test-ns": {"vector_count": 42}},
        }
        stats = s.collection_stats("test-ns")
        assert stats["name"] == "test-ns"
        assert stats["chunks"] == 42

    def test_missing_namespace_returns_zero(self, store) -> None:
        s, mock_index = store
        mock_index.describe_index_stats.return_value = {
            "namespaces": {},
        }
        stats = s.collection_stats("missing")
        assert stats["chunks"] == 0


# ---------------------------------------------------------------------------
# Regression tests — bugs/gaps audit
# ---------------------------------------------------------------------------


class TestTextKeyCollision:
    """Fix: When user metadata contains a ``_text`` key, the system's text
    always wins and a warning is logged."""

    def test_system_text_wins(self, store) -> None:
        s, mock_index = store
        s.add_documents(
            texts=["real text"],
            embeddings=[[0.1]],
            metadatas=[{"_text": "user val"}],
            ids=["id-1"],
        )
        vectors = mock_index.upsert.call_args[1]["vectors"]
        assert vectors[0]["metadata"]["_text"] == "real text"

    def test_text_collision_logs_warning(self, store) -> None:
        s, mock_index = store
        import structlog.testing

        with structlog.testing.capture_logs() as cap:
            s.add_documents(
                texts=["real"],
                embeddings=[[0.1]],
                metadatas=[{"_text": "collide"}],
                ids=["id-1"],
            )
        warnings = [e for e in cap if e.get("log_level") == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["key"] == "_text"
