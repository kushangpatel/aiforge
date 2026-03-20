"""Unit tests for QdrantVectorStore (mocked — no real Qdrant calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for qdrant_client.models
# ---------------------------------------------------------------------------

class _FakeCollectionInfo:
    def __init__(self, name: str):
        self.name = name


class _FakeGetCollectionsResponse:
    def __init__(self, names: list[str]):
        self.collections = [_FakeCollectionInfo(n) for n in names]


class _FakeSearchPoint:
    def __init__(self, id, score, payload, vector=None):
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector


class _FakeCollectionDetail:
    def __init__(self, points_count):
        self.points_count = points_count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_qdrant():
    """Patch qdrant_client and return the mock client instance."""
    mock_client = MagicMock()
    # Default: no existing collections → trigger create_collection
    mock_client.get_collections.return_value = _FakeGetCollectionsResponse([])

    mock_qdrant_module = MagicMock()
    mock_qdrant_module.QdrantClient.return_value = mock_client

    mock_models = MagicMock()

    with patch.dict("sys.modules", {
        "qdrant_client": mock_qdrant_module,
        "qdrant_client.models": mock_models,
    }):
        yield mock_client, mock_models


@pytest.fixture()
def store(mock_qdrant):
    """Create a QdrantVectorStore with a mocked backend."""
    mock_client, mock_models = mock_qdrant
    from aiforge.core.vectorstore_qdrant import QdrantVectorStore

    s = QdrantVectorStore(
        url="http://localhost:6333",
        api_key="qd-key",
        collection_name="test_col",
        vector_size=128,
    )
    return s, mock_client, mock_models


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_collection_on_init(self, store) -> None:
        _, mock_client, _ = store
        mock_client.create_collection.assert_called_once()
        args = mock_client.create_collection.call_args
        assert args[1]["collection_name"] == "test_col"

    def test_skips_create_if_exists(self, mock_qdrant) -> None:
        mock_client, _ = mock_qdrant
        mock_client.get_collections.return_value = _FakeGetCollectionsResponse(
            ["test_col"]
        )
        from aiforge.core.vectorstore_qdrant import QdrantVectorStore

        QdrantVectorStore(
            url="http://localhost:6333",
            collection_name="test_col",
        )
        mock_client.create_collection.assert_not_called()


class TestAddDocuments:
    def test_calls_upsert(self, store) -> None:
        s, mock_client, mock_models = store
        s.add_documents(
            texts=["hello", "world"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "1"}, {"k": "2"}],
            ids=["id-1", "id-2"],
        )
        mock_client.upsert.assert_called_once()
        args = mock_client.upsert.call_args
        assert args[1]["collection_name"] == "test_col"

    def test_batches_at_100(self, store) -> None:
        s, mock_client, _ = store
        n = 250
        s.add_documents(
            texts=[f"t{i}" for i in range(n)],
            embeddings=[[float(i)] for i in range(n)],
            metadatas=[{"i": i} for i in range(n)],
            ids=[f"id-{i}" for i in range(n)],
        )
        assert mock_client.upsert.call_count == 3


class TestQuery:
    def _mock_results(self, include_vector=False):
        return [
            _FakeSearchPoint(
                id="doc-1",
                score=0.9,
                payload={"_text": "hello world", "source": "a.txt"},
                vector=[0.1, 0.2] if include_vector else None,
            ),
        ]

    def test_returns_hits(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = self._mock_results()
        hits = s.query(embedding=[0.1, 0.2], top_k=5)

        assert len(hits) == 1
        assert hits[0]["id"] == "doc-1"
        assert hits[0]["text"] == "hello world"
        assert hits[0]["metadata"] == {"source": "a.txt"}

    def test_converts_similarity_to_distance(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = self._mock_results()
        hits = s.query(embedding=[0.1])
        assert abs(hits[0]["distance"] - 0.1) < 1e-9  # 1 - 0.9

    def test_include_embeddings(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = self._mock_results(include_vector=True)
        hits = s.query(embedding=[0.1], include_embeddings=True)
        assert hits[0]["embeddings"] == [0.1, 0.2]

    def test_no_embeddings_by_default(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = self._mock_results()
        hits = s.query(embedding=[0.1])
        assert "embeddings" not in hits[0]

    def test_filter_passed_through(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = []
        s.query(embedding=[0.1], filter={"genre": "sci-fi"})
        _, kwargs = mock_client.search.call_args
        assert kwargs["query_filter"] == {"genre": "sci-fi"}

    def test_empty_results(self, store) -> None:
        s, mock_client, _ = store
        mock_client.search.return_value = []
        hits = s.query(embedding=[0.1])
        assert hits == []


class TestDeleteCollection:
    def test_calls_delete_collection(self, store) -> None:
        s, mock_client, _ = store
        s.delete_collection("old_col")
        mock_client.delete_collection.assert_called_once_with(
            collection_name="old_col",
        )


class TestListCollections:
    def test_returns_names(self, store) -> None:
        s, mock_client, _ = store
        mock_client.get_collections.return_value = _FakeGetCollectionsResponse(
            ["col_a", "col_b"]
        )
        assert set(s.list_collections()) == {"col_a", "col_b"}


class TestCollectionStats:
    def test_returns_count(self, store) -> None:
        s, mock_client, _ = store
        mock_client.get_collection.return_value = _FakeCollectionDetail(
            points_count=99,
        )
        stats = s.collection_stats("test_col")
        assert stats["name"] == "test_col"
        assert stats["chunks"] == 99


# ---------------------------------------------------------------------------
# Regression tests — bugs/gaps audit
# ---------------------------------------------------------------------------


class TestTextKeyCollision:
    """Fix: ``_text`` in user metadata no longer overwrites the system text.
    The system's ``_text`` always wins, and a warning is logged."""

    def test_system_text_wins_over_user_meta(self, store) -> None:
        s, mock_client, mock_models = store
        s.add_documents(
            texts=["real text"],
            embeddings=[[0.1]],
            metadatas=[{"_text": "user val"}],
            ids=["id-1"],
        )
        # Inspect the kwargs passed to the mocked PointStruct constructor
        ps_call = mock_models.PointStruct.call_args
        payload = ps_call[1]["payload"]
        assert payload["_text"] == "real text"

    def test_text_collision_logs_warning(self, store) -> None:
        s, mock_client, _ = store
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
