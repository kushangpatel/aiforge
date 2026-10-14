"""Unit tests for PgVectorStore (mocked — no real PostgreSQL calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_psycopg():
    """Patch ``psycopg.connect`` and return the mock connection."""
    mock_conn = MagicMock()
    mock_conn.autocommit = True
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    mock_psycopg_mod = MagicMock()
    mock_psycopg_mod.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg": mock_psycopg_mod}):
        yield mock_conn, mock_cursor


@pytest.fixture()
def store(mock_psycopg):
    """Create a PgVectorStore with a mocked backend."""
    mock_conn, mock_cursor = mock_psycopg
    from aiforge.core.vectorstore_pgvector import PgVectorStore

    s = PgVectorStore(
        dsn="postgresql://u:p@localhost/test",
        collection_name="my_docs",
        vector_size=128,
    )
    return s, mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_extension_and_table(self, store) -> None:
        _, _, mock_cursor = store
        # Schema bootstrapping should issue extension/table/index statements.
        assert mock_cursor.execute.call_count >= 3

    def test_table_name_sanitised(self, store) -> None:
        s, _, _ = store
        assert s._collection_name == "aiforge_my_docs"

    def test_table_name_strips_special_chars(self, mock_psycopg) -> None:
        _, _ = mock_psycopg
        from aiforge.core.vectorstore_pgvector import PgVectorStore

        s = PgVectorStore(
            dsn="postgresql://u:p@localhost/test",
            collection_name="my-weird.name!",
        )
        assert s._collection_name == "aiforge_my_weird_name_"


class TestAddDocuments:
    def test_calls_executemany(self, store) -> None:
        s, _, mock_cursor = store
        # Reset call count after init SQL
        mock_cursor.executemany.reset_mock()

        s.add_documents(
            texts=["hello", "world"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "1"}, {"k": "2"}],
            ids=["id-1", "id-2"],
        )
        mock_cursor.executemany.assert_called_once()
        args = mock_cursor.executemany.call_args
        batch = args[0][1]
        assert len(batch) == 2
        assert batch[0][0] == "id-1"
        assert batch[0][1] == "hello"

    def test_metadata_serialised_as_json(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.executemany.reset_mock()

        s.add_documents(
            texts=["t"],
            embeddings=[[0.1]],
            metadatas=[{"key": "value"}],
            ids=["id-1"],
        )
        batch = mock_cursor.executemany.call_args[0][1]
        assert json.loads(batch[0][2]) == {"key": "value"}

    def test_embedding_as_vec_literal(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.executemany.reset_mock()

        s.add_documents(
            texts=["t"],
            embeddings=[[1.0, 2.0, 3.0]],
            metadatas=[{}],
            ids=["id-1"],
        )
        batch = mock_cursor.executemany.call_args[0][1]
        assert batch[0][3] == "[1.0,2.0,3.0]"

    def test_batches_at_500(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.executemany.reset_mock()

        n = 1200
        s.add_documents(
            texts=[f"t{i}" for i in range(n)],
            embeddings=[[float(i)] for i in range(n)],
            metadatas=[{"i": i} for i in range(n)],
            ids=[f"id-{i}" for i in range(n)],
        )
        assert mock_cursor.executemany.call_count == 3  # 500 + 500 + 200


class TestQuery:
    def test_returns_hits(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = [
            ("doc-1", "hello world", '{"source": "a.txt"}', 0.15),
        ]
        hits = s.query(embedding=[0.1, 0.2], top_k=3)

        assert len(hits) == 1
        assert hits[0]["id"] == "doc-1"
        assert hits[0]["text"] == "hello world"
        assert hits[0]["metadata"] == {"source": "a.txt"}
        assert abs(hits[0]["distance"] - 0.15) < 1e-9

    def test_include_embeddings(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = [
            ("doc-1", "hello", "{}", 0.1, [0.1, 0.2]),
        ]
        hits = s.query(embedding=[0.1], include_embeddings=True)
        assert hits[0]["embeddings"] == [0.1, 0.2]

    def test_empty_results(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = []
        hits = s.query(embedding=[0.1])
        assert hits == []

    def test_metadata_dict_passthrough(self, store) -> None:
        """When DB returns metadata as dict (jsonb), it's passed through."""
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = [
            ("doc-1", "t", {"already": "dict"}, 0.1),
        ]
        hits = s.query(embedding=[0.1])
        assert hits[0]["metadata"] == {"already": "dict"}


class TestDeleteCollection:
    def test_drops_table(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.execute.reset_mock()
        s.delete_collection("my_docs")
        mock_cursor.execute.assert_called_once()


class TestListCollections:
    def test_returns_aiforge_tables(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = [
            ("aiforge_docs",),
            ("aiforge_code",),
        ]
        result = s.list_collections()
        assert result == ["aiforge_docs", "aiforge_code"]


class TestCollectionStats:
    def test_returns_count(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = [("aiforge_my_docs",)]
        mock_cursor.fetchone.return_value = (42,)
        stats = s.collection_stats("my_docs")
        assert stats["name"] == "my_docs"
        assert stats["chunks"] == 42

    def test_missing_collection_returns_zero(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = []
        stats = s.collection_stats("missing")
        assert stats["name"] == "missing"
        assert stats["chunks"] == 0


class TestSanitiseName:
    def test_adds_prefix(self) -> None:
        from aiforge.core.vectorstore_pgvector import PgVectorStore

        assert PgVectorStore._sanitise_name("docs") == "aiforge_docs"

    def test_already_prefixed(self) -> None:
        from aiforge.core.vectorstore_pgvector import PgVectorStore

        assert PgVectorStore._sanitise_name("aiforge_docs") == "aiforge_docs"

    def test_replaces_special_chars(self) -> None:
        from aiforge.core.vectorstore_pgvector import PgVectorStore

        assert PgVectorStore._sanitise_name("my-col.v2") == "aiforge_my_col_v2"


# ---------------------------------------------------------------------------
# Regression tests — bugs/gaps audit
# ---------------------------------------------------------------------------


class TestQueryWithFilter:
    """Fix: query() now applies a JSONB containment filter instead of silently
    ignoring the ``filter`` parameter."""

    def test_filter_adds_where_clause(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = []
        s.query(embedding=[0.1], filter={"source": "a.txt"})
        params = mock_cursor.execute.call_args_list[-1][0][1]
        assert json.dumps({"source": "a.txt"}) in params

    def test_filter_none_no_where(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = []
        s.query(embedding=[0.1], filter=None)
        params = mock_cursor.execute.call_args_list[-1][0][1]
        assert all(p != json.dumps({"source": "a.txt"}) for p in params if isinstance(p, str))

    def test_filter_serialised_as_json(self, store) -> None:
        s, _, mock_cursor = store
        mock_cursor.fetchall.return_value = []
        flt = {"source": "a.txt", "lang": "en"}
        s.query(embedding=[0.1], filter=flt)
        args = mock_cursor.execute.call_args_list[-1]
        # The second param should be the json-serialised filter
        params = args[0][1]
        assert json.dumps(flt) in params


class TestClose:
    """Fix: PgVectorStore now exposes a close() method."""

    def test_close_calls_conn_close(self, store) -> None:
        s, mock_conn, _ = store
        s.close()
        mock_conn.close.assert_called_once()
