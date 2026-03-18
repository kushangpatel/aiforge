"""Unit tests for the SemanticSearch module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiforge.modules.search.module import (
    SemanticSearch,
    _coerce_document,
    _estimate_tokens,
)
from aiforge.modules.search.schemas import (
    IndexResponse,
    SearchDocument,
    SearchHit,
    SearchResponse,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SAMPLE_DOCS: list[dict] = [
    {"id": "doc1", "text": "Wireless noise-cancelling headphones", "metadata": {"price": 299}},
    {"id": "doc2", "text": "Portable Bluetooth speaker with bass boost", "metadata": {"price": 79}},
    {"id": "doc3", "text": "USB-C fast charging cable 6ft", "metadata": {"price": 12}},
]


def _make_store(
    query_results: list[dict] | None = None,
    collection_chunks: int = 3,
    collection_name: str = "test",
) -> MagicMock:
    """Return a mock VectorStoreAdapter with sensible defaults."""
    store = MagicMock()
    store.add_documents.return_value = None
    store.query.return_value = query_results if query_results is not None else []
    store.collection_stats.return_value = {
        "name": collection_name,
        "chunks": collection_chunks,
        "persist_dir": "./aiforge_data",
    }
    store.list_collections.return_value = [collection_name]
    store.delete_collection.return_value = None
    return store


def _ss(
    store: MagicMock | None = None,
    collection: str = "test",
    top_k: int = 5,
) -> SemanticSearch:
    """Build a SemanticSearch with a mocked store and embedding provider."""
    mock_store = store if store is not None else _make_store()
    ss = SemanticSearch(collection=collection, top_k=top_k, store=mock_store)

    provider = MagicMock()
    provider.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    provider.embed_async = AsyncMock(
        side_effect=lambda texts: [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    )
    provider.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    provider.embed_query_async = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    provider.model_name = "text-embedding-3-small"
    ss._embedding_provider = provider
    return ss


def _sample_raw_hits() -> list[dict]:
    """Chroma-style raw hits with distances."""
    return [
        {"id": "doc2", "text": "Portable Bluetooth speaker", "metadata": {"price": 79}, "distance": 0.08},
        {"id": "doc1", "text": "Wireless headphones", "metadata": {"price": 299}, "distance": 0.11},
    ]


# ---------------------------------------------------------------------------
# TestSearchDocument
# ---------------------------------------------------------------------------


class TestSearchDocument:
    def test_valid_document_with_metadata(self) -> None:
        doc = SearchDocument(id="d1", text="hello world", metadata={"k": "v"})
        assert doc.id == "d1"
        assert doc.text == "hello world"
        assert doc.metadata == {"k": "v"}

    def test_default_metadata_is_empty(self) -> None:
        doc = SearchDocument(id="d1", text="text")
        assert doc.metadata == {}

    def test_empty_id_raises(self) -> None:
        with pytest.raises(Exception):
            SearchDocument(id="", text="hello")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(Exception):
            SearchDocument(id="d1", text="")


# ---------------------------------------------------------------------------
# TestSearchHit
# ---------------------------------------------------------------------------


class TestSearchHit:
    def test_valid_hit(self) -> None:
        hit = SearchHit(id="d1", text="some result", score=0.85)
        assert hit.score == 0.85
        assert hit.metadata == {}

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(Exception):
            SearchHit(id="1", text="t", score=1.01)

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(Exception):
            SearchHit(id="1", text="t", score=-0.01)


# ---------------------------------------------------------------------------
# TestSearchResponse
# ---------------------------------------------------------------------------


class TestSearchResponse:
    def test_to_text_returns_top_hit_text(self) -> None:
        r = SearchResponse(
            hits=[SearchHit(id="d1", text="best result", score=0.9)],
            query="q",
        )
        assert r.to_text() == "best result"

    def test_to_text_no_hits_returns_empty_string(self) -> None:
        r = SearchResponse(hits=[], query="q")
        assert r.to_text() == ""


# ---------------------------------------------------------------------------
# TestIndexResponse
# ---------------------------------------------------------------------------


class TestIndexResponse:
    def test_to_text_includes_count_and_collection(self) -> None:
        r = IndexResponse(indexed_count=5, collection="products", total_documents=12)
        text = r.to_text()
        assert "5" in text
        assert "products" in text

    def test_to_text_includes_total(self) -> None:
        r = IndexResponse(indexed_count=3, collection="col", total_documents=10)
        assert "10" in r.to_text()


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_coerce_dict_to_search_document(self) -> None:
        doc = _coerce_document({"id": "d1", "text": "hello"})
        assert isinstance(doc, SearchDocument)
        assert doc.id == "d1"

    def test_coerce_already_search_document_passthrough(self) -> None:
        original = SearchDocument(id="d1", text="hello")
        result = _coerce_document(original)
        assert result is original

    def test_estimate_tokens_non_empty(self) -> None:
        # "hello world" = 11 chars → 11 // 4 = 2, but clamped to max(1, 2) = 2
        assert _estimate_tokens(["hello world"]) == 2

    def test_estimate_tokens_multiple_texts(self) -> None:
        total_chars = sum(len(t) for t in ["aa", "bbbb"])  # 6 chars → 1
        assert _estimate_tokens(["aa", "bbbb"]) == max(1, total_chars // 4)

    def test_estimate_tokens_empty_list_returns_one(self) -> None:
        # max(1, 0) = 1
        assert _estimate_tokens([]) == 1


# ---------------------------------------------------------------------------
# TestSemanticSearchConstructor
# ---------------------------------------------------------------------------


class TestSemanticSearchConstructor:
    def test_valid_defaults(self) -> None:
        ss = SemanticSearch(collection="test", store=MagicMock())
        assert ss.collection_name == "test"
        assert ss.default_top_k == 5

    def test_custom_collection_and_top_k(self) -> None:
        ss = SemanticSearch(collection="my_col", top_k=10, store=MagicMock())
        assert ss.collection_name == "my_col"
        assert ss.default_top_k == 10

    def test_strips_whitespace_from_collection(self) -> None:
        ss = SemanticSearch(collection="  mydata  ", store=MagicMock())
        assert ss.collection_name == "mydata"

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection"):
            SemanticSearch(collection="", store=MagicMock())

    def test_whitespace_only_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection"):
            SemanticSearch(collection="   ", store=MagicMock())

    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SemanticSearch(collection="test", top_k=0, store=MagicMock())

    def test_top_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SemanticSearch(collection="test", top_k=-5, store=MagicMock())

    def test_top_k_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SemanticSearch(collection="test", top_k=101, store=MagicMock())

    def test_store_injection(self) -> None:
        mock_store = MagicMock()
        ss = SemanticSearch(collection="test", store=mock_store)
        assert ss._store is mock_store


# ---------------------------------------------------------------------------
# TestIndex
# ---------------------------------------------------------------------------


class TestIndex:
    def test_index_returns_correct_counts(self) -> None:
        store = _make_store(collection_chunks=3)
        ss = _ss(store=store)
        result = ss.index(SAMPLE_DOCS)
        assert result.indexed_count == 3
        assert result.collection == "test"
        assert result.total_documents == 3

    def test_index_calls_embed_and_add_documents(self) -> None:
        store = _make_store()
        ss = _ss(store=store)
        ss.index(SAMPLE_DOCS)
        assert ss._embedding_provider.embed.called
        assert store.add_documents.called

    def test_index_accepts_search_document_instances(self) -> None:
        docs = [SearchDocument(id="d1", text="hello")]
        store = _make_store(collection_chunks=1)
        ss = _ss(store=store)
        result = ss.index(docs)
        assert result.indexed_count == 1

    def test_index_empty_list_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            ss.index([])

    def test_index_records_embedding_cost(self) -> None:
        ss = _ss()
        result = ss.index(SAMPLE_DOCS)
        # text-embedding-3-small has a non-zero input rate
        assert result.cost >= 0.0

    def test_index_mixed_dicts_and_documents(self) -> None:
        mixed: list[Any] = [
            {"id": "a", "text": "dict document"},
            SearchDocument(id="b", text="typed document"),
        ]
        store = _make_store(collection_chunks=2)
        ss = _ss(store=store)
        result = ss.index(mixed)
        assert result.indexed_count == 2

    @pytest.mark.asyncio
    async def test_aindex_returns_correct_count(self) -> None:
        store = _make_store(collection_chunks=2)
        ss = _ss(store=store)
        result = await ss.aindex([
            {"id": "a", "text": "async document one"},
            {"id": "b", "text": "async document two"},
        ])
        assert result.indexed_count == 2
        assert result.collection == "test"

    @pytest.mark.asyncio
    async def test_aindex_empty_list_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            await ss.aindex([])

    @pytest.mark.asyncio
    async def test_aindex_uses_async_embedding(self) -> None:
        store = _make_store(collection_chunks=1)
        ss = _ss(store=store)
        await ss.aindex([{"id": "x", "text": "doc"}])
        assert ss._embedding_provider.embed_async.called


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_hits_with_correct_count(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = ss.query("audio device")
        assert len(result.hits) == 2
        assert result.query == "audio device"

    def test_query_score_is_one_minus_distance(self) -> None:
        store = _make_store(query_results=[
            {"id": "d1", "text": "t", "metadata": {}, "distance": 0.2},
        ])
        ss = _ss(store=store)
        result = ss.query("q")
        assert abs(result.hits[0].score - 0.8) < 1e-9

    def test_query_score_clamped_below_zero(self) -> None:
        # distance > 1.0 should clamp score to 0.0
        store = _make_store(query_results=[
            {"id": "d1", "text": "t", "metadata": {}, "distance": 1.1},
        ])
        ss = _ss(store=store)
        result = ss.query("q")
        assert result.hits[0].score == 0.0

    def test_query_score_clamped_above_one(self) -> None:
        # negative distance (can happen with certain distance metrics)
        store = _make_store(query_results=[
            {"id": "d1", "text": "t", "metadata": {}, "distance": -0.1},
        ])
        ss = _ss(store=store)
        result = ss.query("q")
        assert result.hits[0].score == 1.0

    def test_query_top_k_override_passed_to_store(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        ss.query("q", top_k=2)
        assert store.query.call_args.kwargs["top_k"] == 2

    def test_query_filter_passed_to_store(self) -> None:
        store = _make_store(query_results=[])
        ss = _ss(store=store)
        ss.query("q", filter={"price": {"$lt": 100}})
        assert store.query.call_args.kwargs["filter"] == {"price": {"$lt": 100}}

    def test_query_empty_string_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            ss.query("")

    def test_query_whitespace_only_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            ss.query("   ")

    def test_query_top_k_zero_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="top_k"):
            ss.query("q", top_k=0)

    def test_query_top_k_above_max_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="top_k"):
            ss.query("q", top_k=101)

    def test_query_empty_store_returns_no_hits(self) -> None:
        store = _make_store(query_results=[], collection_chunks=0)
        ss = _ss(store=store)
        result = ss.query("anything")
        assert result.hits == []
        assert result.total_searched == 0

    def test_query_uses_embed_query(self) -> None:
        store = _make_store(query_results=[])
        ss = _ss(store=store)
        ss.query("test query")
        assert ss._embedding_provider.embed_query.called

    def test_query_records_cost(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = ss.query("some long query text here")
        assert result.cost >= 0.0

    @pytest.mark.asyncio
    async def test_aquery_returns_hits(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = await ss.aquery("audio device")
        assert len(result.hits) == 2

    @pytest.mark.asyncio
    async def test_aquery_empty_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            await ss.aquery("")

    @pytest.mark.asyncio
    async def test_aquery_uses_embed_query_async(self) -> None:
        store = _make_store(query_results=[])
        ss = _ss(store=store)
        await ss.aquery("q")
        assert ss._embedding_provider.embed_query_async.called


# ---------------------------------------------------------------------------
# TestRunArun
# ---------------------------------------------------------------------------


class TestRunArun:
    def test_run_delegates_to_query(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = ss.run("audio device")
        assert isinstance(result, SearchResponse)
        assert len(result.hits) == 2

    @pytest.mark.asyncio
    async def test_arun_delegates_to_aquery(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = await ss.arun("audio device")
        assert isinstance(result, SearchResponse)
        assert len(result.hits) == 2

    def test_to_text_returns_top_hit_text(self) -> None:
        store = _make_store(query_results=_sample_raw_hits())
        ss = _ss(store=store)
        result = ss.run("q")
        assert result.to_text() == result.hits[0].text

    def test_run_empty_query_raises(self) -> None:
        ss = _ss()
        with pytest.raises(ValueError, match="empty"):
            ss.run("")


# ---------------------------------------------------------------------------
# TestCollectionManagement
# ---------------------------------------------------------------------------


class TestCollectionManagement:
    def test_list_collections_returns_store_result(self) -> None:
        store = _make_store(collection_name="test")
        ss = _ss(store=store)
        cols = ss.list_collections()
        assert "test" in cols

    def test_collection_stats_uses_instance_collection_by_default(self) -> None:
        store = _make_store()
        ss = _ss(store=store, collection="mydata")
        ss.collection_stats()
        store.collection_stats.assert_called_with("mydata")

    def test_collection_stats_with_explicit_name(self) -> None:
        store = _make_store()
        ss = _ss(store=store)
        ss.collection_stats("other_col")
        store.collection_stats.assert_called_with("other_col")

    def test_delete_collection_uses_instance_collection_by_default(self) -> None:
        store = _make_store()
        ss = _ss(store=store, collection="mydata")
        ss.delete_collection()
        store.delete_collection.assert_called_with("mydata")

    def test_delete_collection_with_explicit_name(self) -> None:
        store = _make_store()
        ss = _ss(store=store)
        ss.delete_collection("old_data")
        store.delete_collection.assert_called_with("old_data")


# ---------------------------------------------------------------------------
# TestPipelineCompatibility
# ---------------------------------------------------------------------------


class TestPipelineCompatibility:
    def test_run_returns_search_response(self) -> None:
        ss = _ss()
        result = ss.run("query")
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_arun_returns_search_response(self) -> None:
        ss = _ss()
        result = await ss.arun("query")
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_sync_async_parity(self) -> None:
        """run() and arun() return responses with the same shape."""
        store = _make_store(query_results=_sample_raw_hits())
        ss_sync = _ss(store=store)
        ss_async = _ss(store=_make_store(query_results=_sample_raw_hits()))
        sync_result = ss_sync.run("audio device")
        async_result = await ss_async.arun("audio device")
        assert len(sync_result.hits) == len(async_result.hits)
        assert sync_result.query == async_result.query


# ---------------------------------------------------------------------------
# TestSearchServerRoute
# ---------------------------------------------------------------------------


class TestSearchServerRoute:
    def _client(self) -> Any:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from aiforge.server.routes.search import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_index_returns_200(self) -> None:
        from unittest.mock import AsyncMock, patch

        client = self._client()
        mock_result = IndexResponse(
            indexed_count=2,
            collection="default",
            total_documents=2,
            model="text-embedding-3-small",
        )
        with patch("aiforge.modules.search.module.SemanticSearch") as MockSS:
            MockSS.return_value.aindex = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/search/index",
                json={"documents": [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]},
            )
        assert resp.status_code == 200
        assert resp.json()["indexed_count"] == 2

    def test_index_empty_documents_returns_422(self) -> None:
        client = self._client()
        resp = client.post("/search/index", json={"documents": []})
        assert resp.status_code == 422

    def test_query_returns_200(self) -> None:
        from unittest.mock import AsyncMock, patch

        client = self._client()
        mock_result = SearchResponse(
            hits=[SearchHit(id="d1", text="result text", score=0.9)],
            query="test",
            total_searched=5,
        )
        with patch("aiforge.modules.search.module.SemanticSearch") as MockSS:
            MockSS.return_value.aquery = AsyncMock(return_value=mock_result)
            resp = client.post("/search/query", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) == 1
        assert data["hits"][0]["score"] == pytest.approx(0.9)

    def test_query_empty_string_returns_422(self) -> None:
        client = self._client()
        resp = client.post("/search/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_top_k_zero_returns_422(self) -> None:
        client = self._client()
        resp = client.post("/search/query", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

    def test_query_top_k_above_max_returns_422(self) -> None:
        client = self._client()
        resp = client.post("/search/query", json={"query": "test", "top_k": 101})
        assert resp.status_code == 422

    def test_query_with_filter_passes_through(self) -> None:
        from unittest.mock import AsyncMock, patch

        client = self._client()
        mock_result = SearchResponse(hits=[], query="test", total_searched=0)
        with patch("aiforge.modules.search.module.SemanticSearch") as MockSS:
            MockSS.return_value.aquery = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/search/query",
                json={"query": "test", "filter": {"price": {"$lt": 100}}},
            )
        assert resp.status_code == 200
        # Verify filter was forwarded
        call_kwargs = MockSS.return_value.aquery.call_args.kwargs
        assert call_kwargs["filter"] == {"price": {"$lt": 100}}

    def test_index_value_error_returns_422(self) -> None:
        from unittest.mock import AsyncMock, patch

        client = self._client()
        with patch("aiforge.modules.search.module.SemanticSearch") as MockSS:
            MockSS.return_value.aindex = AsyncMock(
                side_effect=ValueError("documents must not be empty")
            )
            resp = client.post(
                "/search/index",
                json={"documents": [{"id": "1", "text": "hi"}]},
            )
        assert resp.status_code == 422
