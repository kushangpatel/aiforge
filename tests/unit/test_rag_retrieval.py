"""Unit tests for RAG upgrades: BM25, RRF, hybrid retrieval, reranking and chunking."""

from __future__ import annotations

import hashlib
import math
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog
from langchain_core.documents import Document

from aiforge.core.rerank import (
    LLMReranker,
    Reranker,
    RerankScores,
    ScoreFunctionReranker,
    apply_reranker,
)
from aiforge.core.retrieval import (
    BM25Index,
    Retriever,
    UnsupportedFilterError,
    check_filter,
    match_filter,
    reciprocal_rank_fusion,
    tokenize,
    validate_retrieval_mode,
)
from aiforge.core.text_splitter import TextSplitter, validate_chunking
from aiforge.core.types import CompletionResult
from aiforge.core.vectorstore import ChromaVectorStore, VectorStoreAdapter
from aiforge.infra.config import AIForgeSettings
from aiforge.modules.qa.agent import chunk_label, format_context
from aiforge.modules.qa.module import DocQA
from aiforge.modules.qa.schemas import Citation
from aiforge.modules.search.module import SemanticSearch

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_DIM = 64


def _fake_vector(text: str) -> list[float]:
    """Deterministic bag-of-words embedding: hashed token counts, L2-normalised."""
    vec = [0.0] * _DIM
    vec[0] = 0.05  # never a zero vector
    for tok in tokenize(text):
        idx = 1 + int(hashlib.md5(tok.encode()).hexdigest(), 16) % (_DIM - 1)  # noqa: S324
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class FakeEmbeddings:
    """Offline stand-in for an EmbeddingProvider."""

    model_name = "text-embedding-3-small"

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vector(t) for t in texts]

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return _fake_vector(text)

    async def embed_query_async(self, text: str) -> list[float]:
        return self.embed_query(text)


def _completion(text: str = "", parsed: dict[str, Any] | None = None) -> CompletionResult:
    return CompletionResult(
        text=text, input_tokens=40, output_tokens=10, model="gpt-4o-mini", parsed=parsed
    )


class FakeProvider:
    """Minimal LLMProvider stand-in recording calls."""

    def __init__(self, result: CompletionResult | Exception, structured: bool = True) -> None:
        self.result = result
        self.supports_structured_output = structured
        self.supports_tools = False
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class NoListStore(VectorStoreAdapter):
    """A custom backend that does not implement ``list_all_documents``."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.queries: list[dict[str, Any]] = []

    def add_documents(self, texts, embeddings, metadatas, ids):  # type: ignore[no-untyped-def]
        pass

    def query(self, embedding, top_k=5, filter=None, include_embeddings=False):  # type: ignore[no-untyped-def]
        self.queries.append({"top_k": top_k, "filter": filter})
        return [dict(h) for h in self.hits[:top_k]]

    def delete_collection(self, name):  # type: ignore[no-untyped-def]
        pass

    def delete_by_source(self, source):  # type: ignore[no-untyped-def]
        return 0

    def list_all_metadata(self):  # type: ignore[no-untyped-def]
        return []

    def list_collections(self):  # type: ignore[no-untyped-def]
        return ["c"]

    def collection_stats(self, name):  # type: ignore[no-untyped-def]
        return {"name": name, "chunks": len(self.hits)}


CORPUS = {
    "auth": "The auth middleware validates JWT bearer tokens on every request.",
    "db": "Database connections are pooled with a maximum of twenty sessions.",
    "cache": "The request cache stores completions in memory with a TTL.",
    "err": "Error code E4012 means the upstream quota was exhausted.",
}


@pytest.fixture()
def chroma(tmp_path: Any) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="rag")


def _search(store: VectorStoreAdapter, **kwargs: Any) -> SemanticSearch:
    ss = SemanticSearch(collection="rag", store=store, **kwargs)
    ss._embedding_provider = FakeEmbeddings()  # type: ignore[assignment]
    return ss


def _indexed_search(store: ChromaVectorStore, **kwargs: Any) -> SemanticSearch:
    ss = _search(store, **kwargs)
    ss.index([{"id": k, "text": v, "metadata": {"topic": k}} for k, v in CORPUS.items()])
    return ss


def _docqa(store: VectorStoreAdapter, **kwargs: Any) -> DocQA:
    qa = DocQA(collection="rag", store=store, **kwargs)
    qa._embedding_provider = FakeEmbeddings()  # type: ignore[assignment]
    return qa


def _ingest_markdown(qa: DocQA, tmp_path: Any) -> None:
    md = tmp_path / "guide.md"
    md.write_text(
        "# Guide\n\nIntro to the project.\n\n"
        "## Install\n\nRun pip install aiforge to install the package.\n\n"
        "## Errors\n\nError code E4012 means the upstream quota was exhausted.\n"
    )
    (tmp_path / "notes.txt").write_text(CORPUS["db"])
    qa.ingest(str(tmp_path))


# ===========================================================================
# Tokenizer, settings, validation
# ===========================================================================


class TestBasics:
    def test_tokenize_lowercases_and_splits_underscores(self) -> None:
        assert tokenize("Auth_Middleware: JWT-token v2!") == [
            "auth",
            "middleware",
            "jwt",
            "token",
            "v2",
        ]

    def test_validate_retrieval_mode(self) -> None:
        assert validate_retrieval_mode(" Hybrid ") == "hybrid"
        with pytest.raises(ValueError, match="retrieval_mode must be one of"):
            validate_retrieval_mode("semantic")

    def test_validate_chunking(self) -> None:
        assert validate_chunking("TOKEN") == "token"
        with pytest.raises(ValueError, match="chunking must be one of"):
            validate_chunking("sentences")

    def test_settings_defaults(self) -> None:
        cfg = AIForgeSettings()
        assert cfg.RETRIEVAL_MODE == "vector"
        assert cfg.RERANK is False
        assert cfg.RERANK_TOP_N is None
        assert cfg.FETCH_K == 20
        assert cfg.RRF_K == 60
        assert cfg.CHUNKING == "recursive"

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIFORGE_RETRIEVAL_MODE", "hybrid")
        monkeypatch.setenv("AIFORGE_CHUNKING", "markdown")
        monkeypatch.setenv("AIFORGE_RERANK", "true")
        cfg = AIForgeSettings()
        assert cfg.RETRIEVAL_MODE == "hybrid"
        assert cfg.CHUNKING == "markdown"
        assert cfg.RERANK is True

    def test_module_defaults_backward_compatible(self, chroma: ChromaVectorStore) -> None:
        qa = _docqa(chroma)
        ss = _search(chroma)
        assert (qa.retrieval_mode, qa.rerank, qa.chunking) == ("vector", False, "recursive")
        assert (ss.retrieval_mode, ss.rerank) == ("vector", False)

    def test_invalid_constructor_args(self, chroma: ChromaVectorStore) -> None:
        with pytest.raises(ValueError, match="retrieval_mode"):
            _docqa(chroma, retrieval_mode="fuzzy")
        with pytest.raises(ValueError, match="chunking"):
            _docqa(chroma, chunking="words")
        with pytest.raises(ValueError, match="fetch_k"):
            _search(chroma, fetch_k=0)
        with pytest.raises(ValueError, match="rerank_top_n"):
            _search(chroma, rerank_top_n=0)


# ===========================================================================
# BM25
# ===========================================================================


class TestBM25:
    def test_score_matches_okapi_formula(self) -> None:
        idx = BM25Index(k1=1.5, b=0.75)
        idx.add(["a", "b", "c"], ["apple banana apple", "banana cherry", "cherry date fig grape"])
        n, avgdl = 3, (3 + 2 + 4) / 3
        idf_apple = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
        tf, dl = 2, 3
        expected = idf_apple * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
        assert idx.scores("apple")["a"] == pytest.approx(expected)
        assert idx.idf("apple") == pytest.approx(idf_apple)

    def test_ranking_and_zero_score_exclusion(self) -> None:
        idx = BM25Index()
        idx.add(list(CORPUS), list(CORPUS.values()))
        hits = idx.search("JWT bearer tokens", top_k=5)
        assert [h["id"] for h in hits] == ["auth"]
        assert hits[0]["bm25"] > 0

    def test_rare_term_outweighs_common_term(self) -> None:
        idx = BM25Index()
        idx.add(["x", "y", "z"], ["the quota the", "the cache", "the pool"])
        hits = idx.search("the quota")
        assert hits[0]["id"] == "x"
        assert idx.idf("quota") > idx.idf("the") > 0  # IDF stays positive

    def test_upsert_and_remove(self) -> None:
        idx = BM25Index()
        idx.add(["a"], ["alpha"])
        idx.add(["a"], ["beta"])
        assert len(idx) == 1
        assert idx.search("alpha") == []
        assert idx.search("beta")[0]["id"] == "a"
        idx.remove(["a", "missing"])
        assert len(idx) == 0
        assert idx.scores("beta") == {}

    def test_search_with_filter(self) -> None:
        idx = BM25Index()
        idx.add(["a", "b"], ["error code", "error log"], [{"lang": "py"}, {"lang": "go"}])
        hits = idx.search("error", filter={"lang": "go"})
        assert [h["id"] for h in hits] == ["b"]

    def test_empty_index(self) -> None:
        assert BM25Index().search("anything") == []
        assert BM25Index().idf("x") == 0.0


class TestFilters:
    def test_match_filter_operators(self) -> None:
        meta = {"price": 50, "source": "repo/auth.py", "tag": "a"}
        assert match_filter(meta, {"price": {"$lt": 100}})
        assert not match_filter(meta, {"price": {"$gte": 100}})
        assert match_filter(meta, {"source": {"$contains": "auth"}})
        assert match_filter(meta, {"tag": {"$in": ["a", "b"]}})
        assert match_filter(meta, {"$or": [{"tag": "z"}, {"price": 50}]})
        assert not match_filter(meta, {"$and": [{"tag": "a"}, {"price": {"$ne": 50}}]})
        assert not match_filter({}, {"price": {"$gt": 1}})

    def test_check_filter_rejects_unknown_operator(self) -> None:
        check_filter({"price": {"$lt": 3}, "$and": [{"a": 1}]})
        with pytest.raises(UnsupportedFilterError):
            check_filter({"text": {"$regex": "x"}})
        with pytest.raises(UnsupportedFilterError):
            check_filter({"$not": {"a": 1}})


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================


class TestRRF:
    def test_fusion_math(self) -> None:
        vec = [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}]
        kw = [{"id": "b", "bm25": 3.0}, {"id": "c", "bm25": 1.0}]
        fused = reciprocal_rank_fusion([vec, kw], k=60)
        rrf = {h["id"]: h["rrf"] for h in fused}
        assert rrf["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert rrf["a"] == pytest.approx(1 / 61)
        assert rrf["c"] == pytest.approx(1 / 62)
        assert [h["id"] for h in fused] == ["b", "a", "c"]
        # Fields from both lists are merged.
        assert fused[0]["distance"] == 0.2 and fused[0]["bm25"] == 3.0

    def test_ties_keep_first_seen_order(self) -> None:
        fused = reciprocal_rank_fusion([[{"id": "x"}], [{"id": "y"}]], k=10)
        assert [h["id"] for h in fused] == ["x", "y"]

    def test_weights(self) -> None:
        fused = reciprocal_rank_fusion([[{"id": "x"}], [{"id": "y"}]], k=0, weights=[1.0, 3.0])
        assert [h["id"] for h in fused] == ["y", "x"]
        assert fused[0]["rrf"] == pytest.approx(3.0)
        with pytest.raises(ValueError, match="one entry per ranking"):
            reciprocal_rank_fusion([[{"id": "x"}]], weights=[1.0, 2.0])


# ===========================================================================
# Retriever
# ===========================================================================


class TestRetriever:
    def test_index_built_lazily_from_chroma_and_rebuilt_on_count_change(
        self, chroma: ChromaVectorStore
    ) -> None:
        chroma.add_documents(["alpha one"], [_fake_vector("alpha one")], [{"k": 1}], ["1"])
        r = Retriever(chroma, "rag", mode="keyword")
        assert r.retrieve("alpha", top_k=3)[0]["id"] == "1"
        first = r.keyword_index()
        # A write from elsewhere changes the count -> index is rebuilt.
        chroma.add_documents(["beta two"], [_fake_vector("beta two")], [{"k": 2}], ["2"])
        assert r.keyword_index() is not first
        assert r.retrieve("beta", top_k=3)[0]["id"] == "2"

    def test_fallback_to_vector_when_store_cannot_list(self) -> None:
        store = NoListStore([{"id": "v", "text": "vec", "metadata": {}, "distance": 0.25}])
        r = Retriever(store, "c", mode="hybrid")
        with structlog.testing.capture_logs() as logs:
            assert r.effective_mode() == "vector"
        assert any(e["event"] == "retrieval.keyword_unsupported" for e in logs)
        hits = r.retrieve("q", top_k=1, embedding=[0.1])
        assert hits[0]["score"] == pytest.approx(0.75)

    def test_unsupported_filter_falls_back_to_vector(self, chroma: ChromaVectorStore) -> None:
        r = Retriever(chroma, "rag", mode="keyword")
        with structlog.testing.capture_logs() as logs:
            assert r.effective_mode({"a": {"$regex": "x"}}) == "vector"
        assert any(e["event"] == "retrieval.filter_unsupported" for e in logs)

    def test_vector_and_hybrid_need_embedding(self, chroma: ChromaVectorStore) -> None:
        chroma.add_documents(["alpha"], [_fake_vector("alpha")], [{"k": 1}], ["1"])
        with pytest.raises(ValueError, match="embedding is required"):
            Retriever(chroma, "rag", mode="hybrid").retrieve("alpha", top_k=1)


# ===========================================================================
# SemanticSearch retrieval modes (real Chroma)
# ===========================================================================


class TestSearchModes:
    def test_vector_mode_unchanged(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma)
        resp = ss.query("JWT bearer tokens auth middleware", top_k=2)
        assert resp.hits[0].id == "auth"
        assert all(0.0 <= h.score <= 1.0 for h in resp.hits)
        assert resp.hits[0].rerank_score is None
        assert resp.cost > 0

    def test_keyword_mode_skips_embedding(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, retrieval_mode="keyword")
        provider = ss._embedding_provider
        before = provider.query_calls  # type: ignore[union-attr]
        resp = ss.query("E4012", top_k=3)
        assert [h.id for h in resp.hits] == ["err"]
        assert resp.hits[0].score == 1.0
        assert provider.query_calls == before  # type: ignore[union-attr]
        assert resp.cost == 0.0 and resp.input_tokens == 0

    def test_hybrid_mode_fuses_both(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, retrieval_mode="hybrid")
        resp = ss.query("quota E4012 exhausted", top_k=3)
        assert resp.hits[0].id == "err"
        # Ranked first by both retrievers -> normalised RRF score of 1.0.
        assert resp.hits[0].score == pytest.approx(1.0)
        assert all(0.0 <= h.score <= 1.0 for h in resp.hits)
        assert resp.total_searched == 4

    def test_hybrid_with_filter_and_embeddings(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, retrieval_mode="hybrid")
        resp = ss.query("cache memory", top_k=4, filter={"topic": "cache"}, include_embeddings=True)
        assert [h.id for h in resp.hits] == ["cache"]
        assert resp.hits[0].embeddings is not None

    def test_index_invalidates_keyword_index(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, retrieval_mode="keyword")
        ss.query("JWT")
        ss.index([{"id": "new", "text": "Kubernetes deployment manifests", "metadata": {"t": 1}}])
        assert ss.query("kubernetes").hits[0].id == "new"

    @pytest.mark.asyncio
    async def test_aquery_hybrid(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, retrieval_mode="hybrid")
        resp = await ss.aquery("pooled database sessions", top_k=2)
        assert resp.hits[0].id == "db"

    def test_custom_backend_falls_back_to_vector(self) -> None:
        store = NoListStore(
            [{"id": "a", "text": "alpha", "metadata": {}, "distance": 0.1}],
        )
        ss = _search(store, retrieval_mode="hybrid")
        resp = ss.query("alpha", top_k=1)
        assert [h.id for h in resp.hits] == ["a"]
        assert resp.hits[0].score == pytest.approx(0.9)


# ===========================================================================
# DocQA retrieval modes (real Chroma)
# ===========================================================================


class TestDocQAModes:
    @pytest.mark.parametrize("mode", ["vector", "hybrid", "keyword"])
    def test_retrieve_each_mode(self, mode: str, chroma: ChromaVectorStore, tmp_path: Any) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        qa = _docqa(chroma, retrieval_mode=mode, chunking="markdown")
        _ingest_markdown(qa, docs)
        hits = qa.retrieve("What does error code E4012 mean?", top_k=2)
        assert hits[0]["metadata"]["heading_path"] == "Guide > Errors"
        assert 0.0 <= hits[0]["score"] <= 1.0

    def test_keyword_mode_does_not_embed_queries(
        self, chroma: ChromaVectorStore, tmp_path: Any
    ) -> None:
        qa = _docqa(chroma, retrieval_mode="keyword")
        _ingest_markdown(qa, tmp_path)
        qa._embedding_provider.query_calls = 0  # type: ignore[union-attr]
        qa.retrieve("pooled sessions")
        assert qa._embedding_provider.query_calls == 0  # type: ignore[union-attr]

    def test_ask_hybrid_returns_citations(self, chroma: ChromaVectorStore, tmp_path: Any) -> None:
        qa = _docqa(chroma, retrieval_mode="hybrid", chunking="markdown", retrieval_k=2)
        _ingest_markdown(qa, tmp_path)
        qa._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=[_completion("factual"), _completion("SUFFICIENT: 0.9"), _completion("ok")]
        )
        resp = qa.ask("error code E4012")
        assert resp.text == "ok"
        assert resp.citations and isinstance(resp.citations[0], Citation)
        top = resp.citations[0]
        assert top.source.endswith("guide.md")
        assert top.heading_path == "Guide > Errors"
        assert top.label().endswith("Guide > Errors")
        assert any(s.endswith("guide.md") for s in resp.sources)
        # The answer prompt labels each passage with its citation.
        respond_prompt = qa._call_llm.call_args_list[-1].args[0][-1]["content"]
        assert "[source: " in respond_prompt and "Guide > Errors]" in respond_prompt

    def test_delete_collection_invalidates(self, chroma: ChromaVectorStore, tmp_path: Any) -> None:
        qa = _docqa(chroma, retrieval_mode="keyword")
        _ingest_markdown(qa, tmp_path)
        qa.retrieve("install")
        assert qa._retriever is not None and qa._retriever._index is not None
        qa.delete_collection()
        assert qa._retriever._index is None


# ===========================================================================
# Reranking
# ===========================================================================

CANDIDATES = [
    {"id": "a", "text": "first", "metadata": {}, "score": 0.9},
    {"id": "b", "text": "second", "metadata": {}, "score": 0.8},
    {"id": "c", "text": "third", "metadata": {}, "score": 0.7},
]


class TestLLMReranker:
    def test_structured_output_path(self) -> None:
        parsed = {"scores": [{"index": 0, "score": 0.1}, {"index": 2, "score": 0.95}]}
        provider = FakeProvider(_completion(parsed=parsed))
        out = LLMReranker(provider).rerank("q", CANDIDATES, top_n=3)  # type: ignore[arg-type]
        assert [h["id"] for h in out] == ["c", "a", "b"]  # unscored "b" goes last
        assert out[0]["rerank_score"] == 0.95 and out[2]["rerank_score"] is None
        call = provider.calls[0]
        assert call["response_format"] is RerankScores
        assert "[2] third" in call["messages"][0]["content"]

    def test_scores_clamped_and_bad_indexes_ignored(self) -> None:
        parsed = {"scores": [{"index": 1, "score": 7}, {"index": 99, "score": 1.0}]}
        out = LLMReranker(FakeProvider(_completion(parsed=parsed))).rerank("q", CANDIDATES, 1)  # type: ignore[arg-type]
        assert [h["id"] for h in out] == ["b"]
        assert out[0]["rerank_score"] == 1.0

    def test_json_text_path_without_native_structured_output(self) -> None:
        text = '```json\n{"scores": [{"index": 1, "score": 0.9}]}\n```'
        provider = FakeProvider(_completion(text=text), structured=False)
        out = LLMReranker(provider).rerank("q", CANDIDATES, 2)  # type: ignore[arg-type]
        assert [h["id"] for h in out] == ["b", "a"]
        assert "response_format" not in provider.calls[0]
        assert "Respond with JSON only" in provider.calls[0]["messages"][0]["content"]

    @pytest.mark.parametrize(
        "result",
        [RuntimeError("provider down"), _completion(text="not json"), _completion(parsed={})],
    )
    def test_fallback_to_original_order(self, result: Any) -> None:
        with structlog.testing.capture_logs() as logs:
            out = LLMReranker(FakeProvider(result)).rerank("q", CANDIDATES, 2)  # type: ignore[arg-type]
        assert [h["id"] for h in out] == ["a", "b"]
        assert any(e["event"] == "rerank.llm_failed" for e in logs)

    def test_routes_through_module_call_llm(self, chroma: ChromaVectorStore) -> None:
        ss = _search(chroma)
        ss._provider = FakeProvider(_completion())  # type: ignore[assignment]
        parsed = {"scores": [{"index": 1, "score": 0.9}]}
        ss._call_llm = MagicMock(return_value=_completion(parsed=parsed))  # type: ignore[method-assign]
        hits, result = LLMReranker(ss).rerank_with_result("q", CANDIDATES, 1)
        assert hits[0]["id"] == "b" and result is not None
        assert ss._call_llm.call_args.kwargs["response_format"] is RerankScores

    def test_empty_candidates(self) -> None:
        provider = FakeProvider(_completion())
        assert LLMReranker(provider).rerank("q", [], 3) == []  # type: ignore[arg-type]
        assert provider.calls == []


class TestCustomRerankers:
    def test_score_function_reranker(self) -> None:
        rr = ScoreFunctionReranker(lambda q, docs: [len(d) for d in docs])
        assert isinstance(rr, Reranker)
        out = rr.rerank("q", CANDIDATES, 2)
        # Lengths 5, 6, 5: "b" wins, then the tie keeps retrieval order.
        assert [h["id"] for h in out] == ["b", "a"]
        assert out[0]["rerank_score"] == 6.0

    def test_score_function_wrong_length_falls_back(self) -> None:
        out = ScoreFunctionReranker(lambda q, docs: [1.0]).rerank("q", CANDIDATES, 2)
        assert [h["id"] for h in out] == ["a", "b"]

    def test_apply_reranker_catches_custom_errors(self) -> None:
        class Broken:
            def rerank(self, query: str, candidates: Any, top_n: int) -> list[dict[str, Any]]:
                raise RuntimeError("boom")

        hits, result = apply_reranker(Broken(), "q", CANDIDATES, 2)
        assert [h["id"] for h in hits] == ["a", "b"] and result is None

    def test_custom_reranker_in_search(self, chroma: ChromaVectorStore) -> None:
        # Prefer the shortest text, regardless of the query.
        rr = ScoreFunctionReranker(lambda q, docs: [-len(d) for d in docs])
        ss = _indexed_search(chroma, reranker=rr, fetch_k=10)
        assert ss.rerank is True
        resp = ss.query("JWT auth", top_k=1)
        shortest = min(CORPUS, key=lambda k: len(CORPUS[k]))
        assert resp.hits[0].id == shortest
        assert resp.hits[0].rerank_score == -len(CORPUS[shortest])

    def test_llm_rerank_in_search_adds_cost(self, chroma: ChromaVectorStore) -> None:
        ss = _indexed_search(chroma, rerank=True, rerank_top_n=1, retrieval_mode="hybrid")
        ss._provider = FakeProvider(_completion())  # type: ignore[assignment]

        # Candidate order from retrieval is unknown, so score every index and
        # find which one the "db" document landed on.
        def fake_call(messages: list[dict[str, Any]], **kw: Any) -> CompletionResult:
            content = messages[0]["content"]
            scores = []
            for line in content.splitlines():
                if line.startswith("[") and "]" in line:
                    idx = int(line[1 : line.index("]")])
                    scores.append({"index": idx, "score": 1.0 if "Database" in line else 0.0})
            return _completion(parsed={"scores": scores})

        ss._call_llm = MagicMock(side_effect=fake_call)  # type: ignore[method-assign]
        resp = ss.query("JWT auth")
        assert len(resp.hits) == 1 and resp.hits[0].id == "db"
        assert resp.output_tokens == 10 and resp.input_tokens > 40

    def test_rerank_in_docqa_fetches_candidates_and_counts_tokens(
        self, chroma: ChromaVectorStore, tmp_path: Any
    ) -> None:
        seen: list[int] = []

        def score(q: str, docs: list[str]) -> list[float]:
            seen.append(len(docs))
            return [1.0 if "pooled" in d else 0.0 for d in docs]

        qa = _docqa(chroma, reranker=ScoreFunctionReranker(score), fetch_k=10, rerank_top_n=1)
        _ingest_markdown(qa, tmp_path)
        hits = qa.retrieve("install the package")
        assert len(hits) == 1 and "pooled" in hits[0]["text"]
        assert seen[0] > 1  # candidates were fetched before reranking

        qa._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=[_completion("factual"), _completion("SUFFICIENT: 0.8"), _completion("a")]
        )
        resp = qa.ask("install")
        assert resp.citations[0].rerank_score == 1.0
        assert resp.input_tokens == 120  # 3 LLM calls, no LLM reranker usage

    def test_llm_rerank_usage_added_to_docqa_totals(
        self, chroma: ChromaVectorStore, tmp_path: Any
    ) -> None:
        qa = _docqa(chroma, rerank=True)
        _ingest_markdown(qa, tmp_path)
        qa._provider = FakeProvider(_completion())  # type: ignore[assignment]
        qa._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _completion("factual"),
                _completion(parsed={"scores": [{"index": 0, "score": 0.5}]}),
                _completion("SUFFICIENT: 0.8"),
                _completion("answer"),
            ]
        )
        resp = qa.ask("install")
        assert resp.text == "answer"
        assert resp.input_tokens == 160 and resp.output_tokens == 40


# ===========================================================================
# Chunking
# ===========================================================================

MARKDOWN = (
    "# Title\n\nIntro paragraph.\n\n"
    "## Setup\n\nInstall steps here.\n\n"
    "```bash\n# not a heading\npip install x\n```\n\n"
    "### Linux\n\nUse apt.\n\n"
    "## Usage\n\nCall the API.\n"
)


class _FakeEncoding:
    """tiktoken.Encoding stand-in: one token per whitespace-separated word.

    tiktoken downloads its BPE files on first use, which is not possible
    offline, so tests stub ``tiktoken.get_encoding``.
    """

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [0] * len(text.split())


@pytest.fixture()
def fake_tiktoken(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import tiktoken

    requested: list[str] = []

    def get_encoding(name: str) -> _FakeEncoding:
        requested.append(name)
        return _FakeEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    return requested


class TestChunking:
    def test_markdown_heading_paths(self) -> None:
        doc = Document(page_content=MARKDOWN, metadata={"source": "README.md", "file_type": "md"})
        chunks = TextSplitter(strategy="markdown").split([doc])
        paths = [c.metadata.get("heading_path") for c in chunks]
        assert paths == ["Title", "Title > Setup", "Title > Setup > Linux", "Title > Usage"]
        assert all(c.metadata["source"] == "README.md" for c in chunks)
        # Code-fence comments are not treated as headings.
        assert "# not a heading" in chunks[1].page_content

    def test_markdown_large_section_is_size_split(self) -> None:
        body = "## Big\n\n" + ("word " * 400)
        doc = Document(page_content=body, metadata={"file_type": "md"})
        chunks = TextSplitter(chunk_size=300, strategy="markdown").split([doc])
        assert len(chunks) > 1
        assert all(c.metadata["heading_path"] == "Big" for c in chunks)
        assert all(len(c.page_content) <= 300 for c in chunks)

    def test_markdown_strategy_leaves_other_files_alone(self) -> None:
        doc = Document(page_content="# comment\nx = 1\n", metadata={"file_type": "py"})
        chunks = TextSplitter(strategy="markdown").split([doc])
        assert "heading_path" not in chunks[0].metadata

    def test_markdown_split_text(self) -> None:
        assert len(TextSplitter(strategy="markdown").split_text(MARKDOWN, ext="md")) == 4

    def test_token_strategy_sizes_in_tiktoken_tokens(
        self, fake_tiktoken: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 200
        chunks = TextSplitter(chunk_size=50, strategy="token").split_text(text, ext="txt")
        assert fake_tiktoken == ["cl100k_base"]
        assert len(chunks) > 1
        assert all(len(_FakeEncoding().encode(c)) <= 50 for c in chunks)
        # Default token overlap is min(60, 50 // 4) = 12 shared tokens.
        words = " ".join(f"w{i}" for i in range(200))
        parts = TextSplitter(chunk_size=50, strategy="token").split_text(words)
        shared = set(parts[0].split()) & set(parts[1].split())
        assert 0 < len(shared) <= 12

    def test_token_strategy_code(self, fake_tiktoken: list[str]) -> None:
        code = "def foo():\n    return 1\n\n" * 200
        chunks = TextSplitter(chunk_size=64, strategy="token").split_text(code, ext="py")
        assert len(chunks) > 1
        assert all(len(_FakeEncoding().encode(c)) <= 64 for c in chunks)

    def test_token_strategy_custom_counter(self) -> None:
        text = "a b c d e f g h i j " * 10
        chunks = TextSplitter(
            chunk_size=10, chunk_overlap=0, strategy="token", token_counter=lambda s: len(s.split())
        ).split_text(text)
        assert len(chunks) == 10
        assert all(len(c.split()) == 10 for c in chunks)

    def test_overlap_defaults(self) -> None:
        ts = TextSplitter()
        assert ts._sizes(1200, 200) == (1200, 200)  # unchanged default
        assert TextSplitter(chunk_size=100)._sizes(1200, 200) == (100, 25)  # scaled down
        assert TextSplitter(chunk_size=100, chunk_overlap=0)._sizes(1200, 200) == (100, 0)
        assert TextSplitter(chunk_size=10, chunk_overlap=50)._sizes(1200, 200) == (10, 9)

    def test_invalid_sizes(self) -> None:
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=0)
        with pytest.raises(ValueError):
            TextSplitter(chunk_overlap=-1)

    def test_modules_read_settings_overrides(self, chroma: ChromaVectorStore) -> None:
        qa = DocQA(collection="rag", store=chroma, CHUNKING="token", RETRIEVAL_MODE="hybrid")
        assert (qa.chunking, qa.retrieval_mode) == ("token", "hybrid")
        ss = SemanticSearch(collection="rag", store=chroma, RERANK=True, FETCH_K=7)
        assert (ss.rerank, ss.fetch_k) == (True, 7)
        # Constructor args win over settings.
        qa2 = DocQA(collection="rag", store=chroma, chunking="markdown", CHUNKING="token")
        assert qa2.chunking == "markdown"


# ===========================================================================
# Metadata propagation into citations
# ===========================================================================


class TestMetadataPropagation:
    def test_chunk_metadata_keeps_page_and_heading(self) -> None:
        chunk = Document(
            page_content="x",
            metadata={"source": "a.pdf", "file_type": "pdf", "page": 3, "heading_path": "A > B"},
        )
        meta = DocQA._chunk_metadata(chunk, 7)
        assert meta == {
            "source": "a.pdf",
            "file_type": "pdf",
            "chunk_index": 7,
            "page": 3,
            "heading_path": "A > B",
        }
        bare = DocQA._chunk_metadata(Document(page_content="y", metadata={"source": "b"}), 0)
        assert "page" not in bare and "heading_path" not in bare

    def test_pdf_page_flows_to_citation(self, chroma: ChromaVectorStore) -> None:
        qa = _docqa(chroma, retrieval_mode="hybrid")
        pages = [
            Document(
                page_content=f"Page {i} talks about topic{i}",
                metadata={"source": "manual.pdf", "file_type": "pdf", "page": i},
            )
            for i in range(3)
        ]
        qa._embed_and_store(pages, batch_size=10)
        qa._call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=[_completion("factual"), _completion("SUFFICIENT: 0.9"), _completion("a")]
        )
        resp = qa.ask("topic2")
        assert resp.citations[0].page == 2
        assert resp.citations[0].label() == "manual.pdf, page 2"

    def test_format_context_labels(self) -> None:
        chunk = {"text": "body", "source": "g.md", "page": None, "heading_path": "A > B"}
        assert chunk_label(chunk) == "g.md, A > B"
        assert format_context([chunk]) == "[source: g.md, A > B]\nbody"


# ===========================================================================
# Backend list_all_documents implementations (mocked clients)
# ===========================================================================


class TestListAllDocuments:
    def test_qdrant(self) -> None:
        from aiforge.core.vectorstore_qdrant import QdrantVectorStore

        store = object.__new__(QdrantVectorStore)
        store._collection_name = "c"
        point = MagicMock(id=1, payload={"_text": "hello", "source": "s"})
        store._client = MagicMock()
        store._client.scroll.return_value = ([point], None)
        assert store.list_all_documents() == [
            {"id": "1", "text": "hello", "metadata": {"source": "s"}}
        ]

    def test_pinecone(self) -> None:
        from aiforge.core.vectorstore_pinecone import PineconeVectorStore

        store = object.__new__(PineconeVectorStore)
        store._namespace = "ns"
        store._index = MagicMock()
        store._index.list.return_value = iter([["v1"]])
        vec = MagicMock(metadata={"_text": "hi", "source": "s"})
        store._index.fetch.return_value = MagicMock(vectors={"v1": vec})
        assert store.list_all_documents() == [
            {"id": "v1", "text": "hi", "metadata": {"source": "s"}}
        ]

    def test_pgvector(self) -> None:
        from aiforge.core.vectorstore_pgvector import PgVectorStore

        store = object.__new__(PgVectorStore)
        store._collection_name = "aiforge_c"
        cursor = MagicMock()
        cursor.fetchall.return_value = [("i1", "text", '{"source": "s"}')]
        store._conn = MagicMock()
        store._conn.cursor.return_value.__enter__.return_value = cursor
        assert store.list_all_documents() == [
            {"id": "i1", "text": "text", "metadata": {"source": "s"}}
        ]

    def test_default_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            NoListStore([]).list_all_documents()
