"""Unit tests for the DocQA (RAG) module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.modules.qa.agent import QAAgentState, build_qa_graph
from aiforge.modules.qa.module import DocQA, _estimate_tokens
from aiforge.modules.qa.prompts import (
    CLASSIFY_PROMPT,
    REASON_PROMPT,
    RESPOND_PROMPT,
)
from aiforge.modules.qa.schemas import IngestResponse, QAResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qa(
    collection: str = "test",
    retrieval_k: int = 6,
    max_iterations: int = 3,
    store: Any = None,
) -> DocQA:
    """Build a DocQA instance bypassing BaseModule provider init."""
    qa = object.__new__(DocQA)
    qa.collection = collection
    qa.persist_dir = "./test_data"
    qa.chunk_size = None
    qa.chunk_overlap = None
    qa.retrieval_k = retrieval_k
    qa.max_iterations = max_iterations
    qa._store = store or MagicMock()

    import structlog

    qa.logger = structlog.get_logger()
    qa._provider = None
    qa._embedding_provider = MagicMock()
    qa._embedding_provider.embed_query.return_value = [0.1] * 384
    qa._embedding_provider.model_name = "text-embedding-3-small"
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.MODEL = "gpt-4o-mini"
    qa.settings = settings
    from aiforge.infra.cost_tracker import CostTracker

    qa.cost_tracker = CostTracker.get_session()
    return qa


def _fake_llm(text: str = "test response", inp: int = 50, out: int = 30) -> MagicMock:
    """Create a fake CompletionResult."""
    r = MagicMock()
    r.text = text
    r.input_tokens = inp
    r.output_tokens = out
    r.model = "gpt-4o-mini"
    return r


def _sample_hits(n: int = 2) -> list[dict[str, Any]]:
    """Return sample vectorstore query hits."""
    return [
        {
            "id": f"chunk-{i}",
            "text": f"Sample chunk text {i}",
            "metadata": {"source": f"src/file{i}.py", "chunk_index": i},
            "distance": 0.1 * (i + 1),
        }
        for i in range(n)
    ]


def _initial_state(
    query: str = "How does auth work?",
    max_iterations: int = 3,
) -> dict[str, Any]:
    return {
        "query": query,
        "repo_filter": None,
        "intent": "",
        "retrieved_chunks": [],
        "current_query": query,
        "iteration": 0,
        "max_iterations": max_iterations,
        "trace": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "decision": "",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "model": "",
    }


# ===========================================================================
# IngestResponse schema
# ===========================================================================


class TestIngestResponse:
    def test_defaults(self) -> None:
        r = IngestResponse()
        assert r.chunks_indexed == 0
        assert r.files_processed == 0
        assert r.languages == {}
        assert r.skipped_files == []
        assert r.collection == ""
        assert r.cost == 0.0

    def test_to_text(self) -> None:
        r = IngestResponse(files_processed=5, collection="proj", chunks_indexed=42)
        assert "5 files" in r.to_text()
        assert "'proj'" in r.to_text()
        assert "42 chunks" in r.to_text()

    def test_inherits_module_response(self) -> None:
        r = IngestResponse(cost=0.01, input_tokens=100, model="test")
        assert r.cost == 0.01
        assert r.input_tokens == 100
        assert r.model == "test"

    def test_languages_field(self) -> None:
        r = IngestResponse(languages={"py": 10, "md": 3})
        assert r.languages == {"py": 10, "md": 3}


# ===========================================================================
# QAResponse schema
# ===========================================================================


class TestQAResponse:
    def test_defaults(self) -> None:
        r = QAResponse()
        assert r.text == ""
        assert r.sources == []
        assert r.confidence == 0.0
        assert r.iterations == 0
        assert r.trace is None

    def test_to_text(self) -> None:
        r = QAResponse(text="Auth uses JWT.")
        assert r.to_text() == "Auth uses JWT."

    def test_with_trace(self) -> None:
        r = QAResponse(trace=[{"step": "classify"}])
        assert r.trace is not None
        assert len(r.trace) == 1

    def test_without_trace(self) -> None:
        r = QAResponse(text="answer")
        assert r.trace is None

    def test_confidence_bounds(self) -> None:
        r = QAResponse(confidence=0.95)
        assert r.confidence == 0.95

    def test_sources_list(self) -> None:
        r = QAResponse(sources=["auth.py", "config.py"])
        assert len(r.sources) == 2

    def test_inherits_module_response(self) -> None:
        r = QAResponse(cost=0.05, input_tokens=200, output_tokens=100, model="gpt-4o")
        assert r.cost == 0.05
        assert r.input_tokens == 200


# ===========================================================================
# Prompts
# ===========================================================================


class TestPrompts:
    def test_classify_prompt_format(self) -> None:
        result = CLASSIFY_PROMPT.format(query="How does auth work?")
        assert "How does auth work?" in result
        assert "factual" in result

    def test_reason_prompt_format(self) -> None:
        result = REASON_PROMPT.format(
            query="test", intent="factual", iteration=1, max_iterations=3, context="ctx"
        )
        assert "test" in result
        assert "factual" in result
        assert "1/3" in result

    def test_respond_prompt_format(self) -> None:
        result = RESPOND_PROMPT.format(query="test", context="some context")
        assert "test" in result
        assert "some context" in result


# ===========================================================================
# Agent node tests — build graph with mock callables and test each step
# ===========================================================================


class TestAgentGraphSingleIteration:
    """Full graph execution with SUFFICIENT on first try (3 LLM calls)."""

    def test_basic_flow(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),  # classify
                _fake_llm("SUFFICIENT: 0.88"),  # reason
                _fake_llm("Auth uses JWT tokens."),  # respond
            ]
        )
        llm_fn = lambda prompt: next(calls)
        retrieve_fn = MagicMock(return_value=_sample_hits(2))

        graph = build_qa_graph(llm_fn=llm_fn, retrieve_fn=retrieve_fn, retrieval_k=6)
        result = graph.invoke(_initial_state())

        assert result["intent"] == "factual"
        assert result["answer"] == "Auth uses JWT tokens."
        assert result["confidence"] == pytest.approx(0.88)
        assert result["iteration"] == 1
        assert len(result["sources"]) == 2

    def test_trace_has_all_steps(self) -> None:
        calls = iter(
            [
                _fake_llm("conceptual"),
                _fake_llm("SUFFICIENT: 0.9"),
                _fake_llm("Explanation here."),
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        steps = [t["step"] for t in result["trace"]]
        assert steps == ["classify", "retrieve", "reason", "respond"]

    def test_token_accumulation(self) -> None:
        calls = iter([_fake_llm(inp=10, out=5), _fake_llm(inp=20, out=10), _fake_llm(inp=30, out=15)])
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        # Provide default text for LLM responses
        calls = iter([
            _fake_llm("factual", 10, 5),
            _fake_llm("SUFFICIENT: 0.9", 20, 10),
            _fake_llm("answer", 30, 15),
        ])
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["total_input_tokens"] == 60  # 10 + 20 + 30
        assert result["total_output_tokens"] == 30  # 5 + 10 + 15


class TestAgentGraphMultiIteration:
    """Graph execution with INSUFFICIENT → refined retrieval → SUFFICIENT."""

    def test_two_iterations(self) -> None:
        calls = iter(
            [
                _fake_llm("debugging"),  # classify
                _fake_llm("INSUFFICIENT: auth middleware error handling"),  # reason 1
                _fake_llm("SUFFICIENT: 0.82"),  # reason 2
                _fake_llm("The error is in middleware.py line 42."),  # respond
            ]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits(1))

        graph = build_qa_graph(
            llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn, retrieval_k=4,
        )
        result = graph.invoke(_initial_state(max_iterations=3))

        assert result["iteration"] == 2
        assert result["confidence"] == pytest.approx(0.82)
        assert result["answer"] == "The error is in middleware.py line 42."
        # retrieve_fn called twice (once per iteration)
        assert retrieve_fn.call_count == 2

    def test_max_iterations_forces_respond(self) -> None:
        """Always insufficient — forced respond at max_iterations."""
        calls = iter(
            [
                _fake_llm("factual"),  # classify
                _fake_llm("INSUFFICIENT: better query"),  # reason 1
                _fake_llm("INSUFFICIENT: even better"),  # reason 2
                _fake_llm("Best guess answer."),  # respond (forced)
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
            retrieval_k=4,
        )
        result = graph.invoke(_initial_state(max_iterations=2))
        assert result["iteration"] == 2
        assert result["answer"] == "Best guess answer."
        # Fallback confidence assigned when forced to respond without SUFFICIENT
        assert result["confidence"] == pytest.approx(0.3)

    def test_query_refinement_updates_current_query(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("INSUFFICIENT: refined auth question"),
                _fake_llm("SUFFICIENT: 0.75"),
                _fake_llm("answer"),
            ]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn,
        )
        result = graph.invoke(_initial_state())
        # Second retrieve call should use refined query
        second_call_args = retrieve_fn.call_args_list[1]
        assert second_call_args[0][0] == "refined auth question"

    def test_chunks_accumulated_across_iterations(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("INSUFFICIENT: more details"),
                _fake_llm("SUFFICIENT: 0.9"),
                _fake_llm("answer"),
            ]
        )
        # Different hits per call
        retrieve_fn = MagicMock(
            side_effect=[_sample_hits(2), [{"id": "c3", "text": "new chunk", "metadata": {"source": "new.py"}, "distance": 0.05}]]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn,
        )
        result = graph.invoke(_initial_state())
        # 2 chunks from first retrieval + 1 from second
        assert len(result["retrieved_chunks"]) == 3
        assert "new.py" in result["sources"]


class TestAgentRepoFilter:
    def test_passes_filter_to_retrieve(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: 0.9"), _fake_llm("answer")]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn)

        state = _initial_state()
        state["repo_filter"] = "dal-ingest"
        graph.invoke(state)

        _, kwargs = retrieve_fn.call_args
        assert kwargs["filter"] == {"source": {"$contains": "dal-ingest"}}

    def test_no_filter_when_none(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: 0.9"), _fake_llm("answer")]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn)
        graph.invoke(_initial_state())
        _, kwargs = retrieve_fn.call_args
        assert kwargs["filter"] is None


class TestAgentConfidenceParsing:
    def test_parses_valid_confidence(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: 0.92"), _fake_llm("a")]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["confidence"] == pytest.approx(0.92)

    def test_defaults_on_bad_confidence(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: not-a-number"), _fake_llm("a")]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["confidence"] == pytest.approx(0.7)

    def test_clamps_confidence(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: 1.5"), _fake_llm("a")]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["confidence"] == 1.0

    def test_clamps_negative(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: -0.5"), _fake_llm("a")]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["confidence"] == 0.0


class TestAgentEmptyRetrieval:
    def test_no_hits_still_completes(self) -> None:
        calls = iter(
            [_fake_llm("factual"), _fake_llm("SUFFICIENT: 0.3"), _fake_llm("No relevant docs found.")]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=[]),
        )
        result = graph.invoke(_initial_state())
        assert result["answer"] == "No relevant docs found."
        assert result["retrieved_chunks"] == []


# ===========================================================================
# DocQA constructor
# ===========================================================================


class TestDocQAConstructor:
    def test_defaults(self) -> None:
        qa = _qa()
        assert qa.collection == "test"
        assert qa.retrieval_k == 6
        assert qa.max_iterations == 3

    def test_custom_params(self) -> None:
        qa = _qa(collection="my-proj", retrieval_k=10, max_iterations=5)
        assert qa.collection == "my-proj"
        assert qa.retrieval_k == 10
        assert qa.max_iterations == 5

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection must not be empty"):
            DocQA(collection="")

    def test_whitespace_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="collection must not be empty"):
            DocQA(collection="   ")

    def test_collection_stripped(self) -> None:
        qa = _qa(collection="  myc  ")
        # _qa bypasses validation, test the real constructor
        with patch("aiforge.modules.qa.module.BaseModule.__init__"):
            obj = DocQA.__new__(DocQA)
            DocQA.__init__(obj, collection="  myc  ", store=MagicMock())
            assert obj.collection == "myc"

    def test_retrieval_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="retrieval_k must be >= 1"):
            DocQA(collection="test", retrieval_k=0)

    def test_max_iterations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            DocQA(collection="test", max_iterations=0)


# ===========================================================================
# DocQA.ingest
# ===========================================================================


class TestDocQAIngest:
    def test_ingest_directory(self, tmp_path: Any) -> None:
        (tmp_path / "hello.py").write_text("def hello(): pass\n")
        (tmp_path / "readme.md").write_text("# Hello World\n\nSome text here.\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path))

        assert result.files_processed == 2
        assert result.chunks_indexed >= 1
        assert result.collection == "test"
        assert store.add_documents.called

    def test_ingest_with_extension_filter(self, tmp_path: Any) -> None:
        (tmp_path / "code.py").write_text("x = 1\n")
        (tmp_path / "notes.md").write_text("# Notes\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path), file_extensions=[".py"])

        assert result.files_processed == 1
        assert "py" in result.languages

    def test_ingest_empty_directory(self, tmp_path: Any) -> None:
        qa = _qa()
        result = qa.ingest(str(tmp_path))
        assert result.chunks_indexed == 0
        assert result.files_processed == 0

    def test_ingest_skips_binary(self, tmp_path: Any) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "code.py").write_text("x = 42\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path))
        assert result.files_processed == 1

    def test_ingest_cost_tracking(self, tmp_path: Any) -> None:
        (tmp_path / "code.py").write_text("def foo(): return 42\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path))
        assert result.input_tokens > 0
        assert result.model == "text-embedding-3-small"

    def test_ingest_batch_embedding(self, tmp_path: Any) -> None:
        # Create enough files to trigger multiple batches
        for i in range(3):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path), batch_size=2)
        assert result.chunks_indexed >= 3
        # Embedding called at least 2 times (3 chunks, batch_size=2)
        assert qa._call_embedding.call_count >= 2

    def test_ingest_languages(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text("pass\n")
        (tmp_path / "b.js").write_text("// js\n")
        (tmp_path / "c.md").write_text("# md\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path))
        assert result.files_processed == 3
        assert "py" in result.languages
        assert "js" in result.languages

    def test_ingest_returns_ingest_response(self, tmp_path: Any) -> None:
        (tmp_path / "test.py").write_text("x = 1\n")
        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path))
        assert isinstance(result, IngestResponse)
        assert result.latency_ms > 0


class TestDocQAIngestAsync:
    @pytest.mark.asyncio
    async def test_aingest_basic(self, tmp_path: Any) -> None:
        (tmp_path / "hello.py").write_text("def hello(): pass\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._acall_embedding = AsyncMock(return_value=[[0.1] * 384])

        result = await qa.aingest(str(tmp_path))
        assert result.files_processed == 1
        assert result.chunks_indexed >= 1

    @pytest.mark.asyncio
    async def test_aingest_empty(self, tmp_path: Any) -> None:
        qa = _qa()
        result = await qa.aingest(str(tmp_path))
        assert result.chunks_indexed == 0


# ===========================================================================
# DocQA.ask
# ===========================================================================


class TestDocQAAsk:
    def _setup_ask_mocks(self, qa: DocQA, llm_responses: list) -> None:
        """Set up _call_llm with side_effect and embed_query mock."""
        qa._call_llm = MagicMock(side_effect=llm_responses)
        qa._embedding_provider.embed_query.return_value = [0.1] * 384

    def test_ask_returns_answer(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.85"),
            _fake_llm("The auth module uses JWT."),
        ])

        result = qa.ask("How does auth work?")
        assert result.text == "The auth module uses JWT."
        assert isinstance(result, QAResponse)

    def test_ask_confidence(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.92"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?")
        assert result.confidence == pytest.approx(0.92)

    def test_ask_sources(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits(2)
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?")
        assert "src/file0.py" in result.sources
        assert "src/file1.py" in result.sources

    def test_ask_with_trace(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?", include_trace=True)
        assert result.trace is not None
        steps = [t["step"] for t in result.trace]
        assert "classify" in steps
        assert "retrieve" in steps
        assert "reason" in steps
        assert "respond" in steps

    def test_ask_without_trace(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?", include_trace=False)
        assert result.trace is None

    def test_ask_with_repo_filter(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?", filter_repo="dal-ingest")
        assert result.text == "answer"
        # Verify the store was queried with a filter
        _, kwargs = store.query.call_args
        assert kwargs["filter"] == {"source": {"$contains": "dal-ingest"}}

    def test_ask_empty_collection_raises(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 0}
        qa = _qa(store=store)

        from aiforge.core.types import AIForgeError
        with pytest.raises(AIForgeError, match="No documents ingested"):
            qa.ask("test?")

    def test_ask_cost_tracking(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual", 10, 5),
            _fake_llm("SUFFICIENT: 0.9", 20, 10),
            _fake_llm("answer", 30, 15),
        ])

        result = qa.ask("test?")
        assert result.input_tokens == 60
        assert result.output_tokens == 30

    def test_ask_iterations(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store, max_iterations=2)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("INSUFFICIENT: better query"),
            _fake_llm("SUFFICIENT: 0.8"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?")
        assert result.iterations == 2

    def test_ask_max_iterations_respected(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store, max_iterations=2)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("INSUFFICIENT: q1"),
            _fake_llm("INSUFFICIENT: q2"),
            _fake_llm("forced answer"),
        ])

        result = qa.ask("test?")
        assert result.iterations <= 2

    def test_ask_latency(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?")
        assert result.latency_ms > 0

    def test_ask_trace_id(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        self._setup_ask_mocks(qa, [
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("answer"),
        ])

        result = qa.ask("test?")
        assert len(result.trace_id) == 16


class TestDocQAAskAsync:
    @pytest.mark.asyncio
    async def test_aask_basic(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        qa._call_llm = MagicMock(side_effect=[
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("async answer"),
        ])

        result = await qa.aask("test?")
        assert result.text == "async answer"
        assert isinstance(result, QAResponse)

    @pytest.mark.asyncio
    async def test_aask_with_trace(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        qa._call_llm = MagicMock(side_effect=[
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.85"),
            _fake_llm("answer"),
        ])

        result = await qa.aask("test?", include_trace=True)
        assert result.trace is not None
        assert len(result.trace) >= 4

    @pytest.mark.asyncio
    async def test_aask_empty_collection(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 0}
        qa = _qa(store=store)

        from aiforge.core.types import AIForgeError
        with pytest.raises(AIForgeError, match="No documents ingested"):
            await qa.aask("test?")


# ===========================================================================
# DocQA collection management
# ===========================================================================


class TestDocQACollections:
    def test_list_collections(self) -> None:
        store = MagicMock()
        store.list_collections.return_value = ["proj-a", "proj-b"]
        qa = _qa(store=store)
        assert qa.list_collections() == ["proj-a", "proj-b"]

    def test_collection_stats(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"name": "test", "chunks": 42}
        qa = _qa(store=store)
        stats = qa.collection_stats()
        assert stats["chunks"] == 42

    def test_collection_stats_custom(self) -> None:
        store = MagicMock()
        qa = _qa(store=store)
        qa.collection_stats("other")
        store.collection_stats.assert_called_with("other")

    def test_delete_collection(self) -> None:
        store = MagicMock()
        qa = _qa(store=store)
        qa.delete_collection()
        store.delete_collection.assert_called_with("test")

    def test_delete_collection_custom(self) -> None:
        store = MagicMock()
        qa = _qa(store=store)
        qa.delete_collection("other")
        store.delete_collection.assert_called_with("other")


# ===========================================================================
# Pipeline interface
# ===========================================================================


class TestDocQAPipeline:
    def test_run_delegates_to_ask(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        qa._call_llm = MagicMock(side_effect=[
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("pipeline answer"),
        ])

        result = qa.run("test?")
        assert result.text == "pipeline answer"

    @pytest.mark.asyncio
    async def test_arun_delegates_to_aask(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)
        qa._call_llm = MagicMock(side_effect=[
            _fake_llm("factual"),
            _fake_llm("SUFFICIENT: 0.9"),
            _fake_llm("async pipeline"),
        ])

        result = await qa.arun("test?")
        assert result.text == "async pipeline"


# ===========================================================================
# _estimate_tokens helper
# ===========================================================================


class TestEstimateTokens:
    def test_basic(self) -> None:
        assert _estimate_tokens(["hello world"]) >= 1

    def test_empty_list(self) -> None:
        assert _estimate_tokens([]) == 1

    def test_scales_with_text(self) -> None:
        short = _estimate_tokens(["hello"])
        long = _estimate_tokens(["hello " * 100])
        assert long > short


# ===========================================================================
# Server routes
# ===========================================================================


class TestRoutes:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from aiforge.server.routes.qa import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_ingest_endpoint(self, client, tmp_path) -> None:
        (tmp_path / "test.py").write_text("x = 1\n")
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            mock_qa.aingest = AsyncMock(return_value=IngestResponse(
                chunks_indexed=5,
                files_processed=1,
                languages={"py": 1},
                skipped_files=[],
                collection="test",
                cost=0.001,
            ))
            MockQA.return_value = mock_qa

            resp = client.post("/qa/ingest", json={
                "source": str(tmp_path),
                "collection": "test",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_indexed"] == 5
        assert data["collection"] == "test"

    def test_ask_endpoint(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            mock_qa.aask = AsyncMock(return_value=QAResponse(
                text="JWT is used.",
                sources=["auth.py"],
                confidence=0.9,
                iterations=1,
                cost=0.01,
                model="gpt-4o-mini",
                trace_id="abc123",
            ))
            MockQA.return_value = mock_qa

            resp = client.post("/qa/ask", json={
                "query": "How does auth work?",
                "collection": "test",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "JWT is used."
        assert data["confidence"] == 0.9

    def test_ask_with_trace(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            mock_qa.aask = AsyncMock(return_value=QAResponse(
                text="answer",
                trace=[{"step": "classify"}],
                confidence=0.8,
                iterations=1,
            ))
            MockQA.return_value = mock_qa

            resp = client.post("/qa/ask", json={
                "query": "test?",
                "include_trace": True,
            })

        data = resp.json()
        assert data["trace"] is not None

    def test_ask_empty_collection(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            from aiforge.core.types import AIForgeError
            mock_qa = MagicMock()
            mock_qa.aask = AsyncMock(side_effect=AIForgeError("No documents ingested"))
            MockQA.return_value = mock_qa

            resp = client.post("/qa/ask", json={"query": "test?"})
        assert resp.status_code == 422

    def test_collections_endpoint(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            mock_qa.list_collections.return_value = ["proj-a", "proj-b"]
            MockQA.return_value = mock_qa

            resp = client.get("/qa/collections")

        assert resp.status_code == 200
        assert resp.json()["collections"] == ["proj-a", "proj-b"]

    def test_collection_stats_endpoint(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            mock_qa.collection_stats.return_value = {"chunks": 42, "persist_dir": "./data"}
            MockQA.return_value = mock_qa

            resp = client.get("/qa/collections/proj-a/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "proj-a"
        assert data["chunks"] == 42

    def test_delete_collection_endpoint(self, client) -> None:
        with patch("aiforge.modules.qa.module.DocQA") as MockQA:
            mock_qa = MagicMock()
            MockQA.return_value = mock_qa

            resp = client.delete("/qa/collections/proj-a")

        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert resp.json()["collection"] == "proj-a"
        mock_qa.delete_collection.assert_called_with("proj-a")

    def test_ask_validation_error(self, client) -> None:
        resp = client.post("/qa/ask", json={"query": ""})
        assert resp.status_code == 422


# ===========================================================================
# run_batch with DocQA
# ===========================================================================


class TestDocQABatch:
    @pytest.mark.asyncio
    async def test_run_batch_partial_failure(self) -> None:
        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = _sample_hits()
        qa = _qa(store=store)

        call_count = [0]

        def fake_call_llm(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 1:
                return _fake_llm("factual")
            elif call_count[0] % 3 == 2:
                return _fake_llm("SUFFICIENT: 0.9")
            else:
                return _fake_llm("answer")

        qa._call_llm = fake_call_llm
        qa._acall_llm = AsyncMock(side_effect=fake_call_llm)

        # run_batch uses arun internally
        result = qa.run_batch(["q1", "q2"])
        assert len(result.results) + len(result.failed) == 2


# ===========================================================================
# Gap-audit: additional edge-case tests (T1–T5)
# ===========================================================================


class TestDocQAIngestFileList:
    """T1: Ingest a list of file paths."""

    def test_ingest_file_list(self, tmp_path: Any) -> None:
        f1 = tmp_path / "one.py"
        f2 = tmp_path / "two.md"
        f1.write_text("x = 1\n")
        f2.write_text("# Two\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest([str(f1), str(f2)])
        assert result.files_processed == 2
        assert result.chunks_indexed >= 2


class TestDocQAIngestInvalidSource:
    """T2: Invalid source raises FileNotFoundError."""

    def test_ingest_nonexistent_path(self) -> None:
        qa = _qa()
        with pytest.raises(FileNotFoundError):
            qa.ingest("/nonexistent/path/that/does/not/exist")


class TestAgentInsufficientNoColon:
    """T3: INSUFFICIENT without colon — current_query stays unchanged."""

    def test_insufficient_no_colon_reuses_query(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),  # classify
                _fake_llm("INSUFFICIENT"),  # reason 1 — no colon, no refined query
                _fake_llm("SUFFICIENT: 0.6"),  # reason 2
                _fake_llm("answer"),  # respond
            ]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls), retrieve_fn=retrieve_fn,
        )
        result = graph.invoke(_initial_state(query="original q", max_iterations=3))
        # Second retrieve call should reuse original query (not updated)
        second_call = retrieve_fn.call_args_list[1]
        assert second_call[0][0] == "original q"
        assert result["answer"] == "answer"


class TestDocQAAskQueryValidation:
    """T4: Empty/whitespace query raises ValueError."""

    def test_ask_empty_query(self) -> None:
        qa = _qa()
        with pytest.raises(ValueError, match="query must not be empty"):
            qa.ask("")

    def test_ask_whitespace_query(self) -> None:
        qa = _qa()
        with pytest.raises(ValueError, match="query must not be empty"):
            qa.ask("   ")

    @pytest.mark.asyncio
    async def test_aask_empty_query(self) -> None:
        qa = _qa()
        with pytest.raises(ValueError, match="query must not be empty"):
            await qa.aask("")


class TestDocQAIngestSkippedFiles:
    """T5: Extension filter causes skipped_files to be populated."""

    def test_extension_filter_populates_skipped_files(self, tmp_path: Any) -> None:
        (tmp_path / "code.py").write_text("x = 1\n")
        (tmp_path / "notes.md").write_text("# Notes\n")

        store = MagicMock()
        qa = _qa(store=store)
        qa._call_embedding = MagicMock(return_value=[[0.1] * 384])

        result = qa.ingest(str(tmp_path), file_extensions=[".py"])
        assert result.files_processed == 1
        # DocLoader puts non-whitelisted files in skipped_files
        assert any("notes.md" in s for s in result.skipped_files)


# ===========================================================================
# Reason-node parsing robustness (LLM formatting edge cases)
# ===========================================================================


class TestReasonNodeMarkdownBullet:
    """LLM returns '- SUFFICIENT: 0.8' with markdown bullet prefix."""

    def test_bullet_sufficient_parsed(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("- SUFFICIENT: 0.8"),
                _fake_llm("answer"),
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["decision"] == "sufficient"
        assert result["confidence"] == pytest.approx(0.8)

    def test_asterisk_bullet_sufficient(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("* SUFFICIENT: 0.75"),
                _fake_llm("answer"),
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["decision"] == "sufficient"
        assert result["confidence"] == pytest.approx(0.75)


class TestReasonNodeBracketedConfidence:
    """LLM returns 'SUFFICIENT: [confidence score 0.8]' echoing prompt format."""

    def test_bracketed_score_extracted(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("SUFFICIENT: [confidence score 0.85]"),
                _fake_llm("answer"),
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["decision"] == "sufficient"
        assert result["confidence"] == pytest.approx(0.85)

    def test_bullet_and_brackets(self) -> None:
        """Exact format from the bug report: '- SUFFICIENT: [confidence score 0.8]'."""
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("- SUFFICIENT: [confidence score 0.8]"),
                _fake_llm("answer"),
            ]
        )
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=MagicMock(return_value=_sample_hits()),
        )
        result = graph.invoke(_initial_state())
        assert result["decision"] == "sufficient"
        assert result["confidence"] == pytest.approx(0.8)


class TestReasonNodeInsufficientQuotedQuery:
    """LLM returns INSUFFICIENT with quoted refined query."""

    def test_quoted_query_stripped(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm('INSUFFICIENT: "How did Romeo and Juliet meet?"'),
                _fake_llm("SUFFICIENT: 0.9"),
                _fake_llm("answer"),
            ]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=retrieve_fn,
        )
        result = graph.invoke(_initial_state(max_iterations=3))
        # The refined query should have quotes stripped
        second_retrieve = retrieve_fn.call_args_list[1]
        assert second_retrieve[0][0] == "How did Romeo and Juliet meet?"

    def test_bullet_insufficient_with_query(self) -> None:
        calls = iter(
            [
                _fake_llm("factual"),
                _fake_llm("- INSUFFICIENT: a better query"),
                _fake_llm("SUFFICIENT: 0.9"),
                _fake_llm("answer"),
            ]
        )
        retrieve_fn = MagicMock(return_value=_sample_hits())
        graph = build_qa_graph(
            llm_fn=lambda p: next(calls),
            retrieve_fn=retrieve_fn,
        )
        result = graph.invoke(_initial_state(max_iterations=3))
        second_retrieve = retrieve_fn.call_args_list[1]
        assert second_retrieve[0][0] == "a better query"
