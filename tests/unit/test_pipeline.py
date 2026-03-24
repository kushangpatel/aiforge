"""Unit tests for Pipeline and Router (pipeline/)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aiforge.core.types import BaseModule, ModuleResponse, PipelineError
from aiforge.pipeline.pipeline import Pipeline
from aiforge.pipeline.router import Router
from aiforge.pipeline.schemas import PipelineResult


# ---------------------------------------------------------------------------
# Helpers: minimal module stubs
# ---------------------------------------------------------------------------

class _TextResponse(ModuleResponse):
    text: str = ""

    def to_text(self) -> str:
        return self.text


class _LabelResponse(ModuleResponse):
    label: str = ""
    text: str = ""

    def to_text(self) -> str:
        return self.text


class _EchoModule(BaseModule):
    """Returns the input text as a TextResponse."""

    def __init__(self, suffix: str = "") -> None:
        # Skip real BaseModule init (no settings/provider needed)
        self.suffix = suffix

    def run(self, text: str, **kwargs: Any) -> _TextResponse:  # type: ignore[override]
        return _TextResponse(text=text + self.suffix, cost=0.001)

    async def arun(self, text: str, **kwargs: Any) -> _TextResponse:  # type: ignore[override]
        return self.run(text)


class _ClassifyModule(BaseModule):
    """Always returns a fixed label."""

    def __init__(self, label: str) -> None:
        self.fixed_label = label

    def run(self, text: str, **kwargs: Any) -> _LabelResponse:  # type: ignore[override]
        return _LabelResponse(label=self.fixed_label, text=text, cost=0.002)

    async def arun(self, text: str, **kwargs: Any) -> _LabelResponse:  # type: ignore[override]
        return self.run(text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialPipeline:
    def test_single_step(self) -> None:
        pipe = Pipeline([("echo", _EchoModule())])
        result = pipe.run("hello")
        assert isinstance(result, PipelineResult)
        assert "echo" in result.steps
        assert result.steps["echo"].to_text() == "hello"

    def test_two_steps_chained(self) -> None:
        pipe = Pipeline([
            ("step1", _EchoModule(suffix=" world")),
            ("step2", _EchoModule(suffix="!"))  ,
        ])
        result = pipe.run("hello")
        assert result.steps["step2"].to_text() == "hello world!"

    def test_cost_aggregated(self) -> None:
        pipe = Pipeline([
            ("a", _EchoModule()),
            ("b", _EchoModule()),
        ])
        result = pipe.run("text")
        assert pytest.approx(result.total_cost) == 0.002  # 2 × 0.001

    def test_result_last_property(self) -> None:
        pipe = Pipeline([("only", _EchoModule())])
        result = pipe.run("hi")
        assert result.last is not None
        assert result.last.to_text() == "hi"

    def test_empty_pipeline(self) -> None:
        pipe = Pipeline([])
        result = pipe.run("input")
        assert result.steps == {}
        assert result.total_cost == 0.0


class TestConditionalRouter:
    def test_router_routes_to_correct_branch(self) -> None:
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="support")),
            ("route", Router({
                "support": _EchoModule(suffix="-support"),
                "sales":   _EchoModule(suffix="-sales"),
            })),
        ])
        result = pipe.run("help me")
        assert result.steps["route"].to_text() == "help me-support"

    def test_router_none_branch_halts(self) -> None:
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="spam")),
            ("route", Router({"spam": None, "real": _EchoModule()})),
        ])
        result = pipe.run("buy now")
        assert result.halted is True
        assert result.halted_at == "route"
        assert "route" not in result.steps

    def test_router_unknown_label_raises(self) -> None:
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="unknown_label")),
            ("route", Router({"a": _EchoModule(), "b": _EchoModule()})),
        ])
        with pytest.raises(KeyError, match="unknown_label"):
            pipe.run("text")

    def test_router_default_branch(self) -> None:
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="other")),
            ("route", Router(
                {"known": _EchoModule(suffix="-known")},
                default=_EchoModule(suffix="-default"),
            )),
        ])
        result = pipe.run("text")
        assert result.steps["route"].to_text() == "text-default"

    def test_router_without_previous_label_raises(self) -> None:
        """Router after a non-label step raises PipelineError."""
        pipe = Pipeline([
            ("echo", _EchoModule()),
            ("route", Router({"a": _EchoModule()})),
        ])
        with pytest.raises(PipelineError):
            pipe.run("text")


class TestAsyncPipeline:
    @pytest.mark.asyncio
    async def test_async_run(self) -> None:
        pipe = Pipeline([("step", _EchoModule(suffix="-async"))])
        result = await pipe.arun("hello")
        assert result.steps["step"].to_text() == "hello-async"

    @pytest.mark.asyncio
    async def test_async_two_steps(self) -> None:
        pipe = Pipeline([
            ("a", _EchoModule(suffix=" A")),
            ("b", _EchoModule(suffix=" B")),
        ])
        result = await pipe.arun("start")
        assert result.steps["b"].to_text() == "start A B"

    @pytest.mark.asyncio
    async def test_async_router_routes_correctly(self) -> None:
        """Router.resolve_async() is used in arun() -- must not block event loop."""
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="sales")),
            ("route", Router({
                "support": _EchoModule(suffix="-support"),
                "sales":   _EchoModule(suffix="-sales"),
            })),
        ])
        result = await pipe.arun("buy something")
        assert result.steps["route"].to_text() == "buy something-sales"

    @pytest.mark.asyncio
    async def test_async_router_halt(self) -> None:
        """Router with None branch halts async pipeline."""
        pipe = Pipeline([
            ("classify", _ClassifyModule(label="spam")),
            ("route", Router({"spam": None, "real": _EchoModule()})),
        ])
        result = await pipe.arun("junk")
        assert result.halted is True
        assert "route" not in result.steps


class TestHaltOnFlagFalse:
    def test_halt_on_flag_false_continues_after_none_branch(self) -> None:
        """When halt_on_flag=False, pipeline continues past a None router branch."""
        pipe = Pipeline(
            [
                ("classify", _ClassifyModule(label="spam")),
                ("route", Router({"spam": None, "real": _EchoModule()})),
                ("after", _EchoModule(suffix="-after")),
            ],
            halt_on_flag=False,
        )
        result = pipe.run("text")
        # Router returned None so no "route" step result, but "after" still ran
        assert result.halted is False
        assert "after" in result.steps

    @pytest.mark.asyncio
    async def test_async_halt_on_flag_false(self) -> None:
        pipe = Pipeline(
            [
                ("classify", _ClassifyModule(label="spam")),
                ("route", Router({"spam": None})),
                ("after", _EchoModule()),
            ],
            halt_on_flag=False,
        )
        result = await pipe.arun("anything")
        assert result.halted is False
        assert "after" in result.steps


class TestPipelineResultSerialization:
    def test_model_dump_json_roundtrip(self) -> None:
        """PipelineResult.model_dump_json() must not raise for subclass responses."""
        pipe = Pipeline([("step", _EchoModule())])
        result = pipe.run("hello")
        json_str = result.model_dump_json()
        assert "hello" in json_str
        assert "step" in json_str


# ===========================================================================
# Pipeline route
# ===========================================================================


class TestPipelineRoute:
    def test_empty_steps_rejected(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from aiforge.server.routes.pipeline import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/pipeline", json={"steps": [], "text": "hello"})
        assert resp.status_code == 422

    def test_unknown_module_rejected(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from aiforge.server.routes.pipeline import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/pipeline", json={
            "steps": [{"name": "s1", "module": "nonexistent_xyz"}],
            "text": "hello",
        })
        assert resp.status_code == 422
