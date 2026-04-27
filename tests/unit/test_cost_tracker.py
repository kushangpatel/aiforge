"""Unit tests for CostTracker (infra/cost_tracker.py)."""

from __future__ import annotations

import threading

import pytest

from aiforge.infra.cost_tracker import CostTracker


class TestBasicAccumulation:
    def test_initial_state(self) -> None:
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_single_record(self) -> None:
        tracker = CostTracker()
        tracker.record(
            module="summarize",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.00125,
        )
        assert tracker.call_count == 1
        assert tracker.total_tokens == 150
        assert pytest.approx(tracker.total_cost) == 0.00125

    def test_multiple_records_accumulate(self) -> None:
        tracker = CostTracker()
        for _ in range(5):
            tracker.record(
                module="classify",
                model="gpt-4o-mini",
                input_tokens=20,
                output_tokens=5,
                cost=0.001,
            )
        assert tracker.call_count == 5
        assert pytest.approx(tracker.total_cost) == 0.005
        assert tracker.total_tokens == 125


class TestByModuleAndModel:
    def test_by_module_aggregation(self) -> None:
        tracker = CostTracker()
        tracker.record(
            module="summarize", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01
        )
        tracker.record(
            module="classify", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.02
        )
        tracker.record(
            module="summarize", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.03
        )

        by_module = tracker.by_module
        assert pytest.approx(by_module["summarize"]) == 0.04
        assert pytest.approx(by_module["classify"]) == 0.02

    def test_by_model_aggregation(self) -> None:
        tracker = CostTracker()
        tracker.record(module="a", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)
        tracker.record(
            module="b", model="gpt-4o-mini", input_tokens=10, output_tokens=5, cost=0.005
        )
        tracker.record(module="c", model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.01)

        by_model = tracker.by_model
        assert pytest.approx(by_model["gpt-4o"]) == 0.02
        assert pytest.approx(by_model["gpt-4o-mini"]) == 0.005


class TestSingleton:
    def test_get_session_returns_same_instance(self) -> None:
        a = CostTracker.get_session()
        b = CostTracker.get_session()
        assert a is b

    def test_reset_session_creates_new_instance(self) -> None:
        a = CostTracker.get_session()
        a.record(module="x", model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.001)
        CostTracker.reset_session()
        b = CostTracker.get_session()
        assert b.total_cost == 0.0
        assert a is not b


class TestThreadSafety:
    def test_concurrent_records(self) -> None:
        tracker = CostTracker()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    tracker.record(
                        module="test",
                        model="gpt-4o",
                        input_tokens=1,
                        output_tokens=1,
                        cost=0.001,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.call_count == 1000
        assert pytest.approx(tracker.total_cost) == 1.0


class TestWarnThreshold:
    def test_warn_threshold_logged(self, capsys: pytest.CaptureFixture) -> None:
        tracker = CostTracker(warn_threshold=0.50)
        tracker.record(
            module="summarize",
            model="gpt-4o",
            input_tokens=10000,
            output_tokens=5000,
            cost=1.00,  # exceeds 0.50 threshold
        )
        captured = capsys.readouterr()
        assert "High-cost" in captured.out or "High-cost" in captured.err


class TestReset:
    def test_instance_reset(self) -> None:
        tracker = CostTracker()
        tracker.record(module="x", model="gpt-4o", input_tokens=1, output_tokens=1, cost=0.01)
        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.call_count == 0


class TestSummary:
    def test_summary_shape(self) -> None:
        tracker = CostTracker()
        tracker.record(module="qa", model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.002)
        summary = tracker.summary()
        assert "total_cost_usd" in summary
        assert "total_tokens" in summary
        assert "call_count" in summary
        assert "by_module" in summary
        assert "by_model" in summary
