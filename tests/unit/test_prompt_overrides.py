"""Unit tests for the custom prompt override feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.core.prompt_override import resolve_prompt


# ---------------------------------------------------------------------------
# resolve_prompt() tests
# ---------------------------------------------------------------------------


class TestResolvePrompt:
    def test_call_override_wins(self) -> None:
        result = resolve_prompt(
            call_override="call",
            instance_override="instance",
            config_override="config",
            default="default",
        )
        assert result == "call"

    def test_instance_override_wins_over_config(self) -> None:
        result = resolve_prompt(
            call_override=None,
            instance_override="instance",
            config_override="config",
            default="default",
        )
        assert result == "instance"

    def test_config_override_wins_over_default(self) -> None:
        result = resolve_prompt(
            call_override=None,
            instance_override=None,
            config_override="config",
            default="default",
        )
        assert result == "config"

    def test_falls_back_to_default(self) -> None:
        result = resolve_prompt(
            call_override=None,
            instance_override=None,
            config_override=None,
            default="default",
        )
        assert result == "default"

    def test_empty_string_skipped(self) -> None:
        result = resolve_prompt(
            call_override="",
            instance_override="",
            config_override="",
            default="default",
        )
        assert result == "default"

    def test_whitespace_only_skipped(self) -> None:
        result = resolve_prompt(
            call_override="   ",
            instance_override="\t\n",
            config_override="  \n  ",
            default="default",
        )
        assert result == "default"

    def test_call_override_skipped_when_empty_instance_used(self) -> None:
        result = resolve_prompt(
            call_override="",
            instance_override="instance",
            config_override=None,
            default="default",
        )
        assert result == "instance"

    def test_all_none_returns_default(self) -> None:
        result = resolve_prompt(
            call_override=None,
            instance_override=None,
            config_override=None,
            default="fallback",
        )
        assert result == "fallback"


# ---------------------------------------------------------------------------
# Helpers — lightweight module instantiation
# ---------------------------------------------------------------------------


def _fake_llm(text: str = "test") -> MagicMock:
    r = MagicMock()
    r.text = text
    r.input_tokens = 10
    r.output_tokens = 5
    r.model = "gpt-4o-mini"
    return r


def _mock_settings(custom_prompts: dict | None = None) -> MagicMock:
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.MODEL = "gpt-4o-mini"
    settings.CUSTOM_PROMPTS = custom_prompts or {}
    return settings


# ---------------------------------------------------------------------------
# Classifier — prompt override integration
# ---------------------------------------------------------------------------


class TestClassifierOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None):
        from aiforge.modules.classify.module import Classifier
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        c = object.__new__(Classifier)
        c.labels = ["positive", "negative"]
        c.examples = []
        c.multi_label = False
        c.temperature = 0.0
        c.min_confidence = 0.0
        c._system_prompt_override = system_prompt
        c.logger = structlog.get_logger()
        c._provider = None
        c._embedding_provider = None
        c.settings = _mock_settings(custom_prompts)
        c.cost_tracker = CostTracker.get_session()
        return c

    def test_no_override_uses_default(self) -> None:
        from aiforge.modules.classify.prompts import CLASSIFY_SYSTEM
        c = self._make()
        resp = '{"label": "positive", "confidence": 0.9, "rationale": "good"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp)) as mock:
            c.classify("great product")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == CLASSIFY_SYSTEM

    def test_constructor_override_used(self) -> None:
        c = self._make(system_prompt="Custom classify system")
        resp = '{"label": "positive", "confidence": 0.9, "rationale": "good"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp)) as mock:
            c.classify("great product")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Custom classify system"

    def test_per_call_override_used(self) -> None:
        c = self._make(system_prompt="Constructor prompt")
        resp = '{"label": "positive", "confidence": 0.9, "rationale": "good"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp)) as mock:
            c.classify("great product", system_prompt="Call prompt")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Call prompt"

    def test_config_override_used(self) -> None:
        c = self._make(custom_prompts={"classify": {"system": "YAML override"}})
        resp = '{"label": "positive", "confidence": 0.9, "rationale": "good"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp)) as mock:
            c.classify("great product")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "YAML override"

    def test_run_passes_system_prompt(self) -> None:
        c = self._make()
        resp = '{"label": "positive", "confidence": 0.9, "rationale": "good"}'
        with patch.object(c, "_call_llm", return_value=_fake_llm(resp)) as mock:
            c.run("great product", system_prompt="via-run")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "via-run"


# ---------------------------------------------------------------------------
# Summarizer — prompt override integration
# ---------------------------------------------------------------------------


class TestSummarizerOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None):
        from aiforge.modules.summarize.module import Summarizer
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        s = object.__new__(Summarizer)
        s.style = "concise"
        s.max_length = 500
        s.temperature = 0.0
        s._system_prompt_override = system_prompt
        s.logger = structlog.get_logger()
        s._provider = None
        s._embedding_provider = None
        s.settings = _mock_settings(custom_prompts)
        s.cost_tracker = CostTracker.get_session()
        return s

    def test_no_override_uses_default(self) -> None:
        from aiforge.modules.summarize.prompts import SUMMARIZE_SYSTEM
        s = self._make()
        with patch.object(s, "_call_llm", return_value=_fake_llm("summary")) as mock:
            with patch("aiforge.core.pricing.get_context_window", return_value=100_000):
                s.summarize("Hello world this is text.")
                messages = mock.call_args[0][0]
                assert messages[0]["content"] == SUMMARIZE_SYSTEM

    def test_constructor_override_used(self) -> None:
        s = self._make(system_prompt="Custom summarizer")
        with patch.object(s, "_call_llm", return_value=_fake_llm("summary")) as mock:
            with patch("aiforge.core.pricing.get_context_window", return_value=100_000):
                s.summarize("Hello world this is text.")
                messages = mock.call_args[0][0]
                assert messages[0]["content"] == "Custom summarizer"

    def test_per_call_override_used(self) -> None:
        s = self._make(system_prompt="Constructor")
        with patch.object(s, "_call_llm", return_value=_fake_llm("summary")) as mock:
            with patch("aiforge.core.pricing.get_context_window", return_value=100_000):
                s.summarize("Hello world.", system_prompt="Call prompt")
                messages = mock.call_args[0][0]
                assert messages[0]["content"] == "Call prompt"

    def test_config_override_used(self) -> None:
        s = self._make(custom_prompts={"summarize": {"system": "YAML sum"}})
        with patch.object(s, "_call_llm", return_value=_fake_llm("summary")) as mock:
            with patch("aiforge.core.pricing.get_context_window", return_value=100_000):
                s.summarize("Hello world.")
                messages = mock.call_args[0][0]
                assert messages[0]["content"] == "YAML sum"


# ---------------------------------------------------------------------------
# Extractor — prompt override integration
# ---------------------------------------------------------------------------


class TestExtractorOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None):
        from aiforge.modules.extract.module import Extractor
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        e = object.__new__(Extractor)
        e.output_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        e.examples = []
        e.max_retries = 1
        e.temperature = 0.0
        e._system_prompt_override = system_prompt
        e.logger = structlog.get_logger()
        e._provider = None
        e._embedding_provider = None
        e.settings = _mock_settings(custom_prompts)
        e.cost_tracker = CostTracker.get_session()
        return e

    def test_no_override_uses_default(self) -> None:
        from aiforge.modules.extract.prompts import EXTRACT_SYSTEM
        e = self._make()
        with patch.object(e, "_call_llm", return_value=_fake_llm('{"name": "test"}')) as mock:
            e.extract("John is a developer.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == EXTRACT_SYSTEM

    def test_constructor_override_used(self) -> None:
        e = self._make(system_prompt="Custom extractor")
        with patch.object(e, "_call_llm", return_value=_fake_llm('{"name": "test"}')) as mock:
            e.extract("John is a developer.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Custom extractor"

    def test_per_call_override_used(self) -> None:
        e = self._make(system_prompt="Constructor")
        with patch.object(e, "_call_llm", return_value=_fake_llm('{"name": "test"}')) as mock:
            e.extract("John is a developer.", system_prompt="Call prompt")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Call prompt"


# ---------------------------------------------------------------------------
# Generator — prompt override integration
# ---------------------------------------------------------------------------


class TestGeneratorOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None):
        from aiforge.modules.generate.module import Generator
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        g = object.__new__(Generator)
        g.template = None
        g.guardrails = []
        g.max_tokens = 1024
        g.temperature = 0.7
        g.competitors = []
        g._system_prompt_override = system_prompt
        g.logger = structlog.get_logger()
        g._provider = None
        g._embedding_provider = None
        g.settings = _mock_settings(custom_prompts)
        g.cost_tracker = CostTracker.get_session()
        return g

    def test_no_override_uses_default(self) -> None:
        from aiforge.modules.generate.prompts import GENERATE_SYSTEM
        g = self._make()
        with patch.object(g, "_call_llm", return_value=_fake_llm("content")) as mock:
            g.generate(prompt="Write something.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == GENERATE_SYSTEM

    def test_constructor_override_used(self) -> None:
        g = self._make(system_prompt="Custom generator")
        with patch.object(g, "_call_llm", return_value=_fake_llm("content")) as mock:
            g.generate(prompt="Write something.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Custom generator"

    def test_per_call_override_used(self) -> None:
        g = self._make(system_prompt="Constructor")
        with patch.object(g, "_call_llm", return_value=_fake_llm("content")) as mock:
            g.generate(prompt="Write something.", system_prompt="Call prompt")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Call prompt"


# ---------------------------------------------------------------------------
# Moderator — prompt override integration
# ---------------------------------------------------------------------------


class TestModeratorOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None, checks=None):
        from aiforge.modules.moderate.module import Moderator
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        m = object.__new__(Moderator)
        m.checks = checks or ["toxicity"]
        m.pii_types = ["email", "phone", "ssn", "credit_card"]
        m._system_prompt_override = system_prompt
        m.logger = structlog.get_logger()
        m._provider = None
        m._embedding_provider = None
        m.settings = _mock_settings(custom_prompts)
        m.cost_tracker = CostTracker.get_session()
        return m

    def test_no_override_uses_default(self) -> None:
        from aiforge.modules.moderate.prompts import TOXICITY_SYSTEM
        m = self._make()
        resp = '{"is_toxic": false, "confidence": 0.1, "reason": ""}'
        with patch.object(m, "_call_llm", return_value=_fake_llm(resp)) as mock:
            m.check("Safe text.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == TOXICITY_SYSTEM

    def test_constructor_override_used(self) -> None:
        m = self._make(system_prompt="Custom moderator")
        resp = '{"is_toxic": false, "confidence": 0.1, "reason": ""}'
        with patch.object(m, "_call_llm", return_value=_fake_llm(resp)) as mock:
            m.check("Safe text.")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Custom moderator"

    def test_per_call_override_used(self) -> None:
        m = self._make(system_prompt="Constructor")
        resp = '{"is_toxic": false, "confidence": 0.1, "reason": ""}'
        with patch.object(m, "_call_llm", return_value=_fake_llm(resp)) as mock:
            m.check("Safe text.", system_prompt="Call prompt")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Call prompt"


# ---------------------------------------------------------------------------
# DocQA — prompt override integration
# ---------------------------------------------------------------------------


class TestDocQAOverride:
    def _make(self, *, system_prompt=None, custom_prompts=None):
        from aiforge.modules.qa.module import DocQA
        import structlog
        from aiforge.infra.cost_tracker import CostTracker

        qa = object.__new__(DocQA)
        qa.collection = "test-col"
        qa.persist_dir = "/tmp/test"
        qa.chunk_size = None
        qa.chunk_overlap = None
        qa.retrieval_k = 2
        qa.max_iterations = 1
        qa._system_prompt_override = system_prompt
        qa.logger = structlog.get_logger()
        qa._provider = None
        qa._embedding_provider = MagicMock()
        qa._embedding_provider.embed_query.return_value = [0.1] * 384
        qa._embedding_provider.model_name = "text-embedding-3-small"
        qa.settings = _mock_settings(custom_prompts)
        qa.cost_tracker = CostTracker.get_session()

        store = MagicMock()
        store.collection_stats.return_value = {"chunks": 10}
        store.query.return_value = [
            {"text": "chunk text", "metadata": {"source": "file.py"}},
        ]
        qa._store = store
        return qa

    def test_no_override_no_system_message(self) -> None:
        """With no override, QA uses no system message (empty default)."""
        qa = self._make()
        llm_fn = qa._make_llm_fn()
        with patch.object(qa, "_call_llm", return_value=_fake_llm("answer")) as mock:
            llm_fn("test prompt")
            messages = mock.call_args[0][0]
            # No system message when no override
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_constructor_override_adds_system_message(self) -> None:
        qa = self._make(system_prompt="You are a code expert.")
        llm_fn = qa._make_llm_fn()
        with patch.object(qa, "_call_llm", return_value=_fake_llm("answer")) as mock:
            llm_fn("test prompt")
            messages = mock.call_args[0][0]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a code expert."

    def test_per_call_override_wins(self) -> None:
        qa = self._make(system_prompt="Constructor")
        llm_fn = qa._make_llm_fn(system_prompt="Call override")
        with patch.object(qa, "_call_llm", return_value=_fake_llm("answer")) as mock:
            llm_fn("test prompt")
            messages = mock.call_args[0][0]
            assert messages[0]["content"] == "Call override"


# ---------------------------------------------------------------------------
# AIForgeSettings — CUSTOM_PROMPTS field
# ---------------------------------------------------------------------------


class TestSettingsCustomPrompts:
    def test_default_empty(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}, clear=False):
            from aiforge.infra.config import AIForgeSettings
            s = AIForgeSettings()
            assert s.CUSTOM_PROMPTS == {}

    def test_custom_prompts_dict(self) -> None:
        with patch.dict("os.environ", {"AIFORGE_OPENAI_API_KEY": "test-key"}, clear=False):
            from aiforge.infra.config import AIForgeSettings
            cp = {"classify": {"system": "Override"}}
            s = AIForgeSettings(CUSTOM_PROMPTS=cp)
            assert s.CUSTOM_PROMPTS == cp
            assert s.CUSTOM_PROMPTS["classify"]["system"] == "Override"
