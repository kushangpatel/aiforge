"""Unit tests for the Generator module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiforge.modules.generate.module import (
    Generator,
    _CUSTOM_GUARDRAILS,
    _MAX_RETRIES,
    _PRICE_PATTERNS,
    register_guardrail,
)
from aiforge.modules.generate.prompts import (
    GENERATE_RETRY,
    GENERATE_SYSTEM,
    PROFESSIONAL_TONE_CHECK,
)
from aiforge.modules.generate.schemas import GenerateResponse, GuardrailResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen(
    template: str | None = None,
    guardrails: list[str] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    competitors: list[str] | None = None,
) -> Generator:
    """Instantiate a Generator bypassing BaseModule provider initialisation."""
    g = object.__new__(Generator)
    g.template = template
    g.guardrails = guardrails or []
    g.max_tokens = max_tokens
    g.temperature = temperature
    g.competitors = competitors or []

    import structlog

    g.logger = structlog.get_logger()
    g._provider = None
    g._embedding_provider = None
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    settings.MODEL = "gpt-4o-mini"
    settings.CUSTOM_PROMPTS = {}
    g.settings = settings
    g._system_prompt_override = None
    from aiforge.infra.cost_tracker import CostTracker

    g.cost_tracker = CostTracker.get_session()
    return g


def _fake_llm(text: str = "Generated content here.") -> MagicMock:
    """Create a fake CompletionResult."""
    r = MagicMock()
    r.text = text
    r.input_tokens = 50
    r.output_tokens = 30
    r.model = "gpt-4o-mini"
    return r


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self) -> None:
        g = _gen()
        assert g.template is None
        assert g.guardrails == []
        assert g.max_tokens == 1024
        assert g.temperature == 0.7
        assert g.competitors == []

    def test_custom_params(self) -> None:
        g = _gen(
            template="Write about {topic}",
            guardrails=["no_pii"],
            max_tokens=512,
            temperature=0.5,
            competitors=["Acme"],
        )
        assert g.template == "Write about {topic}"
        assert g.guardrails == ["no_pii"]
        assert g.max_tokens == 512
        assert g.temperature == 0.5
        assert g.competitors == ["Acme"]

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            Generator(max_tokens=0)

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            Generator(max_tokens=-1)

    def test_temperature_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be between"):
            Generator(temperature=2.5)

    def test_temperature_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be between"):
            Generator(temperature=-0.1)

    def test_temperature_boundaries_ok(self) -> None:
        # should not raise
        g0 = _gen(temperature=0.0)
        assert g0.temperature == 0.0
        g2 = _gen(temperature=2.0)
        assert g2.temperature == 2.0


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------


class TestPromptResolution:
    def test_requires_template_or_prompt(self) -> None:
        g = _gen()
        with pytest.raises(ValueError, match="Either template or prompt"):
            g._resolve_prompt(None, {})

    def test_template_and_prompt_raises(self) -> None:
        g = _gen(template="Write about {topic}")
        with pytest.raises(ValueError, match="not both"):
            g._resolve_prompt("hello", {})

    def test_template_only(self) -> None:
        g = _gen(template="Describe {item} in {tone}.")
        result = g._resolve_prompt(None, {"item": "AIForge", "tone": "detail"})
        assert result == "Describe AIForge in detail."

    def test_prompt_only(self) -> None:
        g = _gen()
        result = g._resolve_prompt("Write a poem", {})
        assert result == "Write a poem"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_simple_substitution(self) -> None:
        got = Generator._render_template("Hello {name}!", {"name": "World"})
        assert got == "Hello World!"

    def test_multiple_variables(self) -> None:
        tpl = "{greeting}, {name}. How is {thing}?"
        got = Generator._render_template(tpl, {
            "greeting": "Hi",
            "name": "Alice",
            "thing": "your day",
        })
        assert got == "Hi, Alice. How is your day?"

    def test_list_variable_joined(self) -> None:
        got = Generator._render_template(
            "Features: {features}",
            {"features": ["fast", "scalable", "easy"]},
        )
        assert got == "Features: fast, scalable, easy"

    def test_numeric_variable(self) -> None:
        got = Generator._render_template("Max: {n} words", {"n": 150})
        assert got == "Max: 150 words"

    def test_missing_variable_left_as_is(self) -> None:
        got = Generator._render_template("Hello {name}!", {})
        assert got == "Hello {name}!"

    def test_empty_template(self) -> None:
        got = Generator._render_template("", {"x": "y"})
        assert got == ""


# ---------------------------------------------------------------------------
# Freeform generation (sync)
# ---------------------------------------------------------------------------


class TestSyncGenerate:
    def test_freeform_basic(self) -> None:
        g = _gen()
        fake = _fake_llm("Generated poem.")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="Write a poem")
        assert result.text == "Generated poem."
        assert result.guardrail_flags == []
        assert result.retry_count == 0
        assert result.input_tokens == 50
        assert result.output_tokens == 30
        assert result.model == "gpt-4o-mini"
        assert result.latency_ms >= 0

    def test_template_basic(self) -> None:
        g = _gen(template="Write about {topic}")
        fake = _fake_llm("All about AI.")
        with patch.object(g, "_call_llm", return_value=fake) as llm_mock:
            result = g.generate(topic="AI")
        assert result.text == "All about AI."
        call_args = llm_mock.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == GENERATE_SYSTEM
        assert "Write about AI" in messages[1]["content"]

    def test_no_prompt_or_template_raises(self) -> None:
        g = _gen()
        with pytest.raises(ValueError, match="Either template or prompt"):
            g.generate()

    def test_template_and_prompt_raises(self) -> None:
        g = _gen(template="Write about {topic}")
        with pytest.raises(ValueError, match="not both"):
            g.generate(prompt="ignored")

    def test_llm_kwargs_forwarded(self) -> None:
        g = _gen(max_tokens=256, temperature=0.3)
        fake = _fake_llm("text")
        with patch.object(g, "_call_llm", return_value=fake) as mock:
            g.generate(prompt="hello")
        _, kwargs = mock.call_args
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.3

    def test_cost_accumulated(self) -> None:
        g = _gen()
        fake = _fake_llm("text")
        with (
            patch.object(g, "_call_llm", return_value=fake),
            patch("aiforge.core.pricing.calculate_cost", return_value=0.0042),
        ):
            result = g.generate(prompt="go")
        assert result.cost == pytest.approx(0.0042)


# ---------------------------------------------------------------------------
# Freeform generation (async)
# ---------------------------------------------------------------------------


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_freeform_async(self) -> None:
        g = _gen()
        fake = _fake_llm("Async result.")
        with patch.object(g, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            result = await g.agenerate(prompt="Write something")
        assert result.text == "Async result."
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_template_async(self) -> None:
        g = _gen(template="Hello {name}")
        fake = _fake_llm("Hello Alice.")
        with patch.object(g, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            result = await g.agenerate(name="Alice")
        assert result.text == "Hello Alice."

    @pytest.mark.asyncio
    async def test_no_prompt_raises_async(self) -> None:
        g = _gen()
        with pytest.raises(ValueError, match="Either template or prompt"):
            await g.agenerate()


# ---------------------------------------------------------------------------
# Pipeline interface (run / arun)
# ---------------------------------------------------------------------------


class TestPipelineInterface:
    def test_run_delegates_to_generate(self) -> None:
        g = _gen()
        fake = _fake_llm("pipeline result")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.run("Write something")
        assert result.text == "pipeline result"

    @pytest.mark.asyncio
    async def test_arun_delegates_to_agenerate(self) -> None:
        g = _gen()
        fake = _fake_llm("async pipeline")
        with patch.object(g, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            result = await g.arun("Write something")
        assert result.text == "async pipeline"


# ---------------------------------------------------------------------------
# Guardrail: no_pricing
# ---------------------------------------------------------------------------


class TestNoPricingGuardrail:
    def test_dollar_sign(self) -> None:
        assert Generator._check_pricing("Only $99 today!") is True

    def test_euro_sign(self) -> None:
        assert Generator._check_pricing("€49 per month") is True

    def test_pound_sign(self) -> None:
        assert Generator._check_pricing("From £25") is True

    def test_usd_suffix(self) -> None:
        assert Generator._check_pricing("100 USD") is True

    def test_eur_suffix(self) -> None:
        assert Generator._check_pricing("50 EUR") is True

    def test_gbp_suffix(self) -> None:
        assert Generator._check_pricing("75 GBP") is True

    def test_no_price(self) -> None:
        assert Generator._check_pricing("This is a great product.") is False

    def test_empty_text(self) -> None:
        assert Generator._check_pricing("") is False


# ---------------------------------------------------------------------------
# Guardrail: no_competitor_mentions
# ---------------------------------------------------------------------------


class TestNoCompetitorMentionsGuardrail:
    def test_competitor_found(self) -> None:
        g = _gen(competitors=["Acme", "Globex"])
        assert g._check_competitor_mentions("Acme makes good tools.") is True

    def test_competitor_case_insensitive(self) -> None:
        g = _gen(competitors=["Acme"])
        assert g._check_competitor_mentions("ACME is a rival.") is True

    def test_no_competitor(self) -> None:
        g = _gen(competitors=["Acme"])
        assert g._check_competitor_mentions("Our product is best.") is False

    def test_empty_competitors_list(self) -> None:
        g = _gen(competitors=[])
        assert g._check_competitor_mentions("Acme is mentioned.") is False

    def test_no_competitors_configured(self) -> None:
        g = _gen()
        assert g._check_competitor_mentions("Acme") is False


# ---------------------------------------------------------------------------
# Guardrail: no_pii (delegates to Moderator)
# ---------------------------------------------------------------------------


class TestNoPiiGuardrail:
    def test_pii_detected(self) -> None:
        # Real integration with Moderator's regex-based PII check
        assert Generator._check_no_pii("Email: john@example.com") is True

    def test_no_pii(self) -> None:
        assert Generator._check_no_pii("Hello, how are you?") is False

    def test_ssn_detected(self) -> None:
        assert Generator._check_no_pii("SSN: 123-45-6789") is True


# ---------------------------------------------------------------------------
# Guardrail: professional_tone (LLM-based)
# ---------------------------------------------------------------------------


class TestProfessionalToneGuardrail:
    def test_professional_text(self) -> None:
        g = _gen()
        fake = _fake_llm("YES")
        with patch.object(g, "_call_llm", return_value=fake):
            is_pro, _in, _out, _cost = g._check_professional_tone("We appreciate your inquiry.")
            assert is_pro is True

    def test_unprofessional_text(self) -> None:
        g = _gen()
        fake = _fake_llm("NO")
        with patch.object(g, "_call_llm", return_value=fake):
            is_pro, _in, _out, _cost = g._check_professional_tone("yo dude what's up lol")
            assert is_pro is False

    @pytest.mark.asyncio
    async def test_async_professional_tone(self) -> None:
        g = _gen()
        fake = _fake_llm("YES")
        with patch.object(g, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            is_pro, _in, _out, _cost = await g._acheck_professional_tone("Formal message.")
        assert is_pro is True


# ---------------------------------------------------------------------------
# Unknown guardrail raises
# ---------------------------------------------------------------------------


class TestUnknownGuardrail:
    def test_unknown_guardrail_raises(self) -> None:
        g = _gen(guardrails=["nonexistent_guardrail"])
        with pytest.raises(ValueError, match="Unknown guardrail"):
            g._check_guardrails("some text")

    @pytest.mark.asyncio
    async def test_unknown_guardrail_raises_async(self) -> None:
        g = _gen(guardrails=["nonexistent_guardrail"])
        with pytest.raises(ValueError, match="Unknown guardrail"):
            await g._acheck_guardrails("some text")


# ---------------------------------------------------------------------------
# Retry loop — sync
# ---------------------------------------------------------------------------


class TestRetryLoop:
    def test_no_guardrails_no_retry(self) -> None:
        g = _gen()
        fake = _fake_llm("clean output")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="go")
        assert result.retry_count == 0
        assert result.guardrail_flags == []

    def test_guardrail_passes_first_try(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        fake = _fake_llm("No prices mentioned.")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="describe product")
        assert result.retry_count == 0
        assert result.guardrail_flags == []

    def test_guardrail_violation_triggers_retry(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        bad = _fake_llm("Only $99 today!")
        good = _fake_llm("Great product at a great value.")
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return bad if call_count == 1 else good

        with patch.object(g, "_call_llm", side_effect=side_effect):
            result = g.generate(prompt="describe")
        assert result.retry_count == 1
        assert result.guardrail_flags == []
        assert result.text == "Great product at a great value."

    def test_all_retries_exhausted(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        bad = _fake_llm("Only $99 today!")
        with patch.object(g, "_call_llm", return_value=bad):
            result = g.generate(prompt="describe")
        assert result.retry_count == _MAX_RETRIES
        assert "no_pricing" in result.guardrail_flags
        assert result.text == "Only $99 today!"

    def test_cost_accumulates_across_retries(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        bad = _fake_llm("$50 deal!")
        good = _fake_llm("Great value.")
        calls = iter([bad, good])
        with (
            patch.object(g, "_call_llm", side_effect=lambda *a, **kw: next(calls)),
            patch(
                "aiforge.core.pricing.calculate_cost",
                return_value=0.001,
            ),
        ):
            result = g.generate(prompt="go")
        assert result.cost == pytest.approx(0.002)  # 2 calls * 0.001
        assert result.input_tokens == 100  # 2 * 50
        assert result.output_tokens == 60  # 2 * 30

    def test_retry_prompt_includes_violations(self) -> None:
        prompt = Generator._build_retry_prompt(
            "original prompt",
            "bad output with $99",
            ["no_pricing"],
        )
        assert "no_pricing" in prompt
        assert "bad output with $99" in prompt
        assert "original prompt" in prompt


# ---------------------------------------------------------------------------
# Retry loop — async
# ---------------------------------------------------------------------------


class TestAsyncRetryLoop:
    @pytest.mark.asyncio
    async def test_async_retry_on_violation(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        bad = _fake_llm("$99 special!")
        good = _fake_llm("Affordable product.")
        calls = iter([bad, good])
        with patch.object(
            g, "_acall_llm", new_callable=AsyncMock,
            side_effect=lambda *a, **kw: next(calls),
        ):
            result = await g.agenerate(prompt="describe")
        assert result.retry_count == 1
        assert result.guardrail_flags == []
        assert result.text == "Affordable product."

    @pytest.mark.asyncio
    async def test_async_all_retries_exhausted(self) -> None:
        g = _gen(guardrails=["no_pricing"])
        bad = _fake_llm("€50 per month!")
        with patch.object(
            g, "_acall_llm", new_callable=AsyncMock, return_value=bad,
        ):
            result = await g.agenerate(prompt="describe")
        assert result.retry_count == _MAX_RETRIES
        assert "no_pricing" in result.guardrail_flags


# ---------------------------------------------------------------------------
# Multiple guardrails
# ---------------------------------------------------------------------------


class TestMultipleGuardrails:
    def test_multiple_violations_all_reported(self) -> None:
        g = _gen(guardrails=["no_pricing", "no_pii"])
        bad = _fake_llm("Contact john@test.com for the $50 deal!")
        with patch.object(g, "_call_llm", return_value=bad):
            result = g.generate(prompt="write email")
        assert "no_pricing" in result.guardrail_flags
        assert "no_pii" in result.guardrail_flags

    def test_one_passes_one_fails(self) -> None:
        g = _gen(guardrails=["no_pricing", "no_pii"])
        # Has PII but no pricing
        bad = _fake_llm("Contact john@test.com for details.")
        good = _fake_llm("Contact our support team for details.")
        calls = iter([bad, good])
        with patch.object(g, "_call_llm", side_effect=lambda *a, **kw: next(calls)):
            result = g.generate(prompt="write")
        assert result.retry_count == 1
        assert result.guardrail_flags == []


# ---------------------------------------------------------------------------
# Custom guardrail registration
# ---------------------------------------------------------------------------


class TestCustomGuardrails:
    def setup_method(self) -> None:
        """Clean registry before each test."""
        _CUSTOM_GUARDRAILS.pop("test_guard", None)
        _CUSTOM_GUARDRAILS.pop("always_fail", None)

    def teardown_method(self) -> None:
        """Clean registry after each test."""
        _CUSTOM_GUARDRAILS.pop("test_guard", None)
        _CUSTOM_GUARDRAILS.pop("always_fail", None)

    def test_register_and_pass(self) -> None:
        @register_guardrail("test_guard")
        def _check(text: str) -> GuardrailResult:
            return GuardrailResult(passed=True)

        assert "test_guard" in _CUSTOM_GUARDRAILS

        g = _gen(guardrails=["test_guard"])
        fake = _fake_llm("some content")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="go")
        assert result.guardrail_flags == []

    def test_register_and_fail(self) -> None:
        @register_guardrail("always_fail")
        def _check(text: str) -> GuardrailResult:
            return GuardrailResult(passed=False, message="always fails")

        g = _gen(guardrails=["always_fail"])
        fake = _fake_llm("content")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="go")
        assert "always_fail" in result.guardrail_flags
        assert result.retry_count == _MAX_RETRIES

    def test_custom_guardrail_with_retry(self) -> None:
        call_count = 0

        @register_guardrail("test_guard")
        def _check(text: str) -> GuardrailResult:
            nonlocal call_count
            call_count += 1
            return GuardrailResult(passed=call_count > 1)

        g = _gen(guardrails=["test_guard"])
        fake = _fake_llm("output")
        with patch.object(g, "_call_llm", return_value=fake):
            result = g.generate(prompt="go")
        assert result.retry_count == 1
        assert result.guardrail_flags == []


# ---------------------------------------------------------------------------
# GenerateResponse schema
# ---------------------------------------------------------------------------


class TestGenerateResponse:
    def test_to_text(self) -> None:
        r = GenerateResponse(text="Hello world")
        assert r.to_text() == "Hello world"

    def test_to_text_empty(self) -> None:
        r = GenerateResponse()
        assert r.to_text() == ""

    def test_default_values(self) -> None:
        r = GenerateResponse()
        assert r.text == ""
        assert r.guardrail_flags == []
        assert r.retry_count == 0
        assert r.cost == 0.0
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_inherits_module_response_fields(self) -> None:
        r = GenerateResponse(
            text="hi",
            model="gpt-4o",
            cost=0.01,
            latency_ms=150.0,
        )
        assert r.model == "gpt-4o"
        assert r.cost == 0.01
        assert r.latency_ms == 150.0


# ---------------------------------------------------------------------------
# GuardrailResult dataclass
# ---------------------------------------------------------------------------


class TestGuardrailResult:
    def test_defaults(self) -> None:
        r = GuardrailResult(passed=True)
        assert r.passed is True
        assert r.message == ""

    def test_with_message(self) -> None:
        r = GuardrailResult(passed=False, message="violation detected")
        assert r.passed is False
        assert r.message == "violation detected"


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_not_empty(self) -> None:
        assert len(GENERATE_SYSTEM) > 0

    def test_retry_template_has_placeholders(self) -> None:
        assert "{violations}" in GENERATE_RETRY
        assert "{previous_text}" in GENERATE_RETRY
        assert "{original_prompt}" in GENERATE_RETRY

    def test_professional_tone_has_placeholder(self) -> None:
        assert "{text}" in PROFESSIONAL_TONE_CHECK


# ---------------------------------------------------------------------------
# Batch operations (inherited from BaseModule)
# ---------------------------------------------------------------------------


class TestBatchOperations:
    def test_run_batch_all_succeed(self) -> None:
        g = _gen()
        fake = _fake_llm("batch result")
        with patch.object(g, "_acall_llm", new_callable=AsyncMock, return_value=fake):
            batch = g.run_batch(["prompt 1", "prompt 2"])
        assert len(batch.results) == 2
        assert all(r.text == "batch result" for r in batch.results)

    def test_run_batch_empty_returns_empty(self) -> None:
        g = _gen()
        batch = g.run_batch([])
        assert len(batch.results) == 0


# ---------------------------------------------------------------------------
# Server route
# ---------------------------------------------------------------------------


class TestGenerateServerRoute:
    def _make_app(self) -> Any:
        from fastapi import FastAPI
        from aiforge.server.routes.generate import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_generate_returns_200(self) -> None:
        from starlette.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            fake = _fake_llm("Route result.")
            with (
                patch(
                    "aiforge.modules.generate.module.Generator.agenerate",
                    new_callable=AsyncMock,
                    return_value=GenerateResponse(
                        text="Route result.",
                        guardrail_flags=[],
                        retry_count=0,
                        cost=0.001,
                        input_tokens=50,
                        output_tokens=30,
                        model="gpt-4o-mini",
                        latency_ms=100.0,
                    ),
                ),
            ):
                resp = client.post("/generate", json={"prompt": "Write a story."})
            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "Route result."
            assert data["guardrail_flags"] == []

    def test_generate_with_template(self) -> None:
        from starlette.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            with patch(
                "aiforge.modules.generate.module.Generator.agenerate",
                new_callable=AsyncMock,
                return_value=GenerateResponse(text="Hi Alice.", retry_count=0),
            ):
                resp = client.post(
                    "/generate",
                    json={
                        "template": "Hello {name}",
                        "variables": {"name": "Alice"},
                    },
                )
            assert resp.status_code == 200
            assert resp.json()["text"] == "Hi Alice."

    def test_generate_missing_prompt_and_template_returns_422(self) -> None:
        from starlette.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            with patch(
                "aiforge.modules.generate.module.Generator.agenerate",
                new_callable=AsyncMock,
                side_effect=ValueError("Either template or prompt is required"),
            ):
                resp = client.post("/generate", json={})
            assert resp.status_code == 422

    def test_generate_invalid_temperature_returns_422(self) -> None:
        from starlette.testclient import TestClient

        app = self._make_app()
        with TestClient(app) as client:
            resp = client.post(
                "/generate",
                json={"prompt": "hi", "temperature": 5.0},
            )
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Price pattern coverage
# ---------------------------------------------------------------------------


class TestPricePatterns:
    @pytest.mark.parametrize(
        "text",
        [
            "$100",
            "€50",
            "£25",
            "99 USD",
            "49 EUR",
            "75 GBP",
            "$0",
            "999 usd",
        ],
    )
    def test_price_detected(self, text: str) -> None:
        assert any(p.search(text) for p in _PRICE_PATTERNS)

    @pytest.mark.parametrize(
        "text",
        ["no prices here", "great value", "123 items"],
    )
    def test_no_price(self, text: str) -> None:
        assert not any(p.search(text) for p in _PRICE_PATTERNS)


# ---------------------------------------------------------------------------
# get_custom_guardrails utility
# ---------------------------------------------------------------------------


class TestGetCustomGuardrails:
    def setup_method(self) -> None:
        _CUSTOM_GUARDRAILS.pop("test_util", None)

    def teardown_method(self) -> None:
        _CUSTOM_GUARDRAILS.pop("test_util", None)

    def test_returns_copy(self) -> None:
        from aiforge.modules.generate.module import get_custom_guardrails

        @register_guardrail("test_util")
        def _check(text: str) -> GuardrailResult:
            return GuardrailResult(passed=True)

        snapshot = get_custom_guardrails()
        assert "test_util" in snapshot
        # Mutating the copy must not affect the real registry
        snapshot.pop("test_util")
        assert "test_util" in _CUSTOM_GUARDRAILS

    def test_empty_when_clean(self) -> None:
        from aiforge.modules.generate.module import get_custom_guardrails

        result = get_custom_guardrails()
        assert "test_util" not in result


# ---------------------------------------------------------------------------
# professional_tone through retry loop
# ---------------------------------------------------------------------------


class TestProfessionalToneRetryLoop:
    def test_unprofessional_triggers_retry(self) -> None:
        g = _gen(guardrails=["professional_tone"])
        tone_calls = iter([_fake_llm("NO"), _fake_llm("YES")])
        gen_calls = iter([_fake_llm("yo sup"), _fake_llm("Dear valued customer,")])

        def llm_side_effect(messages: Any, **kwargs: Any) -> MagicMock:
            content = messages[-1]["content"]
            if "professional tone" in content.lower() or "Evaluate" in content:
                return next(tone_calls)
            return next(gen_calls)

        with patch.object(g, "_call_llm", side_effect=llm_side_effect):
            result = g.generate(prompt="Write a greeting")
        assert result.retry_count == 1
        assert result.guardrail_flags == []
        assert result.text == "Dear valued customer,"

    @pytest.mark.asyncio
    async def test_unprofessional_triggers_retry_async(self) -> None:
        g = _gen(guardrails=["professional_tone"])
        tone_calls = iter([_fake_llm("NO"), _fake_llm("YES")])
        gen_calls = iter([_fake_llm("yo"), _fake_llm("Respectfully,")])

        async def llm_side_effect(messages: Any, **kwargs: Any) -> MagicMock:
            content = messages[-1]["content"]
            if "professional tone" in content.lower() or "Evaluate" in content:
                return next(tone_calls)
            return next(gen_calls)

        with patch.object(g, "_acall_llm", new_callable=AsyncMock, side_effect=llm_side_effect):
            result = await g.agenerate(prompt="Write a greeting")
        assert result.retry_count == 1
        assert result.guardrail_flags == []


# ---------------------------------------------------------------------------
# Batch partial failure
# ---------------------------------------------------------------------------


class TestBatchPartialFailure:
    def test_run_batch_partial_failure(self) -> None:
        g = _gen()
        fake = _fake_llm("batch ok")
        call_count = 0

        async def _selective(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("llm error")
            return fake

        with patch.object(g, "_acall_llm", new_callable=AsyncMock, side_effect=_selective):
            batch = g.run_batch(["p1", "p2", "p3"])
        assert len(batch.results) == 2
        assert len(batch.failed) == 1
        assert batch.failed[0]["error"] == "llm error"
