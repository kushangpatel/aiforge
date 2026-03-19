"""
Generator module — template-based and freeform content generation with guardrails.

Usage::

    from aiforge import Generator

    # Template-based generation
    g = Generator(
        template="Write a {tone} product description for {product}.",
        guardrails=["no_pii", "no_pricing"],
    )
    result = g.generate(tone="professional", product="AIForge")
    print(result.text)

    # Freeform generation
    g = Generator(guardrails=["no_pii"])
    result = g.generate(prompt="Write a follow-up email for a customer complaint.")
    print(result.text)

    # Custom guardrails
    from aiforge.modules.generate import register_guardrail, GuardrailResult

    @register_guardrail("max_sentences")
    def _check(text: str) -> GuardrailResult:
        count = text.count(".") + text.count("!") + text.count("?")
        return GuardrailResult(passed=count <= 5, message=f"{count} sentences")
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from aiforge.core.types import BaseModule
from aiforge.modules.generate.prompts import (
    GENERATE_RETRY,
    GENERATE_SYSTEM,
    PROFESSIONAL_TONE_CHECK,
)
from aiforge.modules.generate.schemas import GenerateResponse, GuardrailResult

# ---------------------------------------------------------------------------
# Maximum number of guardrail retry attempts
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Currency patterns for no_pricing guardrail
# ---------------------------------------------------------------------------

_PRICE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\$\d+"),
    re.compile(r"€\d+"),
    re.compile(r"£\d+"),
    re.compile(r"\d+\s*(?:USD|EUR|GBP)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Custom guardrail registry (module-level singleton)
# ---------------------------------------------------------------------------

_CUSTOM_GUARDRAILS: dict[str, Callable[[str], GuardrailResult]] = {}


def register_guardrail(name: str) -> Callable[[Callable[[str], GuardrailResult]], Callable[[str], GuardrailResult]]:
    """Decorator to register a custom guardrail function.

    The decorated function receives the generated text and must return a
    :class:`GuardrailResult`.  It is invoked by name during the guardrail
    check loop.

    Example::

        @register_guardrail("max_words")
        def _check_words(text: str) -> GuardrailResult:
            ok = len(text.split()) <= 200
            return GuardrailResult(passed=ok, message="Too many words" if not ok else "")
    """
    def decorator(fn: Callable[[str], GuardrailResult]) -> Callable[[str], GuardrailResult]:
        _CUSTOM_GUARDRAILS[name] = fn
        return fn
    return decorator


def get_custom_guardrails() -> dict[str, Callable[[str], GuardrailResult]]:
    """Return a *read-only view* of the custom guardrail registry."""
    return dict(_CUSTOM_GUARDRAILS)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class Generator(BaseModule):
    """Content generator with template rendering and guardrail enforcement.

    Args:
        template:    A template string with ``{variable}`` placeholders.
                     Mutually exclusive with *prompt* at call time.
        guardrails:  List of guardrail names to enforce on generated output.
                     Built-in: ``no_pii``, ``no_competitor_mentions``,
                     ``no_pricing``, ``professional_tone``.
                     Custom guardrails via :func:`register_guardrail`.
        max_tokens:  Maximum tokens for the LLM generation call.
        temperature: Sampling temperature (0.0–2.0).
        competitors: List of competitor names for ``no_competitor_mentions``.
        **kwargs:    Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        template: str | None = None,
        guardrails: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        competitors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.template = template
        self.guardrails: list[str] = guardrails or []
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.competitors: list[str] = competitors or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str | None = None, **variables: Any) -> GenerateResponse:
        """Generate content synchronously.

        Either *self.template* (with ``**variables``) **or** *prompt* must
        be provided — not both, and not neither.
        """
        t0 = time.monotonic()
        final_prompt = self._resolve_prompt(prompt, variables)
        return self._generate_loop(final_prompt, t0, sync=True)

    async def agenerate(self, prompt: str | None = None, **variables: Any) -> GenerateResponse:
        """Async version of :meth:`generate`."""
        t0 = time.monotonic()
        final_prompt = self._resolve_prompt(prompt, variables)
        return await self._agenerate_loop(final_prompt, t0)

    # Pipeline interface -----------------------------------------------

    def run(self, text: str, **kwargs: Any) -> GenerateResponse:
        """Pipeline entry point (sync).

        *text* is treated as a freeform prompt.
        """
        return self.generate(prompt=text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> GenerateResponse:
        """Pipeline entry point (async)."""
        return await self.agenerate(prompt=text, **kwargs)

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    def _resolve_prompt(self, prompt: str | None, variables: dict[str, Any]) -> str:
        """Build the final user prompt from template + variables or freeform *prompt*."""
        if self.template and prompt:
            raise ValueError("Provide either template variables or prompt — not both")
        if self.template:
            return self._render_template(self.template, variables)
        if prompt:
            return prompt
        raise ValueError("Either template or prompt is required")

    @staticmethod
    def _render_template(template: str, variables: dict[str, Any]) -> str:
        """Substitute ``{key}`` placeholders in *template*.

        List values are joined with ``", "``.
        """
        rendered = template
        for key, value in variables.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    # ------------------------------------------------------------------
    # Generate-and-guard loop
    # ------------------------------------------------------------------

    def _generate_loop(self, final_prompt: str, t0: float, *, sync: bool = True) -> GenerateResponse:
        """Synchronous generate → check guardrails → retry loop."""
        total_in = total_out = 0
        total_cost = 0.0
        model = ""
        generated_text = ""
        guardrail_flags: list[str] = []

        current_prompt = final_prompt
        for attempt in range(_MAX_RETRIES + 1):
            llm = self._call_llm(
                [
                    {"role": "system", "content": GENERATE_SYSTEM},
                    {"role": "user", "content": current_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            from aiforge.core.pricing import calculate_cost

            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            model = llm.model
            generated_text = llm.text.strip()

            violations = self._check_guardrails(generated_text)
            if not violations:
                return GenerateResponse(
                    text=generated_text,
                    guardrail_flags=[],
                    retry_count=attempt,
                    cost=total_cost,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    model=model,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            guardrail_flags = violations
            if attempt < _MAX_RETRIES:
                current_prompt = self._build_retry_prompt(
                    final_prompt, generated_text, violations,
                )

        # All retries exhausted — return last result with remaining flags
        return GenerateResponse(
            text=generated_text,
            guardrail_flags=guardrail_flags,
            retry_count=_MAX_RETRIES,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def _agenerate_loop(self, final_prompt: str, t0: float) -> GenerateResponse:
        """Async generate → check guardrails → retry loop."""
        total_in = total_out = 0
        total_cost = 0.0
        model = ""
        generated_text = ""
        guardrail_flags: list[str] = []

        current_prompt = final_prompt
        for attempt in range(_MAX_RETRIES + 1):
            llm = await self._acall_llm(
                [
                    {"role": "system", "content": GENERATE_SYSTEM},
                    {"role": "user", "content": current_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            from aiforge.core.pricing import calculate_cost

            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            model = llm.model
            generated_text = llm.text.strip()

            violations = await self._acheck_guardrails(generated_text)
            if not violations:
                return GenerateResponse(
                    text=generated_text,
                    guardrail_flags=[],
                    retry_count=attempt,
                    cost=total_cost,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    model=model,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            guardrail_flags = violations
            if attempt < _MAX_RETRIES:
                current_prompt = self._build_retry_prompt(
                    final_prompt, generated_text, violations,
                )

        return GenerateResponse(
            text=generated_text,
            guardrail_flags=guardrail_flags,
            retry_count=_MAX_RETRIES,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Guardrail checks
    # ------------------------------------------------------------------

    def _check_guardrails(self, text: str) -> list[str]:
        """Check all configured guardrails synchronously.

        Returns the names of violated guardrails (empty ⇒ clean).
        """
        violations: list[str] = []
        for name in self.guardrails:
            if name == "no_pii":
                if self._check_no_pii(text):
                    violations.append("no_pii")
            elif name == "no_competitor_mentions":
                if self._check_competitor_mentions(text):
                    violations.append("no_competitor_mentions")
            elif name == "no_pricing":
                if self._check_pricing(text):
                    violations.append("no_pricing")
            elif name == "professional_tone":
                if not self._check_professional_tone(text, sync=True):
                    violations.append("professional_tone")
            elif name in _CUSTOM_GUARDRAILS:
                result = _CUSTOM_GUARDRAILS[name](text)
                if not result.passed:
                    violations.append(name)
            else:
                raise ValueError(f"Unknown guardrail: {name!r}")
        return violations

    async def _acheck_guardrails(self, text: str) -> list[str]:
        """Async version of :meth:`_check_guardrails`."""
        violations: list[str] = []
        for name in self.guardrails:
            if name == "no_pii":
                if self._check_no_pii(text):
                    violations.append("no_pii")
            elif name == "no_competitor_mentions":
                if self._check_competitor_mentions(text):
                    violations.append("no_competitor_mentions")
            elif name == "no_pricing":
                if self._check_pricing(text):
                    violations.append("no_pricing")
            elif name == "professional_tone":
                if not await self._acheck_professional_tone(text):
                    violations.append("professional_tone")
            elif name in _CUSTOM_GUARDRAILS:
                result = _CUSTOM_GUARDRAILS[name](text)
                if not result.passed:
                    violations.append(name)
            else:
                raise ValueError(f"Unknown guardrail: {name!r}")
        return violations

    # ------------------------------------------------------------------
    # Individual guardrail implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_no_pii(text: str) -> bool:
        """Return *True* if PII is detected (i.e. guardrail violated)."""
        from aiforge.modules.moderate.module import Moderator

        m = Moderator(checks=["pii"])
        return m.check(text).flagged

    def _check_competitor_mentions(self, text: str) -> bool:
        """Regex check for competitor names.  Returns *True* on match."""
        if not self.competitors:
            return False
        pattern = "|".join(re.escape(c) for c in self.competitors)
        return bool(re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _check_pricing(text: str) -> bool:
        """Return *True* if currency / pricing patterns are found."""
        return any(p.search(text) for p in _PRICE_PATTERNS)

    def _check_professional_tone(self, text: str, *, sync: bool = True) -> bool:
        """LLM-based tone check.  Returns *True* when tone IS professional."""
        user_content = PROFESSIONAL_TONE_CHECK.format(text=text)
        llm = self._call_llm(
            [{"role": "user", "content": user_content}],
            temperature=0.0,
            max_tokens=8,
        )
        return "YES" in llm.text.upper()

    async def _acheck_professional_tone(self, text: str) -> bool:
        """Async professional-tone check."""
        user_content = PROFESSIONAL_TONE_CHECK.format(text=text)
        llm = await self._acall_llm(
            [{"role": "user", "content": user_content}],
            temperature=0.0,
            max_tokens=8,
        )
        return "YES" in llm.text.upper()

    # ------------------------------------------------------------------
    # Retry prompt builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_retry_prompt(
        original_prompt: str, generated_text: str, violations: list[str],
    ) -> str:
        """Construct the retry message with violation feedback."""
        return GENERATE_RETRY.format(
            violations=", ".join(violations),
            previous_text=generated_text,
            original_prompt=original_prompt,
        )
