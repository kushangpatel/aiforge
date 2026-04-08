"""
Translator module — cross-language text translation with tone and formality control.

Usage::

    from aiforge import Translator

    # Basic translation
    t = Translator(target_language="Spanish")
    result = t.translate("Hello, how are you?")
    print(result.text)             # "Hola, ¿cómo estás?"
    print(result.source_language)  # "English"

    # With source language and formality
    t = Translator(
        source_language="English",
        target_language="Japanese",
        formality="formal",
    )
    result = t.translate("Please review the attached document.")
    print(result.text)

    # Domain-specific translation
    t = Translator(target_language="French", domain="medical")
    result = t.translate("The patient presents with acute myocardial infarction.")
    print(result.text)
"""

from __future__ import annotations

import time
from typing import Any

from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.types import BaseModule
from aiforge.modules.translate.prompts import (
    DETECT_LANGUAGE,
    TRANSLATE_SYSTEM,
    TRANSLATE_USER,
    TRANSLATE_USER_AUTO_DETECT,
)
from aiforge.modules.translate.schemas import Formality, TranslateResponse


class Translator(BaseModule):
    """Cross-language text translator backed by an LLM.

    Args:
        target_language: The language to translate into (e.g. ``"Spanish"``).
        source_language: The source language.  ``"auto"`` (default) triggers
                         automatic detection.
        formality:       Tone control — ``"formal"``, ``"informal"``, or
                         ``"neutral"`` (default).
        domain:          Optional domain hint (e.g. ``"legal"``, ``"medical"``,
                         ``"technical"``) to improve terminology accuracy.
        temperature:     Sampling temperature for the LLM call.
        **kwargs:        Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        target_language: str,
        source_language: str = "auto",
        formality: Formality = "neutral",
        domain: str | None = None,
        temperature: float = 0.3,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not target_language or not target_language.strip():
            raise ValueError("target_language must be a non-empty string")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        self.target_language = target_language.strip()
        self.source_language = source_language.strip() if source_language else "auto"
        self.formality: Formality = formality
        self.domain = domain
        self.temperature = temperature
        self._system_prompt_override = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, *, system_prompt: str | None = None) -> TranslateResponse:
        """Translate *text* synchronously."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        total_in = total_out = 0
        total_cost = 0.0

        resolved_system = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("translate", {}).get("system"),
            default=TRANSLATE_SYSTEM,
        )

        detected_source = self.source_language
        if self.source_language.lower() == "auto":
            detected_source, d_in, d_out, d_cost, d_model = self._detect_language(text)
            total_in += d_in
            total_out += d_out
            total_cost += d_cost

        messages = self._build_messages(text, detected_source, resolved_system)
        llm = self._call_llm(messages, temperature=self.temperature)

        from aiforge.core.pricing import calculate_cost

        total_in += llm.input_tokens
        total_out += llm.output_tokens
        total_cost += calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)

        return TranslateResponse(
            text=llm.text.strip(),
            source_language=detected_source,
            target_language=self.target_language,
            formality=self.formality,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def atranslate(self, text: str, *, system_prompt: str | None = None) -> TranslateResponse:
        """Async version of :meth:`translate`."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        total_in = total_out = 0
        total_cost = 0.0

        resolved_system = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("translate", {}).get("system"),
            default=TRANSLATE_SYSTEM,
        )

        detected_source = self.source_language
        if self.source_language.lower() == "auto":
            detected_source, d_in, d_out, d_cost, d_model = await self._adetect_language(text)
            total_in += d_in
            total_out += d_out
            total_cost += d_cost

        messages = self._build_messages(text, detected_source, resolved_system)
        llm = await self._acall_llm(messages, temperature=self.temperature)

        from aiforge.core.pricing import calculate_cost

        total_in += llm.input_tokens
        total_out += llm.output_tokens
        total_cost += calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)

        return TranslateResponse(
            text=llm.text.strip(),
            source_language=detected_source,
            target_language=self.target_language,
            formality=self.formality,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # Pipeline interface -----------------------------------------------

    def run(self, text: str, **kwargs: Any) -> TranslateResponse:
        """Pipeline entry point (sync)."""
        sp = kwargs.pop("system_prompt", None)
        return self.translate(text, system_prompt=sp)

    async def arun(self, text: str, **kwargs: Any) -> TranslateResponse:
        """Pipeline entry point (async)."""
        sp = kwargs.pop("system_prompt", None)
        return await self.atranslate(text, system_prompt=sp)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _domain_instruction(self) -> str:
        """Build an optional domain-specific instruction line."""
        if self.domain:
            return f"Domain: {self.domain} — use appropriate terminology for this field."
        return ""

    def _build_messages(
        self,
        text: str,
        source_language: str,
        system: str,
    ) -> list[dict[str, str]]:
        """Construct the system + user message list for the LLM call."""
        domain_instruction = self._domain_instruction()
        if source_language.lower() == "auto":
            user_content = TRANSLATE_USER_AUTO_DETECT.format(
                target_language=self.target_language,
                formality=self.formality,
                domain_instruction=domain_instruction,
                text=text,
            )
        else:
            user_content = TRANSLATE_USER.format(
                source_language=source_language,
                target_language=self.target_language,
                formality=self.formality,
                domain_instruction=domain_instruction,
                text=text,
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _detect_language(self, text: str) -> tuple[str, int, int, float, str]:
        """Detect the source language synchronously.

        Returns ``(language, input_tokens, output_tokens, cost, model)``.
        """
        messages = [
            {"role": "system", "content": "You are a language identification expert."},
            {"role": "user", "content": DETECT_LANGUAGE.format(text=text[:500])},
        ]
        llm = self._call_llm(messages, temperature=0.0)
        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        detected = llm.text.strip().strip('"').strip("'")
        return detected, llm.input_tokens, llm.output_tokens, cost, llm.model

    async def _adetect_language(self, text: str) -> tuple[str, int, int, float, str]:
        """Async version of :meth:`_detect_language`."""
        messages = [
            {"role": "system", "content": "You are a language identification expert."},
            {"role": "user", "content": DETECT_LANGUAGE.format(text=text[:500])},
        ]
        llm = await self._acall_llm(messages, temperature=0.0)
        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        detected = llm.text.strip().strip('"').strip("'")
        return detected, llm.input_tokens, llm.output_tokens, cost, llm.model
