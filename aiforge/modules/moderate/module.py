"""
Moderator module — content safety, PII detection, and prompt injection checks.

Usage::

    from aiforge import Moderator

    m = Moderator(checks=["pii", "toxicity", "prompt_injection"])

    result = m.check("My email is foo@example.com")
    print(result.flagged)   # True
    print(result.flags)     # [ModerationFlag(type='pii', subtype='email', ...)]

    cleaned = m.redact("Call me at 555-123-4567")
    print(cleaned.text)     # "Call me at [PHONE]"
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.types import BaseModule
from aiforge.modules.moderate.prompts import (
    INJECTION_SYSTEM,
    INJECTION_USER,
    TOXICITY_SYSTEM,
    TOXICITY_USER,
)
from aiforge.modules.moderate.schemas import (
    ModerationFlag,
    ModerateResponse,
    RedactResponse,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# US + international phone — runs last to avoid over-matching digits in SSN/cards
_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+?1[-.\s]?)?                # optional US/CA country code
        (?:\(\d{3}\)|\d{3})             # area code
        [-.\s]?\d{3}[-.\s]?\d{4}       # local number
    |
        \+\d{1,3}[-.\s]\d{1,4}[-.\s]\d{1,4}[-.\s]\d{1,9}  # international
    )
    """,
    re.VERBOSE,
)

# Visa (13/16), Mastercard (16), Amex (15), Discover (16) — compact or spaced
_CARD_RE = re.compile(
    r"""
    \b
    (?:
        4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}   # Visa 16
        |4\d{12}                                    # Visa 13
        |5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}  # Mastercard
        |3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}          # Amex
        |6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}  # Discover
    )
    \b
    """,
    re.VERBOSE,
)

_REDACT: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
}

_VALID_CHECKS = frozenset({"pii", "toxicity", "prompt_injection"})
_VALID_PII_TYPES = frozenset({"email", "phone", "ssn", "credit_card"})


# ---------------------------------------------------------------------------
# Luhn checksum
# ---------------------------------------------------------------------------

def _luhn_ok(number: str) -> bool:
    """Return True if *number* (digits + optional separators) passes Luhn."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# JSON extraction helper (C1 fix: handle markdown-fenced LLM responses)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence line (```json or ```)
        text = text.split("\n", 1)[-1]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip().rsplit("```", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def _apply_redactions(text: str, flags: list[ModerationFlag]) -> str:
    """Replace each PII span with its placeholder label.

    Deduplicates identical spans so each unique value is processed once
    (str.replace replaces all occurrences by itself).
    """
    pii = [f for f in flags if f.subtype in _REDACT and f.span]
    # Longest match first so shorter overlapping spans don't leave remnants
    # Deduplicate by span value to avoid redundant replace() calls
    seen: set[str] = set()
    unique_pii: list[ModerationFlag] = []
    for f in sorted(pii, key=lambda f: len(f.span or ""), reverse=True):
        if f.span not in seen:
            seen.add(f.span)  # type: ignore[arg-type]
            unique_pii.append(f)
    for flag in unique_pii:
        placeholder = _REDACT.get(flag.subtype or "", "[REDACTED]")
        text = text.replace(flag.span, placeholder)  # type: ignore[arg-type]
    return text


# ---------------------------------------------------------------------------
# Moderator class
# ---------------------------------------------------------------------------

class Moderator(BaseModule):
    """Content safety, PII detection, and prompt injection scanner.

    Args:
        checks:    Which checks to run. Subset of
                   ``["pii", "toxicity", "prompt_injection"]``.
                   Defaults to all three.
        pii_types: PII sub-types to detect when ``"pii"`` is enabled.
                   Subset of ``["email", "phone", "ssn", "credit_card"]``.
                   Defaults to all four.
        **kwargs:  Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        checks: list[str] | None = None,
        pii_types: list[str] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checks: list[str] = (
            checks if checks is not None else ["pii", "toxicity", "prompt_injection"]
        )
        self.pii_types: list[str] = (
            pii_types if pii_types is not None else ["email", "phone", "ssn", "credit_card"]
        )
        self._system_prompt_override: str | None = system_prompt
        bad_checks = set(self.checks) - _VALID_CHECKS
        if bad_checks:
            raise ValueError(f"Unknown check types: {bad_checks}. Valid: {_VALID_CHECKS}")
        bad_pii = set(self.pii_types) - _VALID_PII_TYPES
        if bad_pii:
            raise ValueError(f"Unknown PII types: {bad_pii}. Valid: {_VALID_PII_TYPES}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str, *, system_prompt: str | None = None) -> ModerateResponse:
        """Scan *text* synchronously and return a :class:`ModerateResponse`.

        PII checks are regex-only (< 1 ms).  Toxicity and prompt-injection
        checks make one LLM call each.
        """
        t0 = time.monotonic()
        flags: list[ModerationFlag] = []
        input_tokens = output_tokens = 0
        model = ""

        if "pii" in self.checks:
            flags.extend(self._run_pii(text))

        if "toxicity" in self.checks:
            mod_cfg = self.settings.CUSTOM_PROMPTS.get("moderate", {})
            tox_sys = resolve_prompt(
                call_override=system_prompt,
                instance_override=self._system_prompt_override,
                config_override=mod_cfg.get("toxicity_system") or mod_cfg.get("system"),
                default=TOXICITY_SYSTEM,
            )
            llm = self._call_llm(
                [
                    {"role": "system", "content": tox_sys},
                    {"role": "user", "content": TOXICITY_USER.format(text=text)},
                ],
                temperature=0.0,
            )
            input_tokens += llm.input_tokens
            output_tokens += llm.output_tokens
            model = llm.model
            flags.extend(self._parse_toxicity(llm.text))

        if "prompt_injection" in self.checks:
            mod_cfg = self.settings.CUSTOM_PROMPTS.get("moderate", {})
            inj_sys = resolve_prompt(
                call_override=system_prompt,
                instance_override=self._system_prompt_override,
                config_override=mod_cfg.get("injection_system") or mod_cfg.get("system"),
                default=INJECTION_SYSTEM,
            )
            llm = self._call_llm(
                [
                    {"role": "system", "content": inj_sys},
                    {"role": "user", "content": INJECTION_USER.format(text=text)},
                ],
                temperature=0.0,
            )
            input_tokens += llm.input_tokens
            output_tokens += llm.output_tokens
            model = model or llm.model
            flags.extend(self._parse_injection(llm.text))

        return self._build_response(flags, input_tokens, output_tokens, model, t0)

    async def acheck(self, text: str, *, system_prompt: str | None = None) -> ModerateResponse:
        """Async version of :meth:`check`."""
        t0 = time.monotonic()
        flags: list[ModerationFlag] = []
        input_tokens = output_tokens = 0
        model = ""

        if "pii" in self.checks:
            flags.extend(self._run_pii(text))

        if "toxicity" in self.checks:
            mod_cfg = self.settings.CUSTOM_PROMPTS.get("moderate", {})
            tox_sys = resolve_prompt(
                call_override=system_prompt,
                instance_override=self._system_prompt_override,
                config_override=mod_cfg.get("toxicity_system") or mod_cfg.get("system"),
                default=TOXICITY_SYSTEM,
            )
            llm = await self._acall_llm(
                [
                    {"role": "system", "content": tox_sys},
                    {"role": "user", "content": TOXICITY_USER.format(text=text)},
                ],
                temperature=0.0,
            )
            input_tokens += llm.input_tokens
            output_tokens += llm.output_tokens
            model = llm.model
            flags.extend(self._parse_toxicity(llm.text))

        if "prompt_injection" in self.checks:
            mod_cfg = self.settings.CUSTOM_PROMPTS.get("moderate", {})
            inj_sys = resolve_prompt(
                call_override=system_prompt,
                instance_override=self._system_prompt_override,
                config_override=mod_cfg.get("injection_system") or mod_cfg.get("system"),
                default=INJECTION_SYSTEM,
            )
            llm = await self._acall_llm(
                [
                    {"role": "system", "content": inj_sys},
                    {"role": "user", "content": INJECTION_USER.format(text=text)},
                ],
                temperature=0.0,
            )
            input_tokens += llm.input_tokens
            output_tokens += llm.output_tokens
            model = model or llm.model
            flags.extend(self._parse_injection(llm.text))

        return self._build_response(flags, input_tokens, output_tokens, model, t0)

    def redact(self, text: str) -> RedactResponse:
        """Redact PII from *text*.  Regex-only — no LLM calls.

        Only runs PII detection when ``"pii"`` is in the configured checks.
        Returns the original text unchanged if PII checks are disabled.
        """
        t0 = time.monotonic()
        flags = self._run_pii(text) if "pii" in self.checks else []
        redacted = _apply_redactions(text, flags)
        return RedactResponse(
            text=redacted,
            redactions=flags,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aredact(self, text: str) -> RedactResponse:
        """Async version of :meth:`redact` (no I/O — runs synchronously)."""
        return self.redact(text)

    def run(self, text: str, **kwargs: Any) -> ModerateResponse:
        """Pipeline-compatible entry point — delegates to :meth:`check`."""
        sp = kwargs.pop("system_prompt", None)
        return self.check(text, system_prompt=sp)

    async def arun(self, text: str, **kwargs: Any) -> ModerateResponse:
        """Async pipeline entry point — delegates to :meth:`acheck`."""
        sp = kwargs.pop("system_prompt", None)
        return await self.acheck(text, system_prompt=sp)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pii(self, text: str) -> list[ModerationFlag]:
        """Run enabled regex PII detectors with position-based overlap suppression.

        Detectors run in priority order (email → SSN → credit_card → phone).
        Phone runs last and uses positional overlap checking against previously
        matched spans so credit card / SSN digit sequences are never double-flagged.
        """
        flags: list[ModerationFlag] = []
        # Track (start, end) character positions of already-matched spans
        occupied: list[tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            return any(start < e and end > s for s, e in occupied)

        if "email" in self.pii_types:
            for m in _EMAIL_RE.finditer(text):
                if not _overlaps(m.start(), m.end()):
                    flags.append(
                        ModerationFlag(type="pii", subtype="email", span=m.group(), confidence=1.0)
                    )
                    occupied.append((m.start(), m.end()))

        if "ssn" in self.pii_types:
            for m in _SSN_RE.finditer(text):
                if not _overlaps(m.start(), m.end()):
                    flags.append(
                        ModerationFlag(type="pii", subtype="ssn", span=m.group(), confidence=1.0)
                    )
                    occupied.append((m.start(), m.end()))

        if "credit_card" in self.pii_types:
            for m in _CARD_RE.finditer(text):
                if _luhn_ok(m.group()) and not _overlaps(m.start(), m.end()):
                    flags.append(
                        ModerationFlag(
                            type="pii", subtype="credit_card", span=m.group().strip(), confidence=1.0
                        )
                    )
                    occupied.append((m.start(), m.end()))

        if "phone" in self.pii_types:
            for m in _PHONE_RE.finditer(text):
                span = m.group().strip()
                digits = re.sub(r"\D", "", span)
                if len(digits) < 7:
                    continue
                # Use positional overlap (not string-containment) to avoid
                # false suppression when card and phone share digit substrings
                if not _overlaps(m.start(), m.end()):
                    flags.append(
                        ModerationFlag(type="pii", subtype="phone", span=span, confidence=1.0)
                    )
                    occupied.append((m.start(), m.end()))

        return flags

    @staticmethod
    def _parse_toxicity(llm_text: str) -> list[ModerationFlag]:
        try:
            data = json.loads(_extract_json(llm_text))
        except json.JSONDecodeError:
            return []
        if data.get("is_toxic"):
            return [ModerationFlag(type="toxicity", confidence=float(data.get("confidence", 1.0)))]
        return []

    @staticmethod
    def _parse_injection(llm_text: str) -> list[ModerationFlag]:
        try:
            data = json.loads(_extract_json(llm_text))
        except json.JSONDecodeError:
            return []
        if data.get("is_injection"):
            return [
                ModerationFlag(
                    type="prompt_injection", confidence=float(data.get("confidence", 1.0))
                )
            ]
        return []

    def _build_response(
        self,
        flags: list[ModerationFlag],
        input_tokens: int,
        output_tokens: int,
        model: str,
        t0: float,
    ) -> ModerateResponse:
        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(model, input_tokens, output_tokens) if model else 0.0
        return ModerateResponse(
            flagged=bool(flags),
            flags=flags,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
