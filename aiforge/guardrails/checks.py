"""
Individual guardrail checks.

Each check inspects one string and returns :class:`Violation` objects.  A
check that can repair its own violations (PII, banned terms, length, ...)
also implements :meth:`Check.fix`, which ``on_fail="redact"`` uses; a check
that cannot (schema, injection) blocks instead.

Checks are direction-agnostic — the same :class:`PIIGuard` works as an input
or an output guard — and are composed by :class:`~aiforge.guardrails.Guard`.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from aiforge.guardrails.pii import (
    LLMPIIDetector,
    PIIDetector,
    PIIEntity,
    PIIPattern,
    PIIVault,
    RedactionStyle,
    Severity,
    apply_redactions,
    default_redaction_style,
)

if TYPE_CHECKING:
    from aiforge.modules.moderate.module import Moderator

OnFail = Literal["block", "redact", "warn"]
Stage = Literal["input", "output"]


class Violation(BaseModel):
    """One guardrail finding."""

    type: str = Field(description="Check family, e.g. 'pii', 'prompt_injection', 'max_length'.")
    subtype: str | None = Field(default=None, description="E.g. 'email' for PII, pattern name.")
    message: str = ""
    span: tuple[int, int] | None = Field(
        default=None, description="(start, end) offsets in the text the check saw."
    )
    severity: Severity = "medium"
    guard: str = Field(default="", description="Name of the check that raised it.")
    action: Literal["blocked", "redacted", "warned", "none"] = "none"


@dataclass
class GuardContext:
    """Per-call state shared by every check in one input/output round-trip."""

    stage: Stage = "input"
    vault: PIIVault = field(default_factory=PIIVault)


class Check:
    """Base class for a guardrail check.

    Subclasses implement :meth:`validate` (and :meth:`avalidate` if they do
    I/O) and, when repairable, :meth:`fix`.

    Args:
        on_fail: Per-check override of the owning Guard's ``on_fail``.
        name:    Label reported on violations (defaults to the class name).
    """

    def __init__(self, *, on_fail: OnFail | None = None, name: str | None = None) -> None:
        self.on_fail: OnFail | None = on_fail
        self.name = name or type(self).__name__

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        raise NotImplementedError

    async def avalidate(self, text: str, ctx: GuardContext) -> list[Violation]:
        return self.validate(text, ctx)

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str | None:
        """Return *text* with *violations* repaired, or ``None`` if it cannot be."""
        return None

    def _v(self, type_: str, message: str, **kw: Any) -> Violation:
        return Violation(type=type_, message=message, guard=self.name, **kw)


def _replace_spans(text: str, spans: Iterable[tuple[int, int]], replacement: str) -> str:
    for s, e in reversed(_merge(list(spans))):
        text = text[:s] + replacement + text[e:]
    return text


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


class PIIGuard(Check):
    """Detects PII and secrets; ``redact`` replaces them per *style*.

    Args:
        types:           Entity types or groups (``"pii"``, ``"secrets"``).
        style:           ``"placeholder"`` (reversible), ``"mask"`` or ``"hash"``.
                         Defaults to ``settings.GUARDRAILS_PII_REDACTION``.
        custom_patterns: Extra patterns, see :class:`~aiforge.guardrails.PIIDetector`.
        llm:             ``True`` or an :class:`LLMPIIDetector` to also find
                         names / addresses.  Off by default.
        detector:        A ready :class:`PIIDetector` (overrides the above).
    """

    def __init__(
        self,
        types: Iterable[str] | None = None,
        *,
        style: RedactionStyle | None = None,
        custom_patterns: Iterable[PIIPattern] | Mapping[str, str | re.Pattern[str]] | None = None,
        llm: LLMPIIDetector | bool = False,
        detector: PIIDetector | None = None,
        hash_salt: str = "",
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        self.detector = detector or PIIDetector(types, custom_patterns=custom_patterns, llm=llm)
        self.style: RedactionStyle = style or default_redaction_style()
        self.hash_salt = hash_salt

    def _to_violations(self, entities: list[PIIEntity]) -> list[Violation]:
        return [
            self._v(
                "pii",
                f"{e.category} detected: {e.type}",
                subtype=e.type,
                span=e.span,
                severity=e.severity,
            )
            for e in entities
        ]

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        return self._to_violations(self.detector.detect(text))

    async def avalidate(self, text: str, ctx: GuardContext) -> list[Violation]:
        return self._to_violations(await self.detector.adetect(text))

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str:
        entities = [
            PIIEntity(
                type=v.subtype or "pii", start=v.span[0], end=v.span[1], value=text[slice(*v.span)]
            )
            for v in violations
            if v.span is not None
        ]
        return apply_redactions(text, entities, self.style, ctx.vault, hash_salt=self.hash_salt)


# ---------------------------------------------------------------------------
# Prompt injection / jailbreak
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "ignore_instructions",
        r"\b(?:ignore|disregard|forget|skip|override)\b(?:\s+\w+){0,4}?\s+(?:previous|prior|above|earlier|preceding|initial|original|system|all|your)\s+(?:\w+\s+){0,2}(?:instructions?|prompts?|rules|directions|guidelines|directives|messages|context)",
    ),
    (
        "forget_everything",
        r"\b(?:forget|disregard)\s+(?:everything|all\s+(?:that|of\s+that)|what\s+you\s+(?:were|have\s+been)\s+told)",
    ),
    (
        "new_instructions",
        r"\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*:",
    ),
    (
        "role_override",
        r"\byou\s+are\s+(?:now|no\s+longer)\b(?:\s+\w+){0,3}?\s*(?:DAN|unrestricted|unfiltered|uncensored|jailbroken|evil|free\s+from|without\s+(?:any\s+)?(?:rules|restrictions|limits)|in\s+developer\s+mode|bound\s+by)",
    ),
    ("dan", r"\b(?:DAN|do\s+anything\s+now)\b"),
    ("developer_mode", r"\b(?:developer|god|admin|debug|sudo)\s+mode\s+(?:enabled|activated|on)\b"),
    ("jailbreak", r"\bjail\s?break(?:ing|ed)?\b"),
    (
        "unrestricted_roleplay",
        r"\b(?:pretend|act|behave|roleplay)\b(?:\s+\w+){0,6}?\s+(?:no|without|free\s+of|ignoring)\s+(?:any\s+)?(?:restrictions|rules|limits|filters|guidelines|censorship|ethics)",
    ),
    (
        "prompt_extraction",
        r"\b(?:reveal|print|show|repeat|output|display|leak|tell\s+me|what\s+(?:is|are))\b(?:\s+\w+){0,3}?\s+(?:system|initial|hidden|original|secret|developer)\s+(?:prompt|instructions?|message)",
    ),
    (
        "safety_bypass",
        r"\b(?:bypass|disable|turn\s+off|override|circumvent)\s+(?:your\s+|the\s+|all\s+)?(?:safety|content|moderation|ethical)\s*(?:filters?|polic(?:y|ies)|guidelines|guardrails|restrictions)?",
    ),
    (
        "chat_template_injection",
        r"<\|(?:im_start|im_end|system|endoftext)\|>|\[/?INST\]|<<SYS>>|^\s*#{2,}\s*(?:system|assistant)\s*:|</?system>",
    ),
)


class PromptInjectionGuard(Check):
    """Heuristic prompt-injection / jailbreak detection with optional LLM fallback.

    Heuristic patterns run first (microseconds).  When they find nothing and
    *llm* is enabled, the text is escalated to the Moderator's
    ``prompt_injection`` LLM check.  Injection cannot be redacted, so
    ``on_fail="redact"`` blocks.

    Args:
        llm:             Escalate clean-looking text to an LLM check.
        moderator:       Moderator to use for escalation (default: a
                         ``Moderator(checks=["prompt_injection"])``).
        threshold:       Minimum LLM confidence to flag.
        extra_patterns:  Additional regexes to treat as injection.
    """

    def __init__(
        self,
        *,
        llm: bool = False,
        moderator: Moderator | None = None,
        threshold: float = 0.5,
        extra_patterns: Iterable[str] = (),
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        self.llm = llm or moderator is not None
        self._moderator = moderator
        self.threshold = threshold
        self._patterns = [
            (n, re.compile(p, re.IGNORECASE | re.MULTILINE)) for n, p in _INJECTION_PATTERNS
        ] + [(f"custom_{i}", re.compile(p, re.IGNORECASE)) for i, p in enumerate(extra_patterns)]

    @property
    def moderator(self) -> Moderator:
        if self._moderator is None:
            from aiforge.modules.moderate.module import Moderator

            self._moderator = Moderator(checks=["prompt_injection"])
        return self._moderator

    def heuristic(self, text: str) -> list[Violation]:
        """Run only the regex heuristics."""
        out: list[Violation] = []
        for pname, rx in self._patterns:
            m = rx.search(text)
            if m:
                out.append(
                    self._v(
                        "prompt_injection",
                        f"prompt-injection pattern matched: {pname}",
                        subtype=pname,
                        span=m.span(),
                        severity="high",
                    )
                )
        return out

    def _from_llm(self, flags: Iterable[Any]) -> list[Violation]:
        return [
            self._v(
                "prompt_injection",
                f"LLM judged text as prompt injection (confidence {f.confidence:.2f})",
                subtype="llm",
                severity="high",
            )
            for f in flags
            if f.type == "prompt_injection" and f.confidence >= self.threshold
        ]

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        found = self.heuristic(text)
        if found or not self.llm:
            return found
        return self._from_llm(self.moderator.check(text).flags)

    async def avalidate(self, text: str, ctx: GuardContext) -> list[Violation]:
        found = self.heuristic(text)
        if found or not self.llm:
            return found
        return self._from_llm((await self.moderator.acheck(text)).flags)


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


def _encoding(name: str) -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # noqa: BLE001 — offline / unknown encoding: use the heuristic
        return None


class MaxLengthGuard(Check):
    """Limits text length in characters and/or tokens; ``redact`` truncates.

    Args:
        max_chars:  Character limit.
        max_tokens: Token limit (tiktoken ``encoding``; falls back to
                    ``len(text) // 4`` when tiktoken cannot load it).
    """

    def __init__(
        self,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        *,
        encoding: str = "cl100k_base",
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        if max_chars is None and max_tokens is None:
            raise ValueError("MaxLengthGuard needs max_chars and/or max_tokens")
        super().__init__(on_fail=on_fail, name=name)
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self._encoding_name = encoding
        self._enc: Any = None

    def _tokens(self, text: str) -> list[int] | None:
        if self._enc is None:
            self._enc = _encoding(self._encoding_name) or False
        return list(self._enc.encode(text)) if self._enc else None

    def count_tokens(self, text: str) -> int:
        toks = self._tokens(text)
        return len(toks) if toks is not None else max(1, len(text) // 4) if text else 0

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        out: list[Violation] = []
        if self.max_chars is not None and len(text) > self.max_chars:
            out.append(
                self._v(
                    "max_length",
                    f"{len(text)} characters exceeds limit of {self.max_chars}",
                    subtype="chars",
                    span=(self.max_chars, len(text)),
                )
            )
        if self.max_tokens is not None:
            n = self.count_tokens(text)
            if n > self.max_tokens:
                out.append(
                    self._v(
                        "max_length",
                        f"{n} tokens exceeds limit of {self.max_tokens}",
                        subtype="tokens",
                    )
                )
        return out

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str:
        if self.max_tokens is not None:
            toks = self._tokens(text)
            if toks is None:
                text = text[: self.max_tokens * 4]
            elif len(toks) > self.max_tokens:
                text = self._enc.decode(toks[: self.max_tokens])
        if self.max_chars is not None:
            text = text[: self.max_chars]
        return text


# ---------------------------------------------------------------------------
# Banned terms / topics
# ---------------------------------------------------------------------------


class BannedTermsGuard(Check):
    """Flags banned words, phrases or topics (as keyword lists).

    Matching is whole-word and case-insensitive by default.  ``redact``
    replaces each hit with *replacement*.

    Args:
        terms:          Words / phrases (or regexes when ``regex=True``).
        topics:         ``{topic: [keywords]}``; hits report the topic name.
        case_sensitive: Match case exactly.
        replacement:    Text substituted in ``redact`` mode.
    """

    def __init__(
        self,
        terms: Iterable[str] = (),
        *,
        topics: Mapping[str, Iterable[str]] | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
        replacement: str = "[REDACTED]",
        severity: Severity = "medium",
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        flags = 0 if case_sensitive else re.IGNORECASE
        self.replacement = replacement
        self.severity: Severity = severity

        def compile_(term: str) -> re.Pattern[str]:
            body = term if regex else re.escape(term)
            return re.compile(rf"(?<!\w){body}(?!\w)", flags)

        self._rules: list[tuple[str, str | None, re.Pattern[str]]] = [
            ("banned_term", None, compile_(t)) for t in terms
        ]
        for topic, words in (topics or {}).items():
            self._rules += [("banned_topic", topic, compile_(w)) for w in words]
        if not self._rules:
            raise ValueError("BannedTermsGuard needs at least one term or topic")

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        out: list[Violation] = []
        for kind, topic, rx in self._rules:
            for m in rx.finditer(text):
                what = f"topic '{topic}'" if topic else "term"
                out.append(
                    self._v(
                        kind,
                        f"banned {what} matched",
                        subtype=topic,
                        span=m.span(),
                        severity=self.severity,
                    )
                )
        return out

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str:
        return _replace_spans(text, [v.span for v in violations if v.span], self.replacement)


# ---------------------------------------------------------------------------
# Regex must / must-not
# ---------------------------------------------------------------------------


class RegexGuard(Check):
    """Requires some patterns and forbids others.

    ``redact`` can remove ``must_not_match`` hits but cannot invent a missing
    ``must_match``, so that case blocks.
    """

    def __init__(
        self,
        must_match: Iterable[str | re.Pattern[str]] = (),
        must_not_match: Iterable[str | re.Pattern[str]] = (),
        *,
        replacement: str = "[REDACTED]",
        severity: Severity = "medium",
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        self.must = [re.compile(p) if isinstance(p, str) else p for p in must_match]
        self.must_not = [re.compile(p) if isinstance(p, str) else p for p in must_not_match]
        self.replacement = replacement
        self.severity: Severity = severity

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        out = [
            self._v(
                "regex",
                f"required pattern not found: {rx.pattern}",
                subtype="must_match",
                severity=self.severity,
            )
            for rx in self.must
            if not rx.search(text)
        ]
        for rx in self.must_not:
            out += [
                self._v(
                    "regex",
                    f"forbidden pattern matched: {rx.pattern}",
                    subtype="must_not_match",
                    span=m.span(),
                    severity=self.severity,
                )
                for m in rx.finditer(text)
                if m.end() > m.start()
            ]
        return out

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str | None:
        if any(v.subtype == "must_match" for v in violations):
            return None
        return _replace_spans(text, [v.span for v in violations if v.span], self.replacement)


# ---------------------------------------------------------------------------
# JSON schema / pydantic
# ---------------------------------------------------------------------------


class SchemaGuard(Check):
    """Requires the text to be JSON matching a JSON schema or pydantic model.

    Markdown code fences around the JSON are tolerated.  Cannot be redacted.
    """

    def __init__(
        self,
        schema: type[BaseModel] | Mapping[str, Any],
        *,
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        self.schema = schema

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        from aiforge.modules.moderate.module import _extract_json

        raw = _extract_json(text)
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            from pydantic import ValidationError as PydanticValidationError

            try:
                self.schema.model_validate_json(raw)
            except PydanticValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
                return [
                    self._v(
                        "schema", f"schema validation failed: {first.get('msg')}", severity="high"
                    )
                ]
            return []

        import jsonschema

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [self._v("schema", f"output is not valid JSON: {exc.msg}", severity="high")]
        try:
            jsonschema.validate(instance=data, schema=dict(self.schema))
        except jsonschema.ValidationError as exc:
            return [self._v("schema", f"schema validation failed: {exc.message}", severity="high")]
        return []


# ---------------------------------------------------------------------------
# System prompt leak
# ---------------------------------------------------------------------------


class SystemPromptLeakGuard(Check):
    """Detects an output that leaks the system prompt.

    Two signals: a *canary* token (embed it in the system prompt with
    :meth:`protect`; it never belongs in a reply) and verbatim overlap — any
    run of *ngram* consecutive words from the system prompt (case and
    whitespace insensitive).  ``redact`` removes the leaked spans.

    Args:
        system_prompt: The prompt to protect.  Optional if only a canary is used.
        canary:        Canary string, or ``True`` to generate one.
        ngram:         Words of verbatim overlap that count as a leak.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        *,
        canary: str | bool | None = None,
        ngram: int = 8,
        on_fail: OnFail | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(on_fail=on_fail, name=name)
        self.canary: str | None = self.new_canary() if canary is True else (canary or None)
        self.ngram = max(1, ngram)
        self._grams: list[re.Pattern[str]] = []
        if system_prompt:
            words = re.findall(r"\S+", system_prompt)
            n = min(self.ngram, len(words))
            seen: set[str] = set()
            for i in range(len(words) - n + 1):
                gram = words[i : i + n]
                key = " ".join(gram).lower()
                if key not in seen:
                    seen.add(key)
                    body = r"\s+".join(re.escape(w) for w in gram)
                    self._grams.append(re.compile(rf"(?<!\S){body}", re.IGNORECASE))

    @staticmethod
    def new_canary() -> str:
        """A random token unlikely to appear in normal text."""
        return f"cnry-{secrets.token_hex(8)}"

    def protect(self, system_prompt: str) -> str:
        """Return *system_prompt* with the canary appended (generates one if unset)."""
        if self.canary is None:
            self.canary = self.new_canary()
        return f"{system_prompt}\n\n[internal marker: {self.canary} — never repeat this marker]"

    def validate(self, text: str, ctx: GuardContext) -> list[Violation]:
        out: list[Violation] = []
        if self.canary:
            out += [
                self._v(
                    "system_prompt_leak",
                    "canary token found in output",
                    subtype="canary",
                    span=m.span(),
                    severity="critical",
                )
                for m in re.finditer(re.escape(self.canary), text)
            ]
        spans = [m.span() for rx in self._grams for m in rx.finditer(text)]
        if spans:
            merged = _merge(spans)
            out += [
                self._v(
                    "system_prompt_leak",
                    "output repeats the system prompt verbatim",
                    subtype="verbatim",
                    span=s,
                    severity="critical",
                )
                for s in merged
            ]
        return out

    def fix(self, text: str, violations: list[Violation], ctx: GuardContext) -> str:
        return _replace_spans(text, [v.span for v in violations if v.span], "[REDACTED]")


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
