"""
PII and secret detection with reversible redaction.

Detection is regex-first with a validation step for every pattern that has
one (Luhn for cards, mod-97 for IBANs, :mod:`ipaddress` for IPs, Shannon
entropy for generic tokens), so the common false positives — order numbers,
version strings, timestamps — are rejected cheaply.  An optional LLM pass
(:class:`LLMPIIDetector`) finds what regexes cannot: person names and postal
addresses.

Usage::

    from aiforge.guardrails import PIIDetector, redact_pii, rehydrate

    entities = PIIDetector().detect("Mail jane@acme.io, card 4111 1111 1111 1111")
    result = redact_pii("Mail jane@acme.io")      # placeholder style by default
    result.text                                    # "Mail [EMAIL_1]"
    rehydrate("Sent to [EMAIL_1]", result.mapping) # "Sent to jane@acme.io"
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aiforge.core.provider import LLMProvider

RedactionStyle = Literal["mask", "placeholder", "hash"]
Severity = Literal["low", "medium", "high", "critical"]


class PIIEntity(BaseModel):
    """One detected PII value or secret."""

    type: str = Field(description="Entity type, e.g. 'email', 'credit_card', 'aws_access_key'.")
    category: str = Field(default="pii", description="'pii', 'secret' or 'custom'.")
    start: int = Field(description="Start offset (inclusive) in the scanned text.")
    end: int = Field(description="End offset (exclusive) in the scanned text.")
    value: str = Field(description="The matched text.")
    severity: Severity = "medium"
    source: str = Field(default="regex", description="'regex' or 'llm'.")

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def luhn_valid(number: str) -> bool:
    """Return True if the digits of *number* pass the Luhn checksum (13-19 digits)."""
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_CARD_PREFIX_RE = re.compile(
    r"^(?:4|5[1-5]|2(?:2[2-9]|[3-6]\d|7[01]|720)|3[47]|3[068]|35|6(?:011|5|4[4-9]|2))"
)


def _card_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return bool(_CARD_PREFIX_RE.match(digits)) and luhn_valid(digits)


def iban_valid(value: str) -> bool:
    """Return True if *value* is a structurally valid IBAN (ISO 13616 mod-97 check)."""
    s = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def shannon_entropy(value: str) -> float:
    """Shannon entropy of *value* in bits per character."""
    if not value:
        return 0.0
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in Counter(value).values())


# Words that, right before a dotted quad, mean it is a version not an address.
_VERSION_CONTEXT_RE = re.compile(
    r"(?:\bv|\bver(?:sion)?|\brelease|\bbuild|@|==|>=|<=|~=)\s*$", re.I
)


def _ipv4_ok(value: str, text: str, start: int) -> bool:
    if _VERSION_CONTEXT_RE.search(text[max(0, start - 12) : start]):
        return False
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _ipv6_ok(value: str, text: str, start: int) -> bool:
    if sum(c.isalnum() for c in value) < 2:
        return False
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return True


def _phone_ok(value: str, text: str, start: int) -> bool:
    digits = re.sub(r"\D", "", value)
    has_plus = value.lstrip().startswith("+")
    if not (8 if has_plus else 10) <= len(digits) <= 15:
        return False
    # Reject ISO dates (2024-01-15) that happen to have enough digits around them.
    return not re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value)


def _ssn_ok(value: str, text: str, start: int) -> bool:
    area, group, serial = re.split(r"[-\s]", value)
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def _entropy_ok(threshold: float) -> Callable[[str, str, int], bool]:
    def check(value: str, text: str, start: int) -> bool:
        return shannon_entropy(value) >= threshold and bool(re.search(r"\d", value))

    return check


# ---------------------------------------------------------------------------
# Pattern table
# ---------------------------------------------------------------------------

Validator = Callable[[str, str, int], bool]


@dataclass(frozen=True)
class PIIPattern:
    """A named detection pattern.

    Args:
        name:      Entity type reported on matches (also the placeholder label).
        pattern:   Regex string or compiled pattern.  If it has a group named
                   ``value``, only that group is reported (useful for
                   ``key=VALUE`` shapes); otherwise the whole match.
        validator: Optional ``(value, text, start) -> bool`` that must accept
                   the match.
        severity:  Severity attached to matches.
        category:  ``"pii"``, ``"secret"`` or ``"custom"``.
    """

    name: str
    pattern: str | re.Pattern[str]
    validator: Validator | None = None
    severity: Severity = "medium"
    category: str = "custom"
    flags: int = 0
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        compiled = (
            self.pattern
            if isinstance(self.pattern, re.Pattern)
            else re.compile(self.pattern, self.flags)
        )
        object.__setattr__(self, "_compiled", compiled)

    def finditer(self, text: str) -> Iterable[tuple[int, int, str]]:
        """Yield validated ``(start, end, value)`` triples."""
        for m in self._compiled.finditer(text):
            grp = "value" if "value" in self._compiled.groupindex else 0
            start, end = m.span(grp)
            value = m.group(grp)
            if start < 0 or not value:
                continue
            if self.validator is None or self.validator(value, text, start):
                yield start, end, value


# Order is priority: earlier patterns win when spans overlap.
BUILTIN_PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern(
        "private_key",
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]+?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "aws_access_key",
        r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "aws_secret_key",
        r"(?i:aws_?secret_?(?:access_?)?key)\s*[\"']?\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "github_token",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "api_key",  # OpenAI / Anthropic style: sk-..., sk-proj-..., sk-ant-...
        r"(?<![\w-])sk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_\-]{20,}",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "google_api_key",
        r"\bAIza[0-9A-Za-z_\-]{35}(?![\w-])",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "slack_token",
        r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
        severity="critical",
        category="secret",
    ),
    PIIPattern(
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        severity="high",
        category="secret",
    ),
    PIIPattern(
        "bearer_token",
        r"(?i:\bbearer)\s+(?P<value>[A-Za-z0-9\-._~+/]{20,}=*)",
        validator=_entropy_ok(3.5),
        severity="high",
        category="secret",
    ),
    PIIPattern(
        "generic_secret",
        r"(?i:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|password|passwd|secret|token))"
        r"[\"']?\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9_\-./+=!@#$%^&*]{16,})",
        validator=_entropy_ok(3.0),
        severity="high",
        category="secret",
    ),
    PIIPattern(
        "email",
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b",
        severity="medium",
        category="pii",
    ),
    PIIPattern(
        "iban",
        r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b",
        validator=lambda v, t, s: iban_valid(v),
        severity="high",
        category="pii",
    ),
    PIIPattern(
        "credit_card",
        r"(?<![\d.\-])\d(?:[ \-]?\d){12,18}(?![\d\-])",
        validator=lambda v, t, s: _card_ok(v),
        severity="critical",
        category="pii",
    ),
    PIIPattern(
        "ssn",
        r"(?<![\d\-])\d{3}[- ]\d{2}[- ]\d{4}(?![\d\-])",
        validator=_ssn_ok,
        severity="high",
        category="pii",
    ),
    PIIPattern(
        "ip_address",
        r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)",
        validator=_ipv4_ok,
        severity="low",
        category="pii",
    ),
    PIIPattern(
        "ip_address",
        r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])",
        validator=_ipv6_ok,
        severity="low",
        category="pii",
    ),
    PIIPattern(
        "phone",
        # International (+CC, lenient grouping) or national (area code + two groups).
        r"(?<![\w+])(?<!\d[\s.\-])"
        r"(?:\+\d{1,3}(?:[\s.\-]?\(?\d{1,4}\)?){2,6}"
        r"|(?:\(\d{2,4}\)\s?|\d{2,4}[\s.\-])\d{3,4}[\s.\-]\d{3,4})"
        r"(?![\w\-]|[\s.]\d)",
        validator=_phone_ok,
        severity="medium",
        category="pii",
    ),
)

#: Every built-in entity type.
PII_TYPES: frozenset[str] = frozenset(p.name for p in BUILTIN_PATTERNS)
#: Type groups accepted anywhere a type list is accepted.
PII_GROUPS: dict[str, frozenset[str]] = {
    "pii": frozenset(p.name for p in BUILTIN_PATTERNS if p.category == "pii"),
    "secrets": frozenset(p.name for p in BUILTIN_PATTERNS if p.category == "secret"),
    "all": PII_TYPES,
}


def _expand_types(types: Iterable[str] | None) -> frozenset[str]:
    if types is None:
        return PII_TYPES
    out: set[str] = set()
    for t in types:
        if t in PII_GROUPS:
            out |= PII_GROUPS[t]
        elif t in PII_TYPES:
            out.add(t)
        else:
            valid = sorted(PII_TYPES | PII_GROUPS.keys())
            raise ValueError(f"Unknown PII type {t!r}. Valid: {valid}")
    return frozenset(out)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PIIDetector:
    """Regex + validation PII / secret detector.

    Args:
        types:           Built-in entity types or groups (``"pii"``,
                         ``"secrets"``, ``"all"``) to detect.  Default: all.
        custom_patterns: Extra patterns, as :class:`PIIPattern` objects or a
                         ``{name: regex}`` mapping.  Custom patterns take
                         priority over built-ins on overlap.
        llm:             Optional :class:`LLMPIIDetector` (or ``True`` to build
                         a default one) for names / addresses.  Off by default.
    """

    def __init__(
        self,
        types: Iterable[str] | None = None,
        *,
        custom_patterns: Iterable[PIIPattern] | Mapping[str, str | re.Pattern[str]] | None = None,
        llm: LLMPIIDetector | bool | None = None,
    ) -> None:
        enabled = _expand_types(types)
        custom: list[PIIPattern] = []
        if isinstance(custom_patterns, Mapping):
            custom = [PIIPattern(name, rx) for name, rx in custom_patterns.items()]
        elif custom_patterns is not None:
            custom = list(custom_patterns)
        self.patterns: list[PIIPattern] = custom + [
            p for p in BUILTIN_PATTERNS if p.name in enabled
        ]
        self.llm: LLMPIIDetector | None = LLMPIIDetector() if llm is True else (llm or None)

    def _regex(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for pat in self.patterns:
            for start, end, value in pat.finditer(text):
                found.append(
                    PIIEntity(
                        type=pat.name,
                        category=pat.category,
                        start=start,
                        end=end,
                        value=value,
                        severity=pat.severity,
                    )
                )
        return found

    def detect(self, text: str) -> list[PIIEntity]:
        """Return non-overlapping entities in *text*, ordered by position."""
        found = self._regex(text)
        if self.llm is not None:
            found.extend(self.llm.detect(text))
        return _resolve_overlaps(found)

    async def adetect(self, text: str) -> list[PIIEntity]:
        """Async :meth:`detect` (only the optional LLM pass does I/O)."""
        found = self._regex(text)
        if self.llm is not None:
            found.extend(await self.llm.adetect(text))
        return _resolve_overlaps(found)


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep the first-listed (highest priority) entity wherever spans overlap."""
    kept: list[PIIEntity] = []
    for e in entities:
        if not any(e.start < k.end and e.end > k.start for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class PIIVault:
    """Reversible placeholder mapping shared across redaction calls.

    The same value always gets the same placeholder, and numbering continues
    across calls, so a prompt and its completion never collide.
    """

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counters: Counter[str] = Counter()

    def placeholder(self, entity_type: str, value: str) -> str:
        key = (entity_type, value)
        if key not in self._by_value:
            self._counters[entity_type] += 1
            ph = f"[{entity_type.upper()}_{self._counters[entity_type]}]"
            self._by_value[key] = ph
            self.mapping[ph] = value
        return self._by_value[key]


class RedactionResult(BaseModel):
    """Result of :func:`redact_pii`."""

    text: str
    entities: list[PIIEntity] = Field(default_factory=list)
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="placeholder -> original value (placeholder style only).",
    )


def replacement_for(
    entity: PIIEntity,
    style: RedactionStyle,
    vault: PIIVault | None = None,
    *,
    mask_char: str = "*",
    hash_salt: str = "",
) -> str:
    """The text that replaces *entity* under *style*."""
    if style == "mask":
        return mask_char * len(entity.value)
    if style == "hash":
        digest = hashlib.sha256((hash_salt + entity.value).encode()).hexdigest()[:12]
        return f"[{entity.type.upper()}:{digest}]"
    return (vault or PIIVault()).placeholder(entity.type, entity.value)


def apply_redactions(
    text: str,
    entities: Iterable[PIIEntity],
    style: RedactionStyle = "placeholder",
    vault: PIIVault | None = None,
    *,
    mask_char: str = "*",
    hash_salt: str = "",
) -> str:
    """Replace each (non-overlapping) entity span in *text*."""
    vault = vault if vault is not None else PIIVault()
    ordered = sorted(entities, key=lambda e: e.start)
    # Assign placeholders in reading order, then splice right-to-left.
    reps = [
        replacement_for(e, style, vault, mask_char=mask_char, hash_salt=hash_salt) for e in ordered
    ]
    for e, rep in reversed(list(zip(ordered, reps, strict=True))):
        text = text[: e.start] + rep + text[e.end :]
    return text


def redact_pii(
    text: str,
    *,
    style: RedactionStyle | None = None,
    detector: PIIDetector | None = None,
    vault: PIIVault | None = None,
) -> RedactionResult:
    """Detect and redact PII in *text*.

    *style* defaults to ``settings.GUARDRAILS_PII_REDACTION``.  Pass a shared
    *vault* to keep placeholder numbering consistent across calls.
    """
    style = style or default_redaction_style()
    detector = detector or PIIDetector()
    vault = vault if vault is not None else PIIVault()
    entities = detector.detect(text)
    before = set(vault.mapping)
    redacted = apply_redactions(text, entities, style, vault)
    mapping = {k: v for k, v in vault.mapping.items() if k not in before}
    return RedactionResult(text=redacted, entities=entities, mapping=mapping)


def rehydrate(text: str, mapping: Mapping[str, str]) -> str:
    """Restore placeholders produced by placeholder-style redaction."""
    if not mapping:
        return text
    rx = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return rx.sub(lambda m: mapping[m.group()], text)


def default_redaction_style() -> RedactionStyle:
    """``settings.GUARDRAILS_PII_REDACTION`` (``"placeholder"`` if unset)."""
    try:
        from aiforge.infra.config import get_settings

        return get_settings().GUARDRAILS_PII_REDACTION
    except Exception:  # noqa: BLE001 — settings errors must not break redaction
        return "placeholder"


# ---------------------------------------------------------------------------
# LLM-assisted detection (names, addresses)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a PII extraction engine. Find personal data that pattern matching "
    "cannot: people's names and postal/street addresses. Copy each value exactly "
    "as it appears in the text. Do not report organisations, products or public "
    "figures mentioned only by role. Return an empty list when there is none."
)
_LLM_USER = (
    "Entity types to report: {types}.\n"
    'Respond ONLY with JSON: {{"entities": [{{"type": "<type>", "text": "<exact text>"}}]}}\n\n'
    "Text:\n{text}"
)


def _llm_schema(types: list[str]) -> dict[str, Any]:
    return {
        "title": "pii_entities",
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": types},
                        "text": {"type": "string"},
                    },
                    "required": ["type", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }


class LLMPIIDetector:
    """Finds names and addresses with one LLM call using native structured output.

    Args:
        provider: An :class:`~aiforge.core.provider.LLMProvider`.  Defaults to
                  the provider configured in settings (resolved lazily).
        types:    Entity types to ask for.
        severity: Severity attached to matches.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        types: Iterable[str] = ("person_name", "address"),
        severity: Severity = "medium",
    ) -> None:
        self._provider = provider
        self.types = list(types)
        self.severity: Severity = severity

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            from aiforge.infra.config import get_settings
            from aiforge.providers import get_provider

            self._provider = get_provider(get_settings())
        return self._provider

    def _request(self, text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages = [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": _LLM_USER.format(types=", ".join(self.types), text=text)},
        ]
        kwargs: dict[str, Any] = {"temperature": 0.0}
        if getattr(self.provider, "supports_structured_output", False) is True:
            kwargs["response_format"] = _llm_schema(self.types)
        return messages, kwargs

    def _record(self, result: Any) -> None:
        from aiforge.infra.cost_tracker import CostTracker

        CostTracker.get_session().record(
            module="guardrails",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=result.cost,
        )

    def _parse(self, text: str, result: Any) -> list[PIIEntity]:
        data = result.parsed
        if data is None:
            from aiforge.modules.moderate.module import _extract_json

            try:
                data = json.loads(_extract_json(result.text))
            except (json.JSONDecodeError, TypeError):
                return []
        items = data.get("entities", []) if isinstance(data, dict) else []
        out: list[PIIEntity] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            etype, value = str(item.get("type", "")), str(item.get("text", "")).strip()
            if etype not in self.types or not value:
                continue
            for m in re.finditer(re.escape(value), text):
                out.append(
                    PIIEntity(
                        type=etype,
                        category="pii",
                        start=m.start(),
                        end=m.end(),
                        value=value,
                        severity=self.severity,
                        source="llm",
                    )
                )
        return out

    def detect(self, text: str) -> list[PIIEntity]:
        messages, kwargs = self._request(text)
        result = self.provider.complete(messages, **kwargs)
        self._record(result)
        return self._parse(text, result)

    async def adetect(self, text: str) -> list[PIIEntity]:
        messages, kwargs = self._request(text)
        result = await self.provider.complete_async(messages, **kwargs)
        self._record(result)
        return self._parse(text, result)
