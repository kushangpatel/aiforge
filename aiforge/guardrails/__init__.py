"""
Guardrails — composable input / output checks for any module or callable.

See ``docs/guardrails.md``.  Quick start::

    from aiforge.guardrails import Guard, PIIGuard, PromptInjectionGuard

    guard = Guard(input=[PIIGuard(), PromptInjectionGuard()], output=[PIIGuard()],
                  on_fail="redact")
    safe_summarize = guard.wrap(Summarizer())
"""

from aiforge.core.types import GuardrailViolation
from aiforge.guardrails.checks import (
    BannedTermsGuard,
    Check,
    GuardContext,
    MaxLengthGuard,
    OnFail,
    PIIGuard,
    PromptInjectionGuard,
    RegexGuard,
    SchemaGuard,
    SystemPromptLeakGuard,
    Violation,
)
from aiforge.guardrails.guard import Guard, GuardedCallable, GuardResult
from aiforge.guardrails.pii import (
    PII_GROUPS,
    PII_TYPES,
    LLMPIIDetector,
    PIIDetector,
    PIIEntity,
    PIIPattern,
    PIIVault,
    RedactionResult,
    RedactionStyle,
    iban_valid,
    luhn_valid,
    redact_pii,
    rehydrate,
)

__all__ = [
    "PII_GROUPS",
    "PII_TYPES",
    "BannedTermsGuard",
    "Check",
    "Guard",
    "GuardContext",
    "GuardResult",
    "GuardedCallable",
    "GuardrailViolation",
    "LLMPIIDetector",
    "MaxLengthGuard",
    "OnFail",
    "PIIDetector",
    "PIIEntity",
    "PIIGuard",
    "PIIPattern",
    "PIIVault",
    "PromptInjectionGuard",
    "RedactionResult",
    "RedactionStyle",
    "RegexGuard",
    "SchemaGuard",
    "SystemPromptLeakGuard",
    "Violation",
    "iban_valid",
    "luhn_valid",
    "redact_pii",
    "rehydrate",
]
