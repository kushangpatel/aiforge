"""Pydantic schemas for the Moderator module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aiforge.core.types import ModuleResponse


class ModerationFlag(BaseModel):
    """A single content-safety or PII flag."""

    type: str = Field(
        description="Top-level check type: 'pii' | 'toxicity' | 'prompt_injection'",
    )
    subtype: str | None = Field(
        default=None,
        description="For PII: 'email' | 'phone' | 'ssn' | 'credit_card' | any guardrails PII type",
    )
    span: str | None = Field(
        default=None,
        description="The exact matched text span (None for LLM-based checks)",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence 0.0–1.0 (always 1.0 for regex matches)",
    )


class ModerateResponse(ModuleResponse):
    """Response from Moderator.check()."""

    flagged: bool = False
    flags: list[ModerationFlag] = Field(default_factory=list)
    input_text: str = Field(
        default="",
        description="Original input text — passed through by to_text() when clean.",
    )

    @property
    def label(self) -> str:
        """Discrete label for pipeline Router branching: ``'flagged'`` or ``'clean'``."""
        return "flagged" if self.flagged else "clean"

    def to_text(self) -> str:
        """Pipeline chaining: pass original text through when clean, flag summary when flagged."""
        if not self.flagged:
            return self.input_text or "clean"
        parts = []
        for f in self.flags:
            lbl = f"{f.type}:{f.subtype}" if f.subtype else f.type
            parts.append(f"[{lbl}] {f.span or ''}")
        return "; ".join(parts)


class RedactResponse(ModuleResponse):
    """Response from Moderator.redact()."""

    text: str = ""
    redactions: list[ModerationFlag] = Field(default_factory=list)

    def to_text(self) -> str:
        return self.text
