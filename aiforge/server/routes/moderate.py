"""Moderator endpoint (/moderate)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter(prefix="/moderate", tags=["moderate"])

_VALID_CHECKS = frozenset({"pii", "toxicity", "prompt_injection"})
_VALID_PII_TYPES = frozenset({"email", "phone", "ssn", "credit_card"})


class _FlagOut(BaseModel):
    type: str
    subtype: str | None = None
    span: str | None = None
    confidence: float = 1.0


class ModerateRequest(BaseModel):
    text: str
    checks: list[str] = ["pii", "toxicity", "prompt_injection"]
    pii_types: list[str] = ["email", "phone", "ssn", "credit_card"]
    action: str = "check"  # "check" | "redact"

    @field_validator("checks")
    @classmethod
    def _validate_checks(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("checks must be non-empty")
        invalid = set(v) - _VALID_CHECKS
        if invalid:
            raise ValueError(f"Unknown check types: {sorted(invalid)}")
        return v

    @field_validator("pii_types")
    @classmethod
    def _validate_pii_types(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("pii_types must be non-empty")
        invalid = set(v) - _VALID_PII_TYPES
        if invalid:
            raise ValueError(f"Unknown PII types: {sorted(invalid)}")
        return v


class ModerateResponse(BaseModel):
    flagged: bool = False
    flags: list[_FlagOut] = []
    redacted_text: str | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@router.post("", response_model=ModerateResponse)
async def run_moderate(body: ModerateRequest) -> Any:
    """Check text for PII, toxicity, and prompt injection."""
    if body.action not in {"check", "redact"}:
        raise HTTPException(status_code=422, detail="action must be 'check' or 'redact'")

    from aiforge.modules.moderate.module import Moderator

    m = Moderator(checks=body.checks, pii_types=body.pii_types)

    if body.action == "redact":
        result = await m.aredact(body.text)
        return ModerateResponse(
            flagged=bool(result.redactions),
            flags=[_FlagOut(**f.model_dump()) for f in result.redactions],
            redacted_text=result.text,
            cost_usd=result.cost,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )

    result = await m.acheck(body.text)
    return ModerateResponse(
        flagged=result.flagged,
        flags=[_FlagOut(**f.model_dump()) for f in result.flags],
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
