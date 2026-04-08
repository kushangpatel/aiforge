"""Translator endpoint (/translate)."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/translate", tags=["translate"])


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "English"
    source_language: str = "auto"
    formality: Literal["formal", "informal", "neutral"] = "neutral"
    domain: str | None = None


class TranslateRouteResponse(BaseModel):
    text: str = ""
    source_language: str = ""
    target_language: str = ""
    formality: str = "neutral"
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@router.post("", response_model=TranslateRouteResponse)
async def run_translate(body: TranslateRequest) -> Any:
    """Translate text between languages."""
    from aiforge.modules.translate.module import Translator

    try:
        t = Translator(
            target_language=body.target_language,
            source_language=body.source_language,
            formality=body.formality,
            domain=body.domain,
        )
        result = await t.atranslate(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TranslateRouteResponse(
        text=result.text,
        source_language=result.source_language,
        target_language=result.target_language,
        formality=result.formality,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )
