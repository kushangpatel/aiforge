"""Generator endpoint (/generate)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    template: str | None = None
    prompt: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    guardrails: list[str] = Field(default_factory=list)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    competitors: list[str] = Field(default_factory=list)


class GenerateRouteResponse(BaseModel):
    text: str = ""
    guardrail_flags: list[str] = Field(default_factory=list)
    retry_count: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@router.post("", response_model=GenerateRouteResponse)
async def run_generate(body: GenerateRequest) -> Any:
    """Generate content from a template or freeform prompt."""
    from aiforge.modules.generate.module import Generator

    try:
        g = Generator(
            template=body.template,
            guardrails=body.guardrails,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            competitors=body.competitors,
        )
        result = await g.agenerate(prompt=body.prompt, **body.variables)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return GenerateRouteResponse(
        text=result.text,
        guardrail_flags=result.guardrail_flags,
        retry_count=result.retry_count,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )
