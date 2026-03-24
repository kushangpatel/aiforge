"""Extractor endpoint (/extract)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/extract", tags=["extract"])


class _ExampleIn(BaseModel):
    input: str
    output: dict[str, Any]


class ExtractRequest(BaseModel):
    text: str
    output_schema: dict[str, Any] = Field(
        description="JSON Schema describing the structure to extract",
    )
    examples: list[_ExampleIn] = Field(
        default_factory=list,
        description="Optional few-shot examples to guide extraction",
    )


class ExtractResponse(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    raw_json: str = ""
    validation_passed: bool = False
    retry_count: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@router.post("", response_model=ExtractResponse)
async def run_extract(body: ExtractRequest) -> Any:
    """Extract structured data from text using the provided JSON schema."""
    from aiforge.modules.extract.module import Extractor
    from aiforge.modules.extract.schemas import ExtractExample

    examples = [ExtractExample(input=e.input, output=e.output) for e in body.examples]
    try:
        e = Extractor(output_schema=body.output_schema, examples=examples or None)
        result = await e.aextract(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from aiforge.core.types import RateLimitError

        if isinstance(exc, RateLimitError):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc))
    return ExtractResponse(
        data=result.data,
        raw_json=result.raw_json,
        validation_passed=result.validation_passed,
        retry_count=result.retry_count,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        latency_ms=result.latency_ms,
    )

