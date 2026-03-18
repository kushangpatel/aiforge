"""Classifier endpoint (/classify)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator, model_validator

router = APIRouter(prefix="/classify", tags=["classify"])

_LABELS_MAX = 100


class _ExampleIn(BaseModel):
    text: str
    label: str


class ClassifyRequest(BaseModel):
    text: str
    labels: list[str]
    examples: list[_ExampleIn] = []
    multi_label: bool = False

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("labels must be non-empty")
        if len(v) > _LABELS_MAX:
            raise ValueError(f"labels must have at most {_LABELS_MAX} entries")
        return v

    @model_validator(mode="after")
    def _validate_example_labels(self) -> "ClassifyRequest":
        """Ensure every example label exists in the provided taxonomy (I5 / I6)."""
        if self.labels and self.examples:
            label_set = set(self.labels)
            bad = [e.label for e in self.examples if e.label not in label_set]
            if bad:
                raise ValueError(
                    f"Example labels not in taxonomy: {sorted(set(bad))}"
                )
        return self


class _LabelScoreOut(BaseModel):
    label: str
    confidence: float = 1.0


class ClassifyResponse(BaseModel):
    label: str | None = None
    scores: list[_LabelScoreOut] = []
    rationale: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@router.post("", response_model=ClassifyResponse)
async def run_classify(body: ClassifyRequest) -> Any:
    """Classify text into one or more of the provided labels using zero/few-shot prompting."""
    from aiforge.modules.classify.module import Classifier
    from aiforge.modules.classify.schemas import ClassifyExample

    examples = [ClassifyExample(text=e.text, label=e.label) for e in body.examples]
    try:
        c = Classifier(
            labels=body.labels,
            examples=examples or None,
            multi_label=body.multi_label,
        )
        result = await c.aclassify(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return ClassifyResponse(
        label=result.label,
        scores=[_LabelScoreOut(**s.model_dump()) for s in result.scores],
        rationale=result.rationale,
        cost_usd=result.cost,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )

