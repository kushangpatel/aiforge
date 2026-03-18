"""Pydantic schemas for the Classifier module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aiforge.core.types import ModuleResponse


class ClassifyExample(BaseModel):
    """A single few-shot example for the Classifier."""

    text: str = Field(description="Example input text")
    label: str = Field(description="Correct label for this text")


class LabelScore(BaseModel):
    """A label with its predicted confidence score."""

    label: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClassifyResponse(ModuleResponse):
    """Response from Classifier.classify()."""

    label: str | None = None
    """Top predicted label, or ``None`` if the LLM could not determine one."""

    scores: list[LabelScore] = Field(default_factory=list)
    """All returned labels with confidence scores, sorted descending."""

    rationale: str = ""
    """Brief explanation provided by the LLM."""

    def to_text(self) -> str:
        """Return the top label for pipeline chaining (``'unknown'`` if unset)."""
        return self.label or "unknown"
