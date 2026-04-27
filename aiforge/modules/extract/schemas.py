"""Pydantic schemas for the Extractor module."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from aiforge.core.types import ModuleResponse


class ExtractExample(BaseModel):
    """A single few-shot example for guided extraction."""

    input: str = Field(description="Example source text", min_length=1)
    output: dict[str, Any] = Field(description="Expected extracted data for this text")


class ExtractResponse(ModuleResponse):
    """Response from Extractor.extract()."""

    data: dict[str, Any] = Field(default_factory=dict)
    """Validated extracted data (dict representation of the Pydantic model, or raw dict)."""

    raw_json: str = ""
    """Raw JSON string returned by the LLM before validation."""

    validation_passed: bool = False
    """Whether the LLM output passed schema validation."""

    retry_count: int = 0
    """Number of retry attempts before arriving at this result (0 = first attempt)."""

    def to_text(self) -> str:
        """Pipeline protocol — return extracted data as a JSON string."""
        return json.dumps(self.data) if self.data else "{}"
