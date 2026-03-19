"""Pydantic schemas for the Generator module."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from aiforge.core.types import ModuleResponse


@dataclass
class GuardrailResult:
    """Result returned by a custom guardrail function."""

    passed: bool
    message: str = ""


class GenerateResponse(ModuleResponse):
    """Response from Generator.generate() / .run()."""

    text: str = ""
    """The generated content."""

    guardrail_flags: list[str] = Field(default_factory=list)
    """Names of guardrails that were still violated after all retry attempts."""

    retry_count: int = 0
    """Number of retry attempts due to guardrail violations (0 = first try passed)."""

    def to_text(self) -> str:
        """Pipeline protocol — return the generated text for downstream modules."""
        return self.text
