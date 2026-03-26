"""Pydantic schemas for the Vision module."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from aiforge.core.types import ModuleResponse

VisionTask = Literal["describe", "qa", "extract", "classify"]


class VisionResponse(ModuleResponse):
    """Response from Vision.analyze() / .run()."""

    text: str = ""
    """The generated description, answer, or extracted content."""

    task: VisionTask = "describe"
    """The task that was performed."""

    image_source: str = ""
    """Path, URL, or identifier of the analyzed image."""

    def to_text(self) -> str:
        """Pipeline protocol — return the analysis text for downstream modules."""
        return self.text
