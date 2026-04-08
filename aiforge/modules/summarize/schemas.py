"""Pydantic schemas for the Summarizer module."""

from __future__ import annotations

from typing import Literal

from aiforge.core.types import ModuleResponse

SummarizeStyle = Literal["concise", "detailed", "bullet_points", "executive"]


class SummarizeResponse(ModuleResponse):
    """Response from Summarizer.summarize()."""

    summary: str = ""
    """The generated summary text."""

    strategy_used: Literal["stuff", "map_reduce"] = "stuff"
    """Whether the text fit in one call ('stuff') or required chunking ('map_reduce')."""

    chunks_processed: int = 1
    """Number of chunks — 1 for stuff, N for map_reduce."""

    def to_text(self) -> str:
        """Pipeline protocol — return the summary for downstream modules."""
        return self.summary
