"""Pydantic schemas for the DocQA (RAG) module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aiforge.core.types import ModuleResponse


class IngestResponse(ModuleResponse):
    """Response from DocQA.ingest() / aingest()."""

    chunks_indexed: int = 0
    files_processed: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    skipped_files: list[str] = Field(default_factory=list)
    collection: str = ""

    def to_text(self) -> str:
        return (
            f"Ingested {self.files_processed} files into '{self.collection}'"
            f" ({self.chunks_indexed} chunks)"
        )


class QAResponse(ModuleResponse):
    """Response from DocQA.ask() / aask()."""

    text: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    iterations: int = 0
    trace: list[dict[str, Any]] | None = None

    def to_text(self) -> str:
        return self.text
