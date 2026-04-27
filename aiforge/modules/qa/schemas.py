"""Pydantic schemas for the DocQA (RAG) module."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aiforge.core.types import ModuleResponse


class IngestResponse(ModuleResponse):
    """Response from DocQA.ingest() / aingest() / ingest_incremental()."""

    chunks_indexed: int = 0
    files_processed: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    skipped_files: list[str] = Field(default_factory=list)
    collection: str = ""

    # Incremental ingestion stats (populated by ingest_incremental only)
    files_added: int = 0
    """New files that didn't exist in the index before."""
    files_updated: int = 0
    """Files whose mtime changed since last ingest (stale chunks purged + reindexed)."""
    files_unchanged: int = 0
    """Files whose mtime matched the index — skipped entirely."""
    files_deleted: int = 0
    """Files present in the index but no longer on disk (chunks purged)."""
    chunks_deleted: int = 0
    """Total stale/orphan chunks removed from the index."""

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
