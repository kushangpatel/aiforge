"""Pydantic schemas for the DocQA (RAG) module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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


class Citation(BaseModel):
    """Where a retrieved chunk came from, for citing it in an answer."""

    source: str
    """File path or URL the chunk was loaded from."""
    page: int | str | None = None
    """Page number (PDFs: 0-based int as stored by the loader; custom loaders may use labels)."""
    heading_path: str | None = None
    """Markdown heading path, e.g. ``"Install > From source"`` (markdown chunking)."""
    chunk_index: int | None = None
    """Index of the chunk within its ingest batch."""
    score: float | None = None
    """Retrieval score in ``[0, 1]`` (meaning depends on the retrieval mode)."""
    rerank_score: float | None = None
    """Reranker relevance score, when reranking is enabled."""

    def label(self) -> str:
        """Human-readable label, e.g. ``"guide.pdf, page 3, Setup > Install"``."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.heading_path:
            parts.append(self.heading_path)
        return ", ".join(parts)


class QAResponse(ModuleResponse):
    """Response from DocQA.ask() / aask()."""

    text: str = ""
    sources: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    """One entry per distinct retrieved chunk, with page / heading metadata."""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    iterations: int = 0
    trace: list[dict[str, Any]] | None = None

    def to_text(self) -> str:
        return self.text
