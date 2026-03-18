"""Pydantic schemas for the SemanticSearch module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from aiforge.core.types import ModuleResponse


class SearchDocument(BaseModel):
    """A document to be indexed in the vector store."""

    id: str = Field(description="Unique document identifier", min_length=1)
    text: str = Field(description="Document text content", min_length=1)
    metadata: dict = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A single ranked search result."""

    id: str
    """Document identifier."""

    text: str
    """Document text."""

    score: float = Field(ge=0.0, le=1.0)
    """Cosine similarity in [0, 1]. Derived as ``1.0 - cosine_distance``."""

    metadata: dict = Field(default_factory=dict)


class IndexResponse(ModuleResponse):
    """Response from SemanticSearch.index() / aindex()."""

    indexed_count: int = 0
    """Number of documents upserted in this call."""

    collection: str = ""
    """Name of the vector store collection."""

    total_documents: int = 0
    """Total documents now in the collection (including previous contents)."""

    def to_text(self) -> str:
        """Return a summary string for pipeline chaining."""
        return (
            f"Indexed {self.indexed_count} documents into '{self.collection}'"
            f" ({self.total_documents} total)"
        )


class SearchResponse(ModuleResponse):
    """Response from SemanticSearch.query() / aquery()."""

    hits: list[SearchHit] = Field(default_factory=list)
    """Ranked results, highest score first."""

    query: str = ""
    """The query string that produced these results."""

    total_searched: int = 0
    """Total documents in the searched collection."""

    def to_text(self) -> str:
        """Return the top hit's text for pipeline chaining, or empty string."""
        return self.hits[0].text if self.hits else ""
