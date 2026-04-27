"""SemanticSearch module — embed, index, and semantically query text."""

from aiforge.modules.search.module import SemanticSearch
from aiforge.modules.search.schemas import (
    IndexRequest,
    IndexResponse,
    SearchDocument,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "SemanticSearch",
    "IndexRequest",
    "IndexResponse",
    "SearchDocument",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]
