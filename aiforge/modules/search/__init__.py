"""SemanticSearch module — embed, index, and semantically query text."""

from aiforge.modules.search.module import SemanticSearch
from aiforge.modules.search.schemas import (
    IndexResponse,
    SearchDocument,
    SearchHit,
    SearchResponse,
)

__all__ = [
    "SemanticSearch",
    "SearchDocument",
    "SearchHit",
    "SearchResponse",
    "IndexResponse",
]
