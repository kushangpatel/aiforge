"""
Language-aware text splitting.

Wraps LangChain text splitters with sensible defaults per language:
- Code files use RecursiveCharacterTextSplitter with language-specific separators
- Prose / markup uses RecursiveCharacterTextSplitter with sentence-aware defaults
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Extension → LangChain Language enum or None (prose)
_EXT_TO_LANGUAGE: dict[str, Language | None] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.JS,   # LangChain uses JS splitter for TS
    ".tsx": Language.JS,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
    ".rst": None,       # prose splitter
    ".txt": None,
    ".html": Language.HTML,
    ".htm": Language.HTML,
}

# Defaults
_CODE_CHUNK_SIZE = 1_000
_CODE_CHUNK_OVERLAP = 200
_PROSE_CHUNK_SIZE = 1_200
_PROSE_CHUNK_OVERLAP = 200


class TextSplitter:
    """Split :class:`~langchain_core.documents.Document` objects into chunks.

    Automatically selects the best splitter based on file extension metadata.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Args:
            chunk_size:    Override default chunk size for all content types.
            chunk_overlap: Override default overlap for all content types.
        """
        self._chunk_size_override = chunk_size
        self._chunk_overlap_override = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        """Split *documents* into appropriately-sized chunks.

        Returns a new list of :class:`Document` objects.  Source metadata
        (``source``, ``file_type``, ``page``, …) is propagated to every chunk.
        """
        output: list[Document] = []
        for doc in documents:
            file_type = doc.metadata.get("file_type", "")
            ext = f".{file_type}" if file_type and not file_type.startswith(".") else file_type
            splitter = self._get_splitter(ext)
            chunks = splitter.split_documents([doc])
            output.extend(chunks)
        return output

    def split_text(self, text: str, ext: str = "") -> list[str]:
        """Split a raw *text* string (no Document metadata required)."""
        splitter = self._get_splitter(ext)
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_splitter(self, ext: str) -> RecursiveCharacterTextSplitter:
        lang = _EXT_TO_LANGUAGE.get(ext.lower())

        if lang is not None:
            # Language-aware code splitter
            chunk_size = self._chunk_size_override or _CODE_CHUNK_SIZE
            chunk_overlap = self._chunk_overlap_override or _CODE_CHUNK_OVERLAP
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # General prose splitter
            chunk_size = self._chunk_size_override or _PROSE_CHUNK_SIZE
            chunk_overlap = self._chunk_overlap_override or _PROSE_CHUNK_OVERLAP
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )
