"""
Language-aware text splitting.

Wraps LangChain text splitters with sensible defaults per language.  Three
chunking strategies are available:

``"recursive"`` (default)
    Character-sized chunks.  Code files use language-specific separators;
    prose splits on paragraphs, lines, then sentences.

``"markdown"``
    Markdown files (``.md``, ``.markdown``) are first split on headings, so a
    chunk never straddles two sections, and each chunk's heading path
    (e.g. ``"Install > From source"``) is stored in
    ``metadata["heading_path"]``.  Oversized sections are then split
    recursively.  Other files use the recursive strategy.

``"token"``
    Like ``"recursive"`` but chunk size and overlap are measured in tiktoken
    tokens (``cl100k_base`` by default), which keeps chunks within an
    embedding model's input limit regardless of language.

When only ``chunk_size`` is overridden, the overlap defaults to the smaller of
the strategy's default overlap and a quarter of the chunk size, so small
chunk sizes stay valid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ChunkingStrategy = Literal["recursive", "markdown", "token"]
CHUNKING_STRATEGIES: tuple[str, ...] = ("recursive", "markdown", "token")

# Extension → LangChain Language enum or None (prose)
_EXT_TO_LANGUAGE: dict[str, Language | None] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.JS,  # LangChain uses JS splitter for TS
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
    ".rst": None,  # prose splitter
    ".txt": None,
    ".html": Language.HTML,
    ".htm": Language.HTML,
}

_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
_PROSE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]
_HEADER_KEYS = [key for _, key in _HEADERS]

# Defaults (characters)
_CODE_CHUNK_SIZE = 1_000
_CODE_CHUNK_OVERLAP = 200
_PROSE_CHUNK_SIZE = 1_200
_PROSE_CHUNK_OVERLAP = 200

# Defaults (tokens) for the "token" strategy
_TOKEN_CHUNK_SIZE = 400
_TOKEN_CHUNK_OVERLAP = 60
_DEFAULT_ENCODING = "cl100k_base"


def validate_chunking(strategy: str) -> ChunkingStrategy:
    """Return *strategy* normalised, or raise :class:`ValueError` if unknown."""
    norm = (strategy or "").strip().lower()
    if norm not in CHUNKING_STRATEGIES:
        raise ValueError(f"chunking must be one of {list(CHUNKING_STRATEGIES)}, got {strategy!r}")
    return norm  # type: ignore[return-value]


def _normalise_ext(file_type: str) -> str:
    return f".{file_type}" if file_type and not file_type.startswith(".") else file_type


class TextSplitter:
    """Split :class:`~langchain_core.documents.Document` objects into chunks.

    Automatically selects the best splitter based on file extension metadata.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        strategy: str = "recursive",
        encoding_name: str = _DEFAULT_ENCODING,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """
        Args:
            chunk_size:    Override default chunk size for all content types
                           (tokens for the ``"token"`` strategy, else characters).
            chunk_overlap: Override default overlap for all content types.
            strategy:      ``"recursive"`` (default), ``"markdown"`` or ``"token"``.
            encoding_name: tiktoken encoding used by the ``"token"`` strategy.
                           tiktoken downloads each encoding once and caches it
                           (set ``TIKTOKEN_CACHE_DIR`` for offline machines).
            token_counter: Custom ``text -> token count`` function for the
                           ``"token"`` strategy (e.g. a Hugging Face tokenizer);
                           overrides *encoding_name*.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap is not None and chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        self._chunk_size_override = chunk_size
        self._chunk_overlap_override = chunk_overlap
        self.strategy: ChunkingStrategy = validate_chunking(strategy)
        self._encoding_name = encoding_name
        self._token_counter = token_counter

    def split(self, documents: list[Document]) -> list[Document]:
        """Split *documents* into appropriately-sized chunks.

        Returns a new list of :class:`Document` objects.  Source metadata
        (``source``, ``file_type``, ``page``, …) is propagated to every chunk;
        the markdown strategy adds ``heading_path``.
        """
        output: list[Document] = []
        for doc in documents:
            ext = _normalise_ext(doc.metadata.get("file_type", ""))
            if self.strategy == "markdown" and ext.lower() in _MARKDOWN_EXTENSIONS:
                output.extend(self._split_markdown(doc, ext))
                continue
            splitter = self._get_splitter(ext)
            output.extend(splitter.split_documents([doc]))
        return output

    def split_text(self, text: str, ext: str = "") -> list[str]:
        """Split a raw *text* string (no Document metadata required)."""
        ext = _normalise_ext(ext)
        if self.strategy == "markdown" and ext.lower() in _MARKDOWN_EXTENSIONS:
            doc = Document(page_content=text, metadata={})
            return [c.page_content for c in self._split_markdown(doc, ext)]
        splitter = self._get_splitter(ext)
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _token_length_fn(self) -> Callable[[str], int]:
        if self._token_counter is not None:
            return self._token_counter
        import tiktoken

        encoding = tiktoken.get_encoding(self._encoding_name)

        def _count(text: str) -> int:
            return len(encoding.encode(text, disallowed_special=()))

        self._token_counter = _count
        return _count

    def _sizes(self, default_size: int, default_overlap: int) -> tuple[int, int]:
        size = self._chunk_size_override or default_size
        if self._chunk_overlap_override is not None:
            overlap = self._chunk_overlap_override
        else:
            overlap = min(default_overlap, size // 4)
        return size, min(overlap, max(0, size - 1))

    def _split_markdown(self, doc: Document, ext: str) -> list[Document]:
        """Split on headings, then size-split each section; record the heading path."""
        header_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
        sections = header_splitter.split_text(doc.page_content)
        sizer = self._get_splitter(ext)
        out: list[Document] = []
        for section in sections:
            headings = [section.metadata[k] for k in _HEADER_KEYS if section.metadata.get(k)]
            meta = dict(doc.metadata)
            if headings:
                meta["heading_path"] = " > ".join(headings)
            for piece in sizer.split_text(section.page_content):
                out.append(Document(page_content=piece, metadata=dict(meta)))
        return out

    def _get_splitter(self, ext: str) -> RecursiveCharacterTextSplitter:
        lang = _EXT_TO_LANGUAGE.get(ext.lower())
        separators = (
            RecursiveCharacterTextSplitter.get_separators_for_language(lang)
            if lang is not None
            else _PROSE_SEPARATORS
        )

        if self.strategy == "token":
            size, overlap = self._sizes(_TOKEN_CHUNK_SIZE, _TOKEN_CHUNK_OVERLAP)
            return RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=separators,
                is_separator_regex=lang is not None,
                length_function=self._token_length_fn(),
            )

        if lang is not None:
            # Language-aware code splitter
            size, overlap = self._sizes(_CODE_CHUNK_SIZE, _CODE_CHUNK_OVERLAP)
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=size,
                chunk_overlap=overlap,
            )
        # General prose splitter
        size, overlap = self._sizes(_PROSE_CHUNK_SIZE, _PROSE_CHUNK_OVERLAP)
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=separators,
        )
