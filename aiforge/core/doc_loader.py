"""
Document loading: files, directories, and git repositories.

Supported formats
-----------------
=========================  ==========================================
Category                   Extensions
=========================  ==========================================
Code / text                .py .js .ts .go .java .rs .c .cpp .rb
                           .md .txt .rst .sh
Data                       .json .yaml .yml .csv .toml
PDF                        .pdf  (requires: pip install pypdf)
Word                       .docx (requires: pip install python-docx)
=========================  ==========================================

Unsupported file types are skipped with a warning; inspect
``IngestStats.skipped_files`` to see which ones were missed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → (loader_class, extra_kwargs)
# ---------------------------------------------------------------------------

# Extensions handled as plain text (no special loader needed)
_TEXT_EXTENSIONS = frozenset(
    {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".go", ".java", ".rs", ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php", ".swift", ".kt", ".scala",
        ".md", ".txt", ".rst", ".sh", ".bash", ".zsh",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".csv", ".xml", ".html", ".htm",
        ".sql", ".dockerfile",
    }
)

_IGNORE_EXTENSIONS = frozenset(
    {
        ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
        ".zip", ".tar", ".gz", ".bz2", ".xz",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
        ".mp3", ".mp4", ".avi", ".mov",
        ".whl", ".egg-info",
        ".lock",  # package lock files — usually large & uninteresting
    }
)

_IGNORE_DIR_NAMES = frozenset(
    {
        ".git", ".svn", ".hg",
        "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
        "node_modules", ".next", ".nuxt",
        "venv", ".venv", "env", ".env",
        "dist", "build", ".build",
        "aiforge_data",
    }
)


@dataclass
class IngestStats:
    """Statistics returned after ingesting a source."""

    files_processed: int = 0
    chunks_indexed: int = 0
    skipped_files: list[str] = field(default_factory=list)
    source: str = ""
    languages: dict[str, int] = field(default_factory=dict)
    collection: str = ""


class DocLoader:
    """Load documents from files, directories, and git repos."""

    def __init__(
        self,
        file_extensions: list[str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """
        Args:
            file_extensions: Whitelist of extensions to include (with dot).
                             ``None`` → use all supported extensions.
            encoding:        Text encoding for plain-text files.
        """
        self._allowed_extensions: frozenset[str] | None = (
            frozenset(e.lower() for e in file_extensions)
            if file_extensions is not None
            else None
        )
        self._encoding = encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str | list[str]) -> tuple[list[Document], IngestStats]:
        """Load documents from *source*.

        Args:
            source: A file path, directory path, git URL
                    (``https://...git``), or a list of file paths.

        Returns:
            Tuple of ``(documents, stats)``.
        """
        stats = IngestStats(source=str(source))

        if isinstance(source, list):
            docs = self._load_file_list(source, stats)
        elif isinstance(source, str) and source.endswith(".git"):
            docs = self._load_git_repo(source, stats)
        else:
            path = Path(source)
            if path.is_dir():
                docs = self._load_directory(path, stats)
            elif path.is_file():
                docs = self._load_single_file(path, stats)
            else:
                raise FileNotFoundError(f"Source not found: {source!r}")

        return docs, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_directory(self, directory: Path, stats: IngestStats) -> list[Document]:
        docs: list[Document] = []
        for file_path in self._iterate_files(directory):
            file_docs = self._load_single_file(file_path, stats)
            docs.extend(file_docs)
        return docs

    def _load_file_list(
        self, paths: list[str], stats: IngestStats
    ) -> list[Document]:
        docs: list[Document] = []
        for p in paths:
            file_docs = self._load_single_file(Path(p), stats)
            docs.extend(file_docs)
        return docs

    def _load_single_file(self, path: Path, stats: IngestStats) -> list[Document]:
        ext = path.suffix.lower()

        if ext in _IGNORE_EXTENSIONS:
            return []

        if self._allowed_extensions is not None and ext not in self._allowed_extensions:
            stats.skipped_files.append(str(path))
            return []

        # Route to the appropriate loader
        loader_fn = self._get_loader_fn(ext)
        if loader_fn is None:
            stats.skipped_files.append(str(path))
            return []

        try:
            docs = loader_fn(path)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to load %s: %s", path, exc)
            stats.skipped_files.append(str(path))
            return []

        stats.files_processed += 1
        lang = ext.lstrip(".")
        stats.languages[lang] = stats.languages.get(lang, 0) + 1
        return docs

    def _iterate_files(self, directory: Path):
        """Yield all non-ignored files under *directory* recursively."""
        for item in directory.rglob("*"):
            # Skip ignored directories anywhere in the path
            if any(part in _IGNORE_DIR_NAMES for part in item.parts):
                continue
            if item.is_file():
                yield item

    def _get_loader_fn(self, ext: str):
        """Return a callable ``(path) -> list[Document]`` for *ext*, or None."""
        if ext in _TEXT_EXTENSIONS:
            return self._load_text
        if ext == ".pdf":
            return self._load_pdf
        if ext == ".docx":
            return self._load_docx
        return None

    def _load_text(self, path: Path) -> list[Document]:
        text = path.read_text(encoding=self._encoding, errors="replace")
        return [
            Document(
                page_content=text,
                metadata={"source": str(path), "file_type": path.suffix.lstrip(".")},
            )
        ]

    def _load_pdf(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            try:
                from langchain.document_loaders import PyPDFLoader  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "PyPDF not installed. Run: pip install \"aiforge[docs]\" to add PDF support."
                ) from None
        loader = PyPDFLoader(str(path))
        return loader.load()

    def _load_docx(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            raise ImportError(
                "python-docx not installed. Run: pip install \"aiforge[docs]\" for DOCX support."
            ) from None
        loader = Docx2txtLoader(str(path))
        return loader.load()

    def _load_git_repo(self, git_url: str, stats: IngestStats) -> list[Document]:
        """Clone a git repo to a temp dir and load it as a directory."""
        import subprocess
        import tempfile

        # Validate URL scheme to prevent SSRF / local path exploitation (M10)
        allowed_schemes = ("https://", "http://", "git://", "ssh://", "git@")
        if not any(git_url.startswith(scheme) for scheme in allowed_schemes):
            raise ValueError(
                f"Git URL {git_url!r} must start with one of: {allowed_schemes}. "
                "Local paths ending with .git are not supported for security reasons."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(  # noqa: S603
                ["git", "clone", "--depth", "1", git_url, tmpdir],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"git clone failed for {git_url!r}:\n{result.stderr}"
                )
            docs = self._load_directory(Path(tmpdir), stats)

        # Rewrite metadata source paths so they reference the git URL rather
        # than the deleted temp directory. This prevents FileNotFoundError if
        # callers try to re-read a path from metadata after the clone is gone.
        for doc in docs:
            original = doc.metadata.get("source", "")
            rel = original.replace(tmpdir, "").lstrip("/\\")
            doc.metadata["source"] = f"{git_url}/{rel}" if rel else git_url
            doc.metadata["git_url"] = git_url

        return docs
