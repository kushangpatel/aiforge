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
from fnmatch import fnmatchcase
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → (loader_class, extra_kwargs)
# ---------------------------------------------------------------------------

# Extensions handled as plain text (no special loader needed)
_TEXT_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".java",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".md",
        ".txt",
        ".rst",
        ".sh",
        ".bash",
        ".zsh",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".xml",
        ".html",
        ".htm",
        ".sql",
        ".dockerfile",
    }
)

_IGNORE_EXTENSIONS = frozenset(
    {
        ".pyc",
        ".pyo",
        ".class",
        ".o",
        ".so",
        ".dll",
        ".exe",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".ico",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".whl",
        ".egg-info",
        ".lock",  # package lock files — usually large & uninteresting
    }
)

_IGNORE_DIR_NAMES = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".next",
        ".nuxt",
        "venv",
        ".venv",
        "env",
        ".env",
        "dist",
        "build",
        ".build",
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
            frozenset(e.lower() for e in file_extensions) if file_extensions is not None else None
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
                # User explicitly chose this file — skip extension filter.
                docs = self._load_single_file(path, stats, apply_extension_filter=False)
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

    def _load_file_list(self, paths: list[str], stats: IngestStats) -> list[Document]:
        docs: list[Document] = []
        gitignore_cache: dict[Path, list[str]] = {}
        for p in paths:
            path = Path(p)
            if path.exists() and self._is_gitignored_from_ancestors(path, gitignore_cache):
                continue
            file_docs = self._load_single_file(path, stats)
            docs.extend(file_docs)
        return docs

    def _load_single_file(
        self,
        path: Path,
        stats: IngestStats,
        *,
        apply_extension_filter: bool = True,
    ) -> list[Document]:
        ext = path.suffix.lower()

        if ext in _IGNORE_EXTENSIONS:
            return []

        if (
            apply_extension_filter
            and self._allowed_extensions is not None
            and ext not in self._allowed_extensions
        ):
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

    def _iterate_files(
        self,
        directory: Path,
    ):
        """Yield all non-ignored files under *directory* recursively."""
        gitignore_cache: dict[Path, list[str]] = {}
        for item in directory.rglob("*"):
            if self._is_gitignored(item, directory, gitignore_cache):
                continue
            # Skip ignored directories anywhere in the path
            if any(part in _IGNORE_DIR_NAMES for part in item.parts):
                continue
            if item.is_file():
                yield item

    @staticmethod
    def _read_gitignore_patterns(directory: Path) -> list[str]:
        """Read .gitignore patterns from *directory* if present."""
        gitignore = directory / ".gitignore"
        if not gitignore.is_file():
            return []

        patterns: list[str] = []
        try:
            for raw_line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                # Escaped comments/exclamation are literal
                if line.startswith("\\#") or line.startswith("\\!"):
                    line = line[1:]
                patterns.append(line)
        except OSError:
            return []
        return patterns

    def _is_gitignored_from_ancestors(
        self,
        path: Path,
        cache: dict[Path, list[str]],
    ) -> bool:
        """Apply .gitignore files from path ancestors for explicit file-list inputs."""
        if not path.exists():
            return False

        # Apply nearest ancestor .gitignore files in root->leaf order.
        ancestors = list(path.parent.resolve().parents)
        ancestors.insert(0, path.parent.resolve())
        ancestors.reverse()
        ignored = False
        for base_dir in ancestors:
            patterns = cache.get(base_dir)
            if patterns is None:
                patterns = self._read_gitignore_patterns(base_dir)
                cache[base_dir] = patterns
            if not patterns:
                continue
            ignored = self._apply_gitignore_rules(path, base_dir, patterns, ignored)
        return ignored

    def _is_gitignored(
        self,
        path: Path,
        root: Path,
        cache: dict[Path, list[str]],
    ) -> bool:
        """Return True when *path* matches applicable .gitignore rules under *root*."""
        ignored = False
        for base_dir in self._gitignore_dirs_for_path(path, root):
            patterns = cache.get(base_dir)
            if patterns is None:
                patterns = self._read_gitignore_patterns(base_dir)
                cache[base_dir] = patterns
            if not patterns:
                continue
            ignored = self._apply_gitignore_rules(path, base_dir, patterns, ignored)
        return ignored

    @staticmethod
    def _gitignore_dirs_for_path(path: Path, root: Path) -> list[Path]:
        """Return directories (root..parent) whose .gitignore can affect *path*."""
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            return []
        dirs = [root]
        cur = root
        # .gitignore in the file's own directory applies; for directories use parent.
        upto = len(rel_parts) - 1
        for i in range(max(0, upto)):
            cur = cur / rel_parts[i]
            dirs.append(cur)
        return dirs

    def _apply_gitignore_rules(
        self,
        path: Path,
        base_dir: Path,
        patterns: list[str],
        ignored: bool,
    ) -> bool:
        """Apply one .gitignore file's rules to *path*, preserving prior state."""
        try:
            rel = path.relative_to(base_dir).as_posix()
        except ValueError:
            return ignored
        is_dir = path.is_dir()

        for rule in patterns:
            negate = rule.startswith("!")
            pat = rule[1:] if negate else rule
            dir_only = pat.endswith("/")
            if dir_only:
                pat = pat[:-1]
            anchored = pat.startswith("/")
            if anchored:
                pat = pat[1:]
            if not pat:
                continue

            if DocLoader._match_gitignore_pattern(
                rel=rel,
                pattern=pat,
                anchored=anchored,
                dir_only=dir_only,
                is_dir=is_dir,
            ):
                ignored = not negate

        return ignored

    @staticmethod
    def _match_gitignore_pattern(
        *,
        rel: str,
        pattern: str,
        anchored: bool,
        dir_only: bool,
        is_dir: bool,
    ) -> bool:
        """Match a simplified gitignore rule against a relative POSIX path."""
        parts = rel.split("/")

        # No slash pattern => applies to path segments (basename style)
        if "/" not in pattern:
            segs = parts if is_dir else parts[:-1]
            if not dir_only:
                segs = parts
            return any(fnmatchcase(seg, pattern) for seg in segs)

        # Slash pattern => full-path style
        if anchored:
            if dir_only:
                return rel == pattern or rel.startswith(pattern + "/")
            return fnmatchcase(rel, pattern) or rel.startswith(pattern + "/")

        # Unanchored slash pattern should match anywhere in the tree.
        rel_parts = rel.split("/")
        pat_parts = pattern.split("/")
        if len(pat_parts) > len(rel_parts):
            return False
        for i in range(0, len(rel_parts) - len(pat_parts) + 1):
            candidate = "/".join(rel_parts[i : i + len(pat_parts)])
            if fnmatchcase(candidate, pattern):
                return True
        return False

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
                    'PyPDF not installed. Run: pip install "aiforge[docs]" to add PDF support.'
                ) from None
        loader = PyPDFLoader(str(path))
        return loader.load()

    def _load_docx(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            raise ImportError(
                'python-docx not installed. Run: pip install "aiforge[docs]" for DOCX support.'
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
                raise RuntimeError(f"git clone failed for {git_url!r}:\n{result.stderr}")
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
