"""Unit tests for DocLoader and TextSplitter."""

from __future__ import annotations

from typing import Any

import pytest

from aiforge.core.doc_loader import DocLoader


class TestDocLoaderFile:
    def test_load_text_file(self, tmp_path: Any) -> None:
        f = tmp_path / "sample.py"
        f.write_text("print('hello')\n")
        loader = DocLoader()
        docs, stats = loader.load(str(f))
        assert len(docs) == 1
        assert "hello" in docs[0].page_content
        assert stats.files_processed == 1
        assert "py" in stats.languages

    def test_load_markdown_file(self, tmp_path: Any) -> None:
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome content.\n")
        loader = DocLoader()
        docs, stats = loader.load(str(f))
        assert len(docs) == 1
        assert "Title" in docs[0].page_content

    def test_load_directory_recursive(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.md").write_text("hello")
        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))
        assert stats.files_processed == 2

    def test_ignores_pycache(self, tmp_path: Any) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.py").write_text("junk")
        (tmp_path / "real.py").write_text("x = 1")
        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))
        assert stats.files_processed == 1

    def test_ignores_binary_extensions(self, tmp_path: Any) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "code.py").write_text("pass")
        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))
        assert stats.files_processed == 1

    def test_extension_whitelist(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.md").write_text("y")
        loader = DocLoader(file_extensions=[".py"])
        docs, stats = loader.load(str(tmp_path))
        assert stats.files_processed == 1
        assert stats.skipped_files  # b.md was skipped

    def test_load_file_list(self, tmp_path: Any) -> None:
        f1 = tmp_path / "one.txt"
        f2 = tmp_path / "two.txt"
        f1.write_text("one")
        f2.write_text("two")
        loader = DocLoader()
        docs, stats = loader.load([str(f1), str(f2)])
        assert stats.files_processed == 2

    def test_honors_root_gitignore_file_and_dir_patterns(self, tmp_path: Any) -> None:
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
        (tmp_path / "keep.py").write_text("print('ok')\n")
        (tmp_path / "debug.log").write_text("secret\n")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "generated.py").write_text("print('gen')\n")

        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))

        assert stats.files_processed == 1
        assert len(docs) == 1
        assert docs[0].metadata["source"].endswith("keep.py")

    def test_honors_gitignore_negation(self, tmp_path: Any) -> None:
        (tmp_path / ".gitignore").write_text("*.py\n!keep.py\n")
        (tmp_path / "keep.py").write_text("print('ok')\n")
        (tmp_path / "drop.py").write_text("print('no')\n")

        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))

        assert stats.files_processed == 1
        assert len(docs) == 1
        assert docs[0].metadata["source"].endswith("keep.py")

    def test_gitignore_unanchored_slash_does_not_match_partial_segment(self, tmp_path: Any) -> None:
        (tmp_path / ".gitignore").write_text("foo/bar\n")
        (tmp_path / "x" / "foo").mkdir(parents=True)
        (tmp_path / "x" / "foo" / "barista.txt").write_text("keep me\n")
        (tmp_path / "x" / "foo" / "bar").mkdir()
        (tmp_path / "x" / "foo" / "bar" / "drop.txt").write_text("drop me\n")

        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))

        sources = [d.metadata["source"] for d in docs]
        assert any(s.endswith("barista.txt") for s in sources)
        assert not any(s.endswith("drop.txt") for s in sources)

    def test_honors_nested_gitignore(self, tmp_path: Any) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / ".gitignore").write_text("*.tmp\n")
        (tmp_path / "pkg" / "a.tmp").write_text("ignored\n")
        (tmp_path / "pkg" / "keep.py").write_text("print('ok')\n")

        loader = DocLoader()
        docs, stats = loader.load(str(tmp_path))

        assert stats.files_processed == 1
        assert docs[0].metadata["source"].endswith("keep.py")

    def test_file_list_honors_gitignore(self, tmp_path: Any) -> None:
        (tmp_path / ".gitignore").write_text("*.secret\n")
        keep = tmp_path / "keep.py"
        secret = tmp_path / "token.secret"
        keep.write_text("print('ok')\n")
        secret.write_text("123\n")

        loader = DocLoader()
        docs, stats = loader.load([str(keep), str(secret)])

        assert stats.files_processed == 1
        assert len(docs) == 1
        assert docs[0].metadata["source"].endswith("keep.py")

    def test_missing_file_raises(self) -> None:
        loader = DocLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.py")

    def test_git_url_validation_rejects_local_path(self) -> None:
        """Local paths ending with .git must be rejected (SSRF prevention)."""
        loader = DocLoader()
        with pytest.raises(ValueError, match="must start with one of"):
            loader.load("/etc/passwd.git")

    def test_git_url_validation_rejects_relative_path(self) -> None:
        loader = DocLoader()
        with pytest.raises(ValueError, match="must start with one of"):
            loader.load("../some/repo.git")


class TestTextSplitter:
    def test_split_plain_text(self) -> None:
        from aiforge.core.text_splitter import TextSplitter

        splitter = TextSplitter()
        long_text = "word " * 500  # ~2500 chars
        chunks = splitter.split_text(long_text, ext="txt")
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_split_python_code(self) -> None:
        from aiforge.core.text_splitter import TextSplitter

        splitter = TextSplitter()
        code = "def foo():\n    pass\n\n" * 100
        chunks = splitter.split_text(code, ext="py")
        assert len(chunks) >= 1
