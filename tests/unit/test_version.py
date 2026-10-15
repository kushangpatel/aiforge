"""Unit tests for the package version and the release helper script."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.version import Version

import aiforge
from aiforge.__version__ import __version__

ROOT = Path(__file__).resolve().parents[2]


def _load_release_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "release_check", ROOT / "scripts" / "release_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_check = _load_release_check()


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    def test_version_is_pep440(self) -> None:
        assert str(Version(__version__)) == __version__

    def test_package_reexports_version(self) -> None:
        assert aiforge.__version__ == __version__

    def test_pyproject_reads_version_from_version_file(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert "version" not in pyproject["project"]
        assert "version" in pyproject["project"]["dynamic"]
        dynamic = pyproject["tool"]["setuptools"]["dynamic"]["version"]
        assert dynamic == {"attr": "aiforge.__version__.__version__"}

    def test_release_script_reads_same_version(self) -> None:
        assert release_check.package_version() == __version__

    def test_changelog_has_section_for_current_version(self) -> None:
        assert release_check.changelog_section(__version__)


# ---------------------------------------------------------------------------
# Release helper
# ---------------------------------------------------------------------------


class TestReleaseCheck:
    def test_check_tag_accepts_matching_tag(self) -> None:
        release_check.check_tag("v1.2.3", "1.2.3")

    @pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3rc1", "release-1.2.3"])
    def test_check_tag_rejects_mismatch(self, tag: str) -> None:
        with pytest.raises(ValueError, match="does not match"):
            release_check.check_tag(tag, "1.2.3")

    def test_package_version_missing_assignment(self, tmp_path: Path) -> None:
        f = tmp_path / "__version__.py"
        f.write_text("VERSION = '1.0'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No __version__"):
            release_check.package_version(f)

    def test_changelog_section_extracts_only_that_version(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n"
            "## [Unreleased]\n\n- wip\n\n"
            "## [1.1.0] - 2026-01-02\n\n### Added\n\n- new thing\n\n"
            "## [1.0.0] - 2026-01-01\n\n- first\n\n"
            "[1.1.0]: https://example.com/1.1.0\n",
            encoding="utf-8",
        )
        assert release_check.changelog_section("1.1.0", f) == "### Added\n\n- new thing"
        assert release_check.changelog_section("1.0.0", f) == "- first"

    def test_changelog_section_does_not_match_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("## [1.10.0]\n\n- ten\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no '## \\[1.1.0\\]'"):
            release_check.changelog_section("1.1.0", f)

    def test_changelog_section_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "CHANGELOG.md"
        f.write_text("## [1.0.0]\n\n## [0.9.0]\n\n- old\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            release_check.changelog_section("1.0.0", f)

    def test_main_exit_codes(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert release_check.main(["version"]) == 0
        assert capsys.readouterr().out.strip() == __version__
        assert release_check.main(["check-tag", f"v{__version__}"]) == 0
        assert release_check.main(["check-tag", "v0.0.0"]) == 1
        assert release_check.main(["bogus"]) == 2
