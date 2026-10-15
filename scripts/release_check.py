"""Release helpers used by .github/workflows/release.yml.

Reads the package version from ``aiforge/__version__.py`` without importing
the package, checks a git tag against it, and extracts that version's section
from ``CHANGELOG.md`` for the GitHub Release notes.

Usage::

    python scripts/release_check.py version
    python scripts/release_check.py check-tag v0.2.0
    python scripts/release_check.py notes 0.2.0 > notes.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "aiforge" / "__version__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def package_version(version_file: Path = VERSION_FILE) -> str:
    """Return the version string declared in ``aiforge/__version__.py``."""
    match = _VERSION_RE.search(version_file.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No __version__ assignment found in {version_file}")
    return match.group(1)


def check_tag(tag: str, version: str) -> None:
    """Raise ``ValueError`` unless ``tag`` is ``v<version>``."""
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match the package version {version!r}")


def changelog_section(version: str, changelog: Path = CHANGELOG) -> str:
    """Return the body of the ``## [<version>]`` section of the changelog.

    Raises ``ValueError`` if the section is missing or empty.
    """
    lines = changelog.read_text(encoding="utf-8").splitlines()
    header = re.compile(rf"^## \[{re.escape(version)}\](\s|$)")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        raise ValueError(f"CHANGELOG.md has no '## [{version}]' section")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## ") or re.match(r"^\[[^\]]+\]: ", line):
            break
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        raise ValueError(f"CHANGELOG.md section for {version} is empty")
    return text


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    command, args = argv[0], argv[1:]
    try:
        if command == "version" and not args:
            print(package_version())
        elif command == "check-tag" and len(args) == 1:
            version = package_version()
            check_tag(args[0], version)
            print(f"Tag {args[0]} matches package version {version}")
        elif command == "notes" and len(args) == 1:
            print(changelog_section(args[0]))
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
