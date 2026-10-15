# Contributing to AI Forge

Thanks for helping out. Bug reports, fixes, docs improvements and new
providers or modules are all welcome.

## Before you start

- For a bug, open an issue with the steps to reproduce it, unless the fix is
  small enough to explain in the pull request itself.
- For a new feature or a change to a public API, open an issue first so the
  design can be agreed before you write the code.
- Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Development setup

AI Forge uses [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
git clone https://github.com/kushangpatel/aiforge.git
cd aiforge
uv sync --all-extras          # exact versions from uv.lock, all extras
cp .env.example .env          # only needed for live provider calls
```

Prefer pip? `pip install -e ".[all]"` works too, but CI tests against
`uv.lock`.

## Checks to run before opening a pull request

These are the same checks CI runs:

```bash
uv run ruff check aiforge/ tests/
uv run ruff format --check aiforge/ tests/
uv run mypy aiforge/
uv run pytest tests/unit -q
```

Unit tests never call a real provider, so they need no API keys.

If you change docs, also run `uv run mkdocs build --strict`.

If you change dependencies in `pyproject.toml`, run `uv lock` and commit the
updated `uv.lock`, or CI will fail. New lower bounds must be real: CI also runs
the unit tests on the oldest version each dependency allows.

## Pull requests

- Keep each pull request to one change, with tests for new behaviour.
- Add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for
  anything a user would notice.
- New public functions and classes need type hints (`mypy --strict` runs on
  the package) and a docstring.

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
