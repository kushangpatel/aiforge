# AIForge — Build & Publish Guide

This document covers every way to build, distribute, and consume the AIForge
Python package: local development, internal-only publishing, and public PyPI.

Releases to PyPI are automated by `.github/workflows/release.yml`: pushing a
`vX.Y.Z` tag builds, checks and publishes the package, then creates a GitHub
Release. See [section 4](#4-publishing-to-public-pypi).

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.12  |
| [uv](https://docs.astral.sh/uv/) | recent (builds, checks and installs) |

`pip` + `python -m build` + `twine` work too; the commands below use uv
because that is what CI runs.

---

## 1. Building the Package

AIForge uses **setuptools** (≥ 77) as the build backend, configured in
`pyproject.toml`.

```bash
uv build                         # sdist + wheel into dist/
uvx twine check --strict dist/*  # metadata and README render check
```

This produces two artifacts in `dist/`:

```
dist/
  aiforge_kit-0.2.0.tar.gz          # source distribution (sdist)
  aiforge_kit-0.2.0-py3-none-any.whl  # wheel
```

The wheel contains only the `aiforge` package plus its runtime data files
(`py.typed`, `templates/`, the Streamlit `playground/pages/`). Tests, docs,
`development-docs/`, `scripts/` and any local `aiforge_data/` are left out.

To prove a wheel is self-contained, install it into a fresh virtualenv and run
the smoke test from outside the repository (CI does this on every PR, in
`.github/workflows/package.yml`):

```bash
uv venv /tmp/aiforge-wheel
uv pip install --python /tmp/aiforge-wheel/bin/python dist/*.whl
cd /tmp && /tmp/aiforge-wheel/bin/python -I /path/to/aiforge/scripts/smoke_wheel.py
```

> The version is defined once, in `aiforge/__version__.py`; `pyproject.toml`
> reads it (`dynamic = ["version"]`). See [section 6](#6-versioning).

---

## 2. Using Without Publishing (Local Development)

No registry is needed. Pick the option that fits your workflow.

### Option A — Editable Install (Recommended for Active Development)

```bash
cd /path/to/aiforge
pip install -e ".[dev,playground,server]"
```

- Changes to source files are reflected immediately — no reinstall needed.
- The `aiforge` CLI is available in the active environment.
- Use whichever extras you need: `dev`, `server`, `playground`, `docs`, `all`, etc.

### Option B — Direct Path Install

Install from the local directory (non-editable):

```bash
pip install /path/to/aiforge
# With extras:
pip install "/path/to/aiforge[server,docs]"
```

### Option C — Install from a Built Wheel

Build once, then distribute the `.whl` file to teammates:

```bash
# Build
cd /path/to/aiforge && uv build

# Install (on any machine)
pip install dist/aiforge_kit-0.2.0-py3-none-any.whl
# With extras:
pip install "dist/aiforge_kit-0.2.0-py3-none-any.whl[server,playground]"
```

### Option D — Git-Based pip Install

Consumers install directly from a Git repo (no registry required):

```bash
# Latest from main
pip install "git+https://github.com/kushangpatel/aiforge.git@main"

# Pinned to a tag
pip install "git+https://github.com/kushangpatel/aiforge.git@v0.2.0"

# With extras
pip install "git+https://github.com/kushangpatel/aiforge.git@v0.2.0#egg=aiforge-kit[server,docs]"
```

Tag releases for stability:

```bash
git tag v0.2.0
git push origin v0.2.0
```

### Option E — Reference in requirements.txt

```txt
# From local path
aiforge @ file:///absolute/path/to/aiforge

# From Git
aiforge @ git+https://github.com/kushangpatel/aiforge.git@v0.2.0

# From Git with extras (PEP 508)
aiforge-kit[server] @ git+https://github.com/kushangpatel/aiforge.git@v0.2.0
```

---

## 3. Publishing Internally Only (Private Registry)

### Option A — AWS CodeArtifact

Best fit if your team already uses AWS.

**One-time setup:**

```bash
# Create a CodeArtifact domain and repository (once)
aws codeartifact create-domain --domain mycompany
aws codeartifact create-repository --domain mycompany --repository python-internal

# Connect upstream PyPI so pip can resolve non-internal deps
aws codeartifact associate-external-connection \
  --domain mycompany \
  --repository python-internal \
  --external-connection public:pypi
```

**Publish:**

```bash
# Authenticate twine (token valid for 12h by default)
aws codeartifact login --tool twine \
  --domain mycompany \
  --repository python-internal

# Build & upload
uv build
twine upload --repository codeartifact dist/*
```

**Consume:**

```bash
# Authenticate pip
aws codeartifact login --tool pip \
  --domain mycompany \
  --repository python-internal

# Install normally — CodeArtifact proxies PyPI for other deps
pip install aiforge-kit
pip install "aiforge-kit[server,playground]"
```

Or set the index URL explicitly:

```bash
pip install aiforge-kit \
  --index-url https://mycompany-123456.d.codeartifact.us-east-1.amazonaws.com/pypi/python-internal/simple/
```

### Option B — GitHub Packages

**Configure `~/.pypirc`:**

```ini
[distutils]
index-servers = github

[github]
repository = https://maven.pkg.github.com/kushangpatel/aiforge
username = __token__
password = ghp_YOUR_PERSONAL_ACCESS_TOKEN
```

**Publish:**

```bash
uv build
twine upload --repository github dist/*
```

**Consume:**

```bash
pip install aiforge-kit \
  --index-url https://maven.pkg.github.com/kushangpatel/aiforge/simple/
```

### Option C — JFrog Artifactory

**Publish:**

```bash
uv build
twine upload \
  --repository-url https://artifactory.yourcompany.com/api/pypi/pypi-local \
  --username your-user \
  --password your-token \
  dist/*
```

**Consume:**

```bash
pip install aiforge-kit \
  --index-url https://artifactory.yourcompany.com/api/pypi/pypi-local/simple/
```

### Option D — Sonatype Nexus

```bash
twine upload \
  --repository-url https://nexus.yourcompany.com/repository/pypi-internal/ \
  dist/*
```

### Option E — Lightweight: devpi (Self-Hosted)

A minimal PyPI-compatible server you can run on any VM:

```bash
# Server setup
pip install devpi-server devpi-web
devpi-server --init
devpi-server --start --host 0.0.0.0 --port 3141

# Publish
devpi use http://your-server:3141/root/pypi
devpi login root
devpi upload dist/*

# Consume
pip install aiforge-kit --index-url http://your-server:3141/root/pypi/+simple/
```

---

## 4. Publishing to Public PyPI

Publishing is done by GitHub Actions with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/): GitHub
proves to PyPI which repository, workflow and environment is uploading, so no
API token is stored anywhere.

`.github/workflows/release.yml` runs:

| Trigger | Jobs |
|---------|------|
| push of a tag `v*` | build → check tag = package version → `twine check` → smoke-install the wheel → **publish to PyPI** (environment `pypi`) → **GitHub Release** with the CHANGELOG section as notes |
| *Run workflow* (workflow_dispatch) | the same build and checks → **publish to TestPyPI** (environment `testpypi`); no tag check, no GitHub Release |

### One-time setup (repository owner)

**1. Project name.** The package is published as **`aiforge-kit`**
(`pip install aiforge-kit`); the import name is still `aiforge`. `aiforge`
itself could not be used: PyPI treats it as too similar to the existing
`ai-forge` and `ai-forge-sdk` projects, and on TestPyPI it belongs to an
unrelated project. `aiforge-kit` was free on both indexes in October 2026.

**2. Register the trusted publisher on PyPI.** Sign in at
<https://pypi.org>, open *Your account → Publishing → Add a new pending
publisher* (a pending publisher creates the project on its first upload) and
enter:

| Field | Value |
|-------|-------|
| PyPI Project Name | `aiforge-kit` |
| Owner | `kushangpatel` |
| Repository name | `aiforge` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

**3. Do the same on TestPyPI** (a separate account at
<https://test.pypi.org>), with environment name `testpypi`.

**4. Create the GitHub environments.** In the repository, *Settings →
Environments → New environment*:

- `pypi`: recommended protection is *Required reviewers* (yourself) and
  *Deployment branches and tags* limited to tags matching `v*`.
- `testpypi`: no protection needed.

No secrets are needed in either environment.

### Releasing a version

1. Make sure `main` is green (the CI and Package workflows).
2. Bump `__version__` in `aiforge/__version__.py`.
3. In `CHANGELOG.md`, move the finished items from `## [Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` section and update the compare links at the
   bottom. The release fails if this section is missing.
4. Merge to `main`, then tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "AI Forge 0.2.0"
   git push origin v0.2.0
   ```

5. Approve the `pypi` deployment in the Actions tab if you added a reviewer.
   The workflow publishes to PyPI and creates the GitHub Release.

If the tag does not match `__version__`, the workflow stops before
publishing: delete the tag (`git push --delete origin vX.Y.Z`), fix the
version and tag again. A version that reached PyPI can never be uploaded
again, even after it is deleted; release a new patch version instead.

### Dry run on TestPyPI

*Actions → Release → Run workflow* on any branch builds the package and
uploads it to TestPyPI. Install it with PyPI as the fallback index for
dependencies:

```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ aiforge
```

Each version can be uploaded to TestPyPI only once; re-runs skip files that
are already there.

### Manual upload (fallback)

If Actions is unavailable, an API token still works:

```bash
uv build
uvx twine check --strict dist/*
uvx twine upload dist/*            # prompts for a pypi-... API token
```

### Consumers install with

```bash
pip install aiforge-kit
pip install "aiforge-kit[all]"
pip install "aiforge-kit[server,playground,anthropic]"
```

---

## 5. Available Extras

These are defined in `pyproject.toml` under `[project.optional-dependencies]`:

| Extra           | What it adds                                |
|-----------------|---------------------------------------------|
| `openai`        | OpenAI LLM + embeddings (also in the core install) |
| `anthropic`     | Anthropic Claude + Voyage embeddings (`anthropicai` is an alias) |
| `local`         | Ollama (local LLM)                          |
| `agentic`       | LangGraph for agentic workflows             |
| `pinecone`      | Pinecone vector store                       |
| `qdrant`        | Qdrant vector store                         |
| `pgvector`      | PostgreSQL pgvector store                   |
| `server`        | FastAPI + Uvicorn HTTP API                  |
| `playground`    | Streamlit interactive playground            |
| `observability` | OpenTelemetry + LangSmith tracing           |
| `docs`          | PDF + DOCX document ingestion               |
| `mcp`           | MCP client (stdio, Streamable HTTP, SSE)    |
| `site`          | MkDocs + Material, to build the docs site   |
| `dev`           | pytest, mypy, ruff, coverage, httpx         |
| `all`           | Everything except `site`                    |

Install multiple extras with comma separation:

```bash
pip install "aiforge-kit[server,playground,anthropic,docs]"
```

---

## 6. Versioning

The single source of truth for the version is `aiforge/__version__.py`:

```python
__version__ = "0.2.0"
```

`pyproject.toml` declares `dynamic = ["version"]`, and setuptools reads that
file statically at build time. `tests/unit/test_version.py` checks the wiring
and that `CHANGELOG.md` has a section for the current version; the release
workflow checks that the pushed tag is `v` + that version
(`python scripts/release_check.py check-tag vX.Y.Z`).

Follow [Semantic Versioning](https://semver.org/). While the version is 0.x:

- **Patch** (`0.2.1`): bug fixes
- **Minor** (`0.3.0`): new features; may include breaking changes, listed in
  the changelog
- **Major** (`1.0.0`): first stable API

Pre-releases use PEP 440 suffixes (`0.3.0rc1`, tag `v0.3.0rc1`); their GitHub
Release is marked as a pre-release.

---

## Quick Reference

| Goal | Command |
|------|---------|
| Build | `uv build` |
| Check package metadata | `uvx twine check --strict dist/*` |
| Smoke-test the wheel | `python -I scripts/smoke_wheel.py` (fresh venv, outside the repo) |
| Development install | `uv sync --all-extras` or `pip install -e ".[dev]"` |
| Install from wheel | `pip install dist/aiforge_kit-*.whl` |
| Install from Git | `pip install "git+https://github.com/kushangpatel/aiforge.git@v0.2.0"` |
| Current version | `python scripts/release_check.py version` |
| Release to PyPI | bump version + CHANGELOG, merge, then `git tag -a vX.Y.Z -m "..."` and `git push origin vX.Y.Z` |
| Dry run on TestPyPI | Actions → Release → Run workflow |
| Publish to CodeArtifact | `aws codeartifact login --tool twine ... && twine upload --repository codeartifact dist/*` |
| Run tests before release | `uv run pytest tests/unit -q` |
