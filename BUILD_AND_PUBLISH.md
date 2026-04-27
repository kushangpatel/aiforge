# AIForge — Build & Publish Guide

This document covers every way to build, distribute, and consume the AIForge
Python package: local development, internal-only publishing, and public PyPI.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.12  |
| pip         | ≥ 23.0  |
| build       | latest (`pip install build`) |
| twine       | latest (`pip install twine`) — only needed for publishing |

```bash
pip install build twine
```

---

## 1. Building the Package

AIForge uses **setuptools** as the build backend (configured in `pyproject.toml`).

```bash
cd aiforge/
python -m build
```

This produces two artifacts in `dist/`:

```
dist/
  aiforge-0.1.0.tar.gz          # source distribution (sdist)
  aiforge-0.1.0-py3-none-any.whl  # wheel (binary distribution)
```

> **Bump the version** in `pyproject.toml` → `[project] version = "..."` before
> each release.

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
cd /path/to/aiforge && python -m build

# Install (on any machine)
pip install dist/aiforge-0.1.0-py3-none-any.whl
# With extras:
pip install "dist/aiforge-0.1.0-py3-none-any.whl[server,playground]"
```

### Option D — Git-Based pip Install

Consumers install directly from a Git repo (no registry required):

```bash
# Latest from main
pip install "git+https://github.com/your-org/aiforge.git@main"

# Pinned to a tag
pip install "git+https://github.com/your-org/aiforge.git@v0.1.0"

# With extras
pip install "git+https://github.com/your-org/aiforge.git@v0.1.0#egg=aiforge[server,docs]"
```

Tag releases for stability:

```bash
git tag v0.1.0
git push origin v0.1.0
```

### Option E — Reference in requirements.txt

```txt
# From local path
aiforge @ file:///absolute/path/to/aiforge

# From Git
aiforge @ git+https://github.com/your-org/aiforge.git@v0.1.0

# From Git with extras (PEP 508)
aiforge[server] @ git+https://github.com/your-org/aiforge.git@v0.1.0
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
python -m build
twine upload --repository codeartifact dist/*
```

**Consume:**

```bash
# Authenticate pip
aws codeartifact login --tool pip \
  --domain mycompany \
  --repository python-internal

# Install normally — CodeArtifact proxies PyPI for other deps
pip install aiforge
pip install "aiforge[server,playground]"
```

Or set the index URL explicitly:

```bash
pip install aiforge \
  --index-url https://mycompany-123456.d.codeartifact.us-east-1.amazonaws.com/pypi/python-internal/simple/
```

### Option B — GitHub Packages

**Configure `~/.pypirc`:**

```ini
[distutils]
index-servers = github

[github]
repository = https://maven.pkg.github.com/your-org/aiforge
username = __token__
password = ghp_YOUR_PERSONAL_ACCESS_TOKEN
```

**Publish:**

```bash
python -m build
twine upload --repository github dist/*
```

**Consume:**

```bash
pip install aiforge \
  --index-url https://maven.pkg.github.com/your-org/aiforge/simple/
```

### Option C — JFrog Artifactory

**Publish:**

```bash
python -m build
twine upload \
  --repository-url https://artifactory.yourcompany.com/api/pypi/pypi-local \
  --username your-user \
  --password your-token \
  dist/*
```

**Consume:**

```bash
pip install aiforge \
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
pip install aiforge --index-url http://your-server:3141/root/pypi/+simple/
```

---

## 4. Publishing to Public PyPI

### Create a PyPI Account & API Token

1. Register at https://pypi.org/account/register/
2. Go to **Account Settings → API tokens** → create a token scoped to the `aiforge` project.

### Configure `~/.pypirc`

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN
```

Or use environment variables instead:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_API_TOKEN
```

### Build & Upload

```bash
python -m build
twine upload dist/*
```

### Test on TestPyPI First (Recommended)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ aiforge
```

### Consumers Install With:

```bash
pip install aiforge
pip install "aiforge[all]"
pip install "aiforge[server,playground,anthropicai]"
```

---

## 5. Available Extras

These are defined in `pyproject.toml` under `[project.optional-dependencies]`:

| Extra           | What it adds                                |
|-----------------|---------------------------------------------|
| `openai`        | OpenAI LLM + embeddings                     |
| `anthropicai`   | Anthropic Claude + Voyage embeddings         |
| `local`         | Ollama (local LLM)                           |
| `agentic`       | LangGraph for agentic workflows              |
| `pinecone`      | Pinecone vector store                        |
| `qdrant`        | Qdrant vector store                          |
| `pgvector`      | PostgreSQL pgvector store                    |
| `server`        | FastAPI + Uvicorn HTTP API                   |
| `playground`    | Streamlit interactive playground             |
| `observability` | OpenTelemetry + LangSmith tracing            |
| `docs`          | PDF + DOCX document ingestion                |
| `dev`           | pytest, mypy, ruff, coverage, httpx          |
| `all`           | Everything above                             |

Install multiple extras with comma separation:

```bash
pip install "aiforge[server,playground,anthropicai,docs]"
```

---

## 6. Versioning

The single source of truth for the version is:

```
pyproject.toml → [project] → version
```

Follow [Semantic Versioning](https://semver.org/):

- **Patch** (`0.1.1`): bug fixes, no API changes
- **Minor** (`0.2.0`): new features, backward-compatible
- **Major** (`1.0.0`): breaking API changes

Tag each release in Git:

```bash
# After bumping version in pyproject.toml
git add pyproject.toml
git commit -m "release: v0.2.0"
git tag v0.2.0
git push origin main --tags
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Build | `python -m build` |
| Editable install | `pip install -e ".[dev]"` |
| Install from wheel | `pip install dist/aiforge-*.whl` |
| Install from Git | `pip install "git+https://github.com/org/aiforge.git@v0.1.0"` |
| Publish to PyPI | `twine upload dist/*` |
| Publish to CodeArtifact | `aws codeartifact login --tool twine ... && twine upload --repository codeartifact dist/*` |
| Run tests before release | `python -m pytest tests/ -q` |
| Check package metadata | `twine check dist/*` |
