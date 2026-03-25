# CLI

AIForge provides a command-line interface built on [Typer](https://typer.tiangolo.com/)
for scaffolding projects, starting servers, launching the playground, and
managing vector store collections. The `aiforge` command is registered
automatically when you install the package.

---

## Table of Contents

- [CLI](#cli)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Global Help](#global-help)
  - [Commands](#commands)
    - [aiforge init](#aiforge-init)
    - [aiforge serve](#aiforge-serve)
    - [aiforge playground](#aiforge-playground)
    - [aiforge migrate](#aiforge-migrate)
  - [Configuration Priority](#configuration-priority)
  - [Environment Variables](#environment-variables)
  - [Exit Codes](#exit-codes)
  - [Examples](#examples)
    - [Scaffold a New Project](#scaffold-a-new-project)
    - [Start the API Server](#start-the-api-server)
    - [Launch the Playground](#launch-the-playground)
    - [Manage Collections](#manage-collections)
    - [Docker Compose Workflow](#docker-compose-workflow)
  - [Troubleshooting](#troubleshooting)
  - [See Also](#see-also)

---

## Installation

The `aiforge` command is available after installing the package:

```bash
pip install -e "."
# or
pip install aiforge
```

Verify the installation:

```bash
aiforge --help
```

Some commands require optional extras:

| Command | Required Extra | Install |
|---------|---------------|---------|
| `aiforge serve` | `server` | `pip install "aiforge[server]"` |
| `aiforge playground` | `playground` | `pip install "aiforge[playground]"` |
| `aiforge init` | *(none)* | — |
| `aiforge migrate` | *(none)* | — |

---

## Global Help

```
$ aiforge --help

 Usage: aiforge [OPTIONS] COMMAND [ARGS]...

 AIForge — plug-and-play AI toolkit

╭─ Commands ───────────────────────────────────────────────────────────────╮
│ init          Scaffold aiforge.yaml + .env.example in the current dir.  │
│ serve         Start the FastAPI REST API server.                        │
│ playground    Start the Streamlit interactive playground.               │
│ migrate       Inspect or manage vector store collections.               │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

## Commands

### aiforge init

Scaffold configuration files in the current directory.

```bash
aiforge init [--force]
```

Creates two files from bundled templates:

| File Created | Source Template | Purpose |
|-------------|----------------|---------|
| `aiforge.yaml` | `aiforge.yaml.example` | Main configuration |
| `.env` | `.env.example` | API keys and secrets |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--force` | `-f` | `bool` | `False` | Overwrite existing files |

**Behaviour:**

- If files already exist and `--force` is not set, they are silently skipped
- Prints a green checkmark for each file created
- After running, edit `aiforge.yaml` and `.env` with your provider and API keys

**Examples:**

```bash
# Scaffold a new project
mkdir my-project && cd my-project
aiforge init

# Reset config to defaults
aiforge init --force
```

---

### aiforge serve

Start the FastAPI REST API server.

```bash
aiforge serve [--host HOST] [--port PORT] [--reload]
```

Requires the `server` extra (`pip install "aiforge[server]"`).

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--host` | `str` | `0.0.0.0` | Bind address (all interfaces by default) |
| `--port` | `int` | `8000` | Bind port |
| `--reload` | `bool` | `False` | Enable hot-reload for development |

**Behaviour:**

- Launches a Uvicorn server running the FastAPI app at
  `aiforge/server/app.py`
- Prints the server URL and Swagger docs URL on startup
- Server runs until interrupted (Ctrl+C)
- With `--reload`, the server auto-restarts when source files change

**Server Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe (validates config) |
| `/stats` | GET | Cost and token usage statistics |
| `/summarize` | POST | Summariser module |
| `/qa` | POST | Document Q&A module |
| `/classify` | POST | Classifier module |
| `/extract` | POST | Extractor module |
| `/search` | POST | Semantic search module |
| `/generate` | POST | Generator module |
| `/chat` | POST | Chat agent module |
| `/moderate` | POST | Content moderator module |
| `/pipeline` | POST | Pipeline execution |
| `/jobs` | POST/GET | Job management |
| `/docs` | GET | Swagger UI |

**Configuration used:**

The server reads all settings from the standard configuration system
(`aiforge.yaml`, `.env`, environment variables). Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIFORGE_API_KEY` | — | Bearer token for auth (optional; if set, all endpoints require it) |
| `AIFORGE_CORS_ORIGINS` | `*` | CORS allowed origins |
| `AIFORGE_PROVIDER` | `openai` | LLM provider |
| `AIFORGE_MODEL` | `gpt-4o` | LLM model |
| `AIFORGE_LOG_LEVEL` | `INFO` | Log level |
| `AIFORGE_LOG_FORMAT` | `console` | Log format (`console` or `json`) |

**Examples:**

```bash
# Default — listen on all interfaces, port 8000
aiforge serve

# Localhost-only on custom port
aiforge serve --host localhost --port 9000

# Development mode with auto-reload
aiforge serve --reload

# Production with env vars
AIFORGE_API_KEY=my-secret-key AIFORGE_LOG_FORMAT=json aiforge serve
```

---

### aiforge playground

Launch the Streamlit interactive playground UI.

```bash
aiforge playground [--port PORT]
```

Requires the `playground` extra (`pip install "aiforge[playground]"`).

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--port` | `int` | `8501` | Streamlit server port |

**Behaviour:**

- Spawns a separate Streamlit process running `aiforge/playground/app.py`
- Prints the playground URL on startup
- The playground reads all AIForge settings from the standard configuration
- Blocks until the Streamlit process exits (Ctrl+C to stop)

**Playground Pages:**

| Page | Module |
|------|--------|
| Summarize | `Summarizer` |
| Q&A | `DocQA` |
| Search | `SemanticSearch` |
| Classify | `Classifier` |
| Extract | `Extractor` |
| Generate | `Generator` |
| Chat | `ChatAgent` |
| Moderate | `Moderator` |
| Pipeline | `Pipeline` |
| Settings | Configuration dashboard |

**Examples:**

```bash
# Default port
aiforge playground

# Custom port
aiforge playground --port 8080
```

---

### aiforge migrate

Inspect or manage vector store collections.

```bash
aiforge migrate [--all] [--status] [--collection NAME] [--delete NAME]
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--all` | `-a` | `bool` | `False` | List all collections |
| `--status` | `-s` | `bool` | `False` | Show collection statistics |
| `--collection` | `-c` | `str` | — | Inspect a specific collection |
| `--delete` | `-d` | `str` | — | Delete a named collection (with confirmation) |

At least one flag is required. If no action flag is provided, the command
prints a usage hint and exits.

**Behaviour:**

- Initialises the vector store using `AIFORGE_CHROMA_PERSIST_DIR`
- **`--all` / `--status`**: Displays a Rich table with collection names and
  chunk counts
- **`--collection`**: Filters the listing to a single collection
- **`--delete`**: Prompts for confirmation before deleting the collection
  (irreversible)

**Output format (Rich table):**

```
┌──────────────┬────────┐
│ Name         │ Chunks │
├──────────────┼────────┤
│ default      │  1,247 │
│ products     │     89 │
│ support-docs │    403 │
└──────────────┴────────┘
```

**Configuration used:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AIFORGE_CHROMA_PERSIST_DIR` | `./aiforge_data` | ChromaDB storage directory |

**Examples:**

```bash
# List all collections with chunk counts
aiforge migrate --all

# List with detailed stats
aiforge migrate -a -s

# Inspect a single collection
aiforge migrate --collection my_docs

# Delete a collection (prompts for confirmation)
aiforge migrate --delete old_collection
```

---

## Configuration Priority

CLI commands interact with AIForge's layered configuration system. The
priority order (lowest to highest):

```
1. Built-in defaults in code
2. aiforge.yaml (if present in cwd)
3. .env file (if present in cwd)
4. Environment variables (AIFORGE_* prefix)
5. CLI flags (--host, --port, etc.)
```

CLI flags like `--host`, `--port`, and `--reload` are command-specific and
override nothing in the settings system — they are passed directly to the
underlying server process. All other configuration (provider, model, API keys,
etc.) flows through the standard settings hierarchy.

**Example flow:**

```bash
# aiforge.yaml has provider: openai
# .env has AIFORGE_MODEL=gpt-4o-mini
# Shell has: export AIFORGE_MODEL=gpt-4o

aiforge serve --port 9000
# → Port: 9000 (from CLI flag)
# → Model: gpt-4o (env var wins over .env)
# → Provider: openai (from aiforge.yaml)
```

---

## Environment Variables

Common environment variables used across CLI commands:

| Variable | Default | Used By | Description |
|----------|---------|---------|-------------|
| `AIFORGE_PROVIDER` | `openai` | `serve`, `playground` | LLM provider |
| `AIFORGE_MODEL` | `gpt-4o` | `serve`, `playground` | LLM model |
| `AIFORGE_OPENAI_API_KEY` | — | `serve`, `playground` | OpenAI API key |
| `AIFORGE_ANTHROPIC_API_KEY` | — | `serve`, `playground` | Anthropic API key |
| `AIFORGE_VOYAGE_API_KEY` | — | `serve`, `playground` | Voyage AI embedding key |
| `AIFORGE_AZURE_OPENAI_API_KEY` | — | `serve`, `playground` | Azure OpenAI key |
| `AIFORGE_AZURE_OPENAI_ENDPOINT` | — | `serve`, `playground` | Azure resource URL |
| `AIFORGE_OLLAMA_BASE_URL` | `http://localhost:11434` | `serve`, `playground` | Ollama server URL |
| `AIFORGE_CHROMA_PERSIST_DIR` | `./aiforge_data` | `migrate` | ChromaDB data directory |
| `AIFORGE_API_KEY` | — | `serve` | API auth bearer token |
| `AIFORGE_LOG_LEVEL` | `INFO` | `serve` | Logging level |
| `AIFORGE_LOG_FORMAT` | `console` | `serve` | Log format (`console` / `json`) |
| `AIFORGE_CORS_ORIGINS` | `*` | `serve` | CORS allowed origins |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Missing dependency (`uvicorn` or `streamlit` not installed) |

All commands use Rich console output for coloured, formatted messages:
- Green checkmarks for success
- Red text for errors with install instructions
- Yellow text for warnings or no-action hints

---

## Examples

### Scaffold a New Project

```bash
mkdir my-ai-app && cd my-ai-app
aiforge init
# ✓ Created aiforge.yaml
# ✓ Created .env

# Edit your configuration
nano aiforge.yaml    # Set provider, model, etc.
nano .env            # Add API keys
```

### Start the API Server

```bash
# Quick start
aiforge serve

# Production-like
AIFORGE_API_KEY=secret-token \
AIFORGE_LOG_FORMAT=json \
aiforge serve --host 0.0.0.0 --port 8000

# Development with hot-reload
aiforge serve --reload --port 9000
```

### Launch the Playground

```bash
# Default
aiforge playground

# Alongside the API server (two terminals)
# Terminal 1:
aiforge serve --port 8000
# Terminal 2:
AIFORGE_API_BASE=http://localhost:8000 aiforge playground --port 8501
```

### Manage Collections

```bash
# See what's indexed
aiforge migrate --all

# Check a specific collection
aiforge migrate -c customer-docs -s

# Clean up old data
aiforge migrate --delete test-collection
# Delete collection 'test-collection'? This cannot be undone. [y/N]: y
# ✓ Deleted collection 'test-collection'.
```

### Docker Compose Workflow

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    command: aiforge serve --host 0.0.0.0 --port 8000

  playground:
    build: .
    ports:
      - "8501:8501"
    env_file: .env
    environment:
      - AIFORGE_API_BASE=http://api:8000
    command: aiforge playground --port 8501
    depends_on:
      - api
```

```bash
docker compose up -d
# API:        http://localhost:8000
# Swagger:    http://localhost:8000/docs
# Playground: http://localhost:8501
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `command not found: aiforge` | Package not installed or venv not activated | `pip install -e "."` and activate your virtual environment |
| `uvicorn not installed` when running `serve` | Missing server extra | `pip install "aiforge[server]"` |
| `Streamlit not installed` when running `playground` | Missing playground extra | `pip install "aiforge[playground]"` |
| `aiforge init` skips files silently | Files already exist | Use `aiforge init --force` to overwrite |
| `aiforge migrate` prints "No action specified" | No flag provided | Add `--all`, `--status`, `--collection`, or `--delete` |
| Server starts but requests fail with 401 | `AIFORGE_API_KEY` is set | Include `Authorization: Bearer <key>` header in requests |
| Server starts but LLM calls fail | API keys not configured | Edit `.env` or set `AIFORGE_OPENAI_API_KEY` etc. |
| Playground shows stale config | Config loaded at process start | Restart: Ctrl+C then `aiforge playground` |
| `migrate --delete` has no effect | Collection name doesn't exist | Check exact name with `aiforge migrate --all` |
| Port already in use | Another process on that port | Use `--port <other>` or stop the conflicting process |

---

## See Also

- [Quick Start](quickstart.md) — Getting started guide
- [Configuration](configuration.md) — Full configuration reference
- [API Server](api-server.md) — REST API endpoints and authentication
- [API Auth](api-auth.md) — API key authentication details
- [Playground](playground.md) — Interactive UI documentation
- [Providers](providers.md) — Provider setup and configuration
