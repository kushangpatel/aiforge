# CLI

AIForge provides a command-line interface built on [Typer](https://typer.tiangolo.com/)
for scaffolding projects, starting servers, launching the playground,
running every SDK module from the terminal, and managing infrastructure.
The `aiforge` command is registered automatically when you install the package.

---

## Table of Contents

- [CLI](#cli)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Global Help](#global-help)
  - [Infrastructure Commands](#infrastructure-commands)
    - [aiforge init](#aiforge-init)
    - [aiforge serve](#aiforge-serve)
    - [aiforge playground](#aiforge-playground)
    - [aiforge migrate](#aiforge-migrate)
  - [Module Commands](#module-commands)
    - [aiforge summarize](#aiforge-summarize)
    - [aiforge classify](#aiforge-classify)
    - [aiforge extract](#aiforge-extract)
    - [aiforge generate](#aiforge-generate)
    - [aiforge moderate](#aiforge-moderate)
  - [RAG Commands](#rag-commands)
    - [aiforge qa](#aiforge-qa)
    - [aiforge search](#aiforge-search)
  - [Agent Commands](#agent-commands)
    - [aiforge chat](#aiforge-chat)
  - [Discovery Commands](#discovery-commands)
    - [aiforge config](#aiforge-config)
    - [aiforge models](#aiforge-models)
    - [aiforge registry](#aiforge-registry)
  - [Common Options](#common-options)
  - [Configuration Priority](#configuration-priority)
  - [Environment Variables](#environment-variables)
  - [Exit Codes](#exit-codes)
  - [Examples](#examples)
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
| All other commands | *(none)* | — |

---

## Global Help

```
$ aiforge --help

 Usage: aiforge [OPTIONS] COMMAND [ARGS]...

 AIForge — plug-and-play AI toolkit

╭─ Commands ───────────────────────────────────────────────────────────────╮
│ init          Scaffold aiforge.yaml + .env.example                      │
│ serve         Start the FastAPI REST API server                         │
│ playground    Start the Streamlit interactive playground                │
│ migrate       Inspect or manage vector store collections                │
│ summarize     Summarise text                                            │
│ classify      Classify text into labels                                 │
│ extract       Extract structured data from text                         │
│ generate      Generate text from a prompt / template                    │
│ moderate      Content moderation & PII redaction                        │
│ qa            Document Q&A: ingest documents and ask questions           │
│ search        Semantic search: index documents and query                │
│ chat          Interactive AI chat agent                                  │
│ config        Show resolved configuration                               │
│ models        List models & estimate costs                              │
│ registry      Explore registered components                             │
╰──────────────────────────────────────────────────────────────────────────╯
```

---

## Infrastructure Commands

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

## Module Commands

Every SDK module is exposed as a top-level CLI command. Module commands share
a common set of options described in [Common Options](#common-options).

### aiforge summarize

Summarise text with configurable style and strategy.

```bash
aiforge summarize --text "long article…"
aiforge summarize --file article.txt --style bullets --strategy map_reduce
cat article.txt | aiforge summarize
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--text` | `-t` | `str` | — | Text to summarise |
| `--file` | `-f` | `path` | — | Read text from a file |
| `--style` | `-s` | `str` | `paragraph` | Output style: `paragraph` \| `bullets` \| `headline` |
| `--max-length` | | `int` | — | Approximate max output length (tokens) |
| `--strategy` | | `str` | — | Chunking strategy: `stuff` \| `map_reduce` \| `refine` |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

If `--text` and `--file` are both omitted, the command reads from **stdin**.

---

### aiforge classify

Classify text into one or more labels.

```bash
aiforge classify --text "I love this product" --labels "positive,negative,neutral"
aiforge classify -f review.txt -l "spam,ham" --multi-label
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--text` | `-t` | `str` | — | Text to classify |
| `--file` | `-f` | `path` | — | Read text from a file |
| `--labels` | `-l` | `str` | **(required)** | Comma-separated candidate labels |
| `--multi-label` | | `bool` | `False` | Allow multiple labels |
| `--examples-file` | | `path` | — | JSON file with few-shot examples |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

---

### aiforge extract

Extract structured data from text using a JSON schema.

```bash
aiforge extract --text "John Doe, age 32, NYC" --schema-file person.json
cat receipt.txt | aiforge extract --schema-file receipt_schema.json
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--text` | `-t` | `str` | — | Text to extract from |
| `--file` | `-f` | `path` | — | Read text from a file |
| `--schema-file` | | `path` | — | JSON Schema file defining the output structure |
| `--max-retries` | | `int` | `2` | Max validation retries |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

---

### aiforge generate

Generate text from a prompt or Jinja2 template.

```bash
aiforge generate --prompt "Write a haiku about coding"
aiforge generate -f template.txt --var topic=AI --var tone=formal
aiforge generate -p "List 5 ideas" --guardrails competitors --max-tokens 200
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--prompt` | `-p` | `str` | — | Prompt text |
| `--file` | `-f` | `path` | — | Read prompt from a file |
| `--var` | | `str` | — | Template variable as `key=value` (repeatable) |
| `--guardrails` | `-g` | `str` | — | Comma-separated guardrail names |
| `--max-tokens` | | `int` | — | Max output tokens |
| `--temperature` | | `float` | — | Sampling temperature (0.0–2.0) |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

---

### aiforge moderate

Content moderation and PII redaction.

```bash
aiforge moderate --text "Check this user post…"
aiforge moderate --file comment.txt --checks toxicity,bias
aiforge moderate --text "My SSN is 123-45-6789" --redact
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--text` | `-t` | `str` | — | Text to moderate |
| `--file` | `-f` | `path` | — | Read text from a file |
| `--checks` | `-c` | `str` | — | Comma-separated checks (e.g. `toxicity,bias,pii`) |
| `--pii-types` | | `str` | — | Comma-separated PII types to detect |
| `--redact` | `-r` | `bool` | `False` | Redact PII instead of just flagging |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

---

## RAG Commands

AIForge has two RAG-oriented modules. They both use vector stores and
embeddings, but serve different purposes:

| | `qa` | `search` |
|---|---|---|
| **Purpose** | Answer a question in natural language | Find the most similar documents |
| **LLM calls** | Yes — agentic pipeline generates an answer | None — embedding-only |
| **Ingestion** | Raw files/dirs/URLs (auto-chunks & embeds) | Pre-structured documents (id + text + metadata) |
| **Output** | Synthesized answer, sources, confidence | Ranked hits with similarity scores |
| **Cost** | Embedding + LLM tokens | Embedding tokens only |
| **Use when** | You need a *written answer* derived from docs | You need a *ranked list* of matching passages |

Think of `search` as a smarter `grep` and `qa` as a research assistant that
reads the results and writes you an answer.

### aiforge qa

Document Q&A using retrieval-augmented generation (RAG). Has two
subcommands: `ingest` (load documents) and `ask` (query them).

#### aiforge qa ingest

```bash
aiforge qa ingest ./docs/ --collection my-docs --extensions ".md,.txt"
aiforge qa ingest https://github.com/user/repo --collection repo-docs
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `SOURCE` | Yes | File path, directory, or git URL to ingest |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--collection` | `-c` | `str` | `default` | Target collection name |
| `--extensions` | `-e` | `str` | — | Comma-separated file extensions (e.g. `.py,.md`) |
| `--batch-size` | | `int` | `100` | Embedding batch size |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

#### aiforge qa ask

```bash
aiforge qa ask "How does authentication work?"
aiforge qa ask "What is the retry policy?" --collection my-docs --trace
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY` | Yes | The question to ask |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--collection` | `-c` | `str` | `default` | Collection to search |
| `--filter` | | `str` | — | Filter results by source path substring |
| `--trace` | | `bool` | `False` | Include agent reasoning trace |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

---

### aiforge search

Semantic search over indexed documents. Has two subcommands:
`index` (build index) and `query` (search).

#### aiforge search index

```bash
aiforge search index ./src/ --collection codebase --extensions ".py,.ts"
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `SOURCE` | Yes | File path or directory to index |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--collection` | `-c` | `str` | `default` | Target collection |
| `--extensions` | `-e` | `str` | — | Comma-separated file extensions |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` |

#### aiforge search query

```bash
aiforge search query "database connection pooling" --top-k 10
aiforge search query "error handling" --format table
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `QUERY` | Yes | Search query text |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--collection` | `-c` | `str` | `default` | Collection to search |
| `--top-k` | `-k` | `int` | `5` | Number of results (1–100) |
| `--output` | `-o` | `path` | — | Write result to file |
| `--format` | | `str` | `json` | Output format: `json` \| `yaml` \| `table` |

When `--format table` is used, results are displayed as a Rich table with
columns: **#**, **Score**, **ID**, **Text**.

---

## Agent Commands

### aiforge chat

Interactive chat with an AI agent. Supports single-turn and REPL modes.

```bash
# Single message
aiforge chat "What is the capital of France?"

# Interactive REPL
aiforge chat
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MESSAGE` | No | Single message (omit for interactive REPL) |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--system` | `-s` | `str` | — | System prompt |
| `--memory` | `-m` | `str` | `buffer` | Memory mode: `buffer` \| `summary` \| `none` |
| `--memory-k` | | `int` | `10` | Messages to keep in memory |

**Interactive REPL commands:**

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit the chat |
| `Ctrl+C` | Exit the chat |

---

## Discovery Commands

### aiforge config

Display the resolved AIForge configuration.

```bash
aiforge config
aiforge config --format yaml
aiforge config --reveal  # show secret values
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--format` | `str` | `json` | Output format: `json` \| `yaml` |
| `--reveal` | `bool` | `False` | Show secret values (keys, tokens); masked by default |

Secrets (any key containing `key`, `secret`, `token`, `password`, or
`credential`) are replaced with `***` unless `--reveal` is specified.

---

### aiforge models

List available models and estimate costs.

#### aiforge models list

```bash
aiforge models list
aiforge models list --quality fast --provider openai
aiforge models list --format json
```

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--quality` | `-q` | `str` | — | Filter by tier: `fast` \| `balanced` \| `quality` |
| `--provider` | `-p` | `str` | — | Filter by provider name |
| `--format` | | `str` | `table` | Output format: `json` \| `yaml` \| `table` |

Table output columns: **Model**, **Provider**, **Quality**, **Context**,
**Input $/1K**, **Output $/1K**.

#### aiforge models estimate

```bash
aiforge models estimate gpt-4o --input 2000 --output 1000
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `MODEL` | Yes | Model name (e.g. `gpt-4o-mini`) |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | `int` | `1000` | Estimated input tokens |
| `--output` | `-o` | `int` | `500` | Estimated output tokens |

---

### aiforge registry

Show all registered modules, providers, and vector stores.

```bash
aiforge registry
aiforge registry --format json
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--format` | `str` | `table` | Output format: `json` \| `yaml` \| `table` |

Table output shows three categories: **modules**, **providers**,
**vectorstores** and their registered items.

---

## Common Options

Most module commands share these patterns:

**Input:** Every module command accepts `--text/-t` for inline text,
`--file/-f` to read from a file, or **stdin** when both are omitted:

```bash
aiforge summarize --text "inline text"
aiforge summarize --file doc.txt
echo "piped text" | aiforge summarize
```

**Output:** Use `--output/-o` to write results to a file instead of stdout,
and `--format` to switch between JSON and YAML:

```bash
aiforge classify -t "hello" -l "en,fr" --format yaml
aiforge extract -f data.txt --schema-file s.json -o result.json
```

**Metrics footer:** After every module call, the CLI prints a one-line metrics
footer showing cost, token counts, and latency:

```
⏱  $0.000340 │ 150 in │ 42 out │ 1.23 s │ gpt-4o-mini
```

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

nano aiforge.yaml    # Set provider, model, etc.
nano .env            # Add API keys
```

### Start the API Server

```bash
aiforge serve
aiforge serve --reload --port 9000
AIFORGE_API_KEY=secret-token AIFORGE_LOG_FORMAT=json aiforge serve
```

### Launch the Playground

```bash
aiforge playground
aiforge playground --port 8080
```

### Run Module Commands

```bash
# Summarise a file
aiforge summarize --file article.txt --style bullets

# Classify piped input
echo "This is spam" | aiforge classify -l "spam,ham"

# Extract structured data
aiforge extract --text "Alice, 30, NYC" --schema-file person.json --format yaml

# Generate with template variables
aiforge generate -f email_template.txt --var name=Alice --var product=Widget

# Moderate and redact PII
aiforge moderate --text "Call me at 555-0123" --redact
```

### Document Q&A Workflow

```bash
# Ingest documentation
aiforge qa ingest ./docs/ --collection my-docs --extensions ".md,.txt"
# ✓ Indexed 247 chunks from 12 files into 'my-docs'.

# Ask questions
aiforge qa ask "How do I configure authentication?" --collection my-docs
```

### Semantic Search Workflow

```bash
# Index a codebase
aiforge search index ./src/ --collection codebase --extensions ".py,.ts"

# Search with a Rich table
aiforge search query "error handling patterns" --top-k 10 --format table
```

### Interactive Chat

```bash
# Single question
aiforge chat "Explain Python decorators"

# REPL session
aiforge chat --system "You are a helpful coding assistant"
# you > How do I sort a dict by value?
# ...
# you > /quit
```

### Explore Models & Config

```bash
# See resolved config (secrets masked)
aiforge config

# List all models in table form
aiforge models list

# Estimate cost
aiforge models estimate gpt-4o --input 5000 --output 2000

# See registered components
aiforge registry
```

### Manage Collections

```bash
aiforge migrate --all
aiforge migrate -c customer-docs -s
aiforge migrate --delete test-collection
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
| Module command hangs waiting for input | No `--text`/`--file` and stdin is a terminal | Provide `--text` or `--file`, or pipe input |
| Server requests fail with 401 | `AIFORGE_API_KEY` is set | Include `Authorization: Bearer <key>` header in requests |
| LLM calls fail | API keys not configured | Edit `.env` or set `AIFORGE_OPENAI_API_KEY` etc. |
| `config` shows `***` for keys | Secrets masked by default | Use `aiforge config --reveal` to view |
| `models list` returns empty table | No provider configured | Set `AIFORGE_PROVIDER` and API key |
| Port already in use | Another process on that port | Use `--port <other>` or stop the conflicting process |

---

## See Also

- [Quick Start](quickstart.md) — Getting started guide
- [Configuration](configuration.md) — Full configuration reference
- [API Server](api-server.md) — REST API endpoints and authentication
- [API Auth](api-auth.md) — API key authentication details
- [Playground](playground.md) — Interactive UI documentation
- [Providers](providers.md) — Provider setup and configuration
