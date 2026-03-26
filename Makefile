.PHONY: help install install-dev install-all lint format typecheck test test-cov \
       serve playground docker-up docker-down docker-build docs docs-serve clean

PYTHON   ?= python3
VENV     ?= venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
RUFF     := $(VENV)/bin/ruff
MYPY     := $(VENV)/bin/mypy
AIFORGE  := $(VENV)/bin/aiforge

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv ## Install core dependencies
	$(PIP) install -e "."

install-dev: venv ## Install with dev extras
	$(PIP) install -e ".[dev]"

install-all: venv ## Install all extras
	$(PIP) install -e ".[all]"

# ── Quality ──────────────────────────────────────────────
lint: ## Run ruff linter
	$(RUFF) check aiforge/ tests/

format: ## Auto-format code with ruff
	$(RUFF) check --fix aiforge/ tests/
	$(RUFF) format aiforge/ tests/

typecheck: ## Run mypy type checks
	$(MYPY) aiforge/

# ── Testing ──────────────────────────────────────────────
test: ## Run tests
	$(PYTEST)

test-cov: ## Run tests with coverage report
	$(PYTEST) --cov=aiforge --cov-report=term-missing --cov-report=html

# ── Run ──────────────────────────────────────────────────
serve: ## Start the API server (dev mode)
	$(AIFORGE) serve --reload

playground: ## Start the Streamlit playground
	$(AIFORGE) playground

# ── Docker ───────────────────────────────────────────────
docker-build: ## Build Docker image
	docker compose build

docker-up: ## Start all services (API + Playground)
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Tail service logs
	docker compose logs -f

# ── Documentation ────────────────────────────────────────
docs: ## Build documentation site
	$(VENV)/bin/mkdocs build

docs-serve: ## Serve docs locally with live-reload
	$(VENV)/bin/mkdocs serve

# ── Cleanup ──────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	rm -rf site/
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
