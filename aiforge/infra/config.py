"""
AIForgeSettings — Pydantic Settings v2 config model.

Layered priority (low → high):
  1. Built-in defaults
  2. aiforge.yaml  (if present)
  3. .env file     (if present)
  4. Environment variables  (AIFORGE_*)
  5. Constructor keyword arguments
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Supported combinations (provider → allowed embedding models)
# ---------------------------------------------------------------------------

_PROVIDER_EMBEDDING_MAP: dict[str, list[str]] = {
    "openai": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
    "anthropic": [
        "voyage-3",
        "voyage-3-lite",
        "voyage-large-2",
        "voyage-code-2",
    ],
    "ollama": [
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm",
    ],
    "azure": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
}


def _load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load YAML config file, returning empty dict if not present."""
    if yaml_path.exists():
        with yaml_path.open() as fh:
            data = yaml.safe_load(fh) or {}
        # Flatten top-level keys to AIFORGE_* env-var style
        flat: dict[str, Any] = {}
        for k, v in data.items():
            key = k.upper()
            if not key.startswith("AIFORGE_"):
                key = f"AIFORGE_{key}"
            flat[key] = v
        return flat
    return {}


class AIForgeSettings(BaseSettings):
    """Unified configuration model for AIForge.

    All environment variables are prefixed with ``AIFORGE_``.
    Constructor kwargs take highest priority (override everything).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        secrets_dir=None,
    )

    # ------------------------------------------------------------------
    # Provider & model
    # ------------------------------------------------------------------
    PROVIDER: str = "openai"
    MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    VOYAGE_API_KEY: str | None = None
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    VECTOR_STORE: str = "chroma"  # "chroma" | "pgvector" | registered name
    CHROMA_PERSIST_DIR: str = "./aiforge_data"

    # pgvector
    PGVECTOR_DSN: str | None = None  # e.g. postgresql://user:pass@host:5432/db
    PGVECTOR_COLLECTION: str = "default"

    # Pinecone
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX: str | None = None
    PINECONE_ENVIRONMENT: str | None = None

    # Qdrant
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str | None = None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    RATE_LIMIT_RPM: int = 500
    RATE_LIMIT_TPM: int = 150_000

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------
    COST_TRACKING_ENABLED: bool = True
    COST_WARN_THRESHOLD: float = 1.0  # USD per single call

    # ------------------------------------------------------------------
    # Tracing / observability
    # ------------------------------------------------------------------
    TRACING_ENABLED: bool = False
    TRACING_BACKEND: str = "langsmith"  # "langsmith" | "otel" | "console"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "aiforge"
    OTEL_EXPORTER_ENDPOINT: str | None = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" | "console"

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------
    API_KEY: str | None = None  # Bearer token for REST API auth
    API_HOST: str = "0.0.0.0"  # noqa: S104
    API_PORT: int = 8000

    # ------------------------------------------------------------------
    # Playground
    # ------------------------------------------------------------------
    PLAYGROUND_PORT: int = 8501

    # ------------------------------------------------------------------
    # Custom prompt overrides (from aiforge.yaml custom_prompts: section)
    # ------------------------------------------------------------------
    CUSTOM_PROMPTS: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Model router
    # ------------------------------------------------------------------
    MODEL_ROUTER_ENABLED: bool = True
    DEFAULT_QUALITY: str = "medium"  # "high" | "medium" | "low"
    PREFER_LOCAL: bool = False
    CUSTOM_MODELS: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openai", "anthropic", "ollama", "azure"}
        if v not in allowed:
            raise ValueError(f"AIFORGE_PROVIDER must be one of {allowed}, got {v!r}")
        return v

    @field_validator("DEFAULT_QUALITY")
    @classmethod
    def validate_default_quality(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v not in allowed:
            raise ValueError(
                f"AIFORGE_DEFAULT_QUALITY must be one of {allowed}, got {v!r}"
            )
        return v

    @field_validator("TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("AIFORGE_TEMPERATURE must be between 0.0 and 2.0")
        return v

    @model_validator(mode="after")
    def validate_provider_embedding_combo(self) -> "AIForgeSettings":
        """Ensure the embedding model is compatible with the chosen provider."""
        allowed = _PROVIDER_EMBEDDING_MAP.get(self.PROVIDER, [])
        if allowed and self.EMBEDDING_MODEL not in allowed:
            # Must raise ValueError here (not ConfigError) because Pydantic wraps
            # model_validator exceptions in pydantic.ValidationError.  Callers
            # wanting to catch this should catch pydantic.ValidationError.
            raise ValueError(
                f"Embedding model {self.EMBEDDING_MODEL!r} is not compatible with "
                f"provider {self.PROVIDER!r}. "
                f"Allowed embedding models: {allowed}"
            )
        return self

    @model_validator(mode="after")
    def validate_api_keys_present(self) -> "AIForgeSettings":
        """Warn (not error) if a required API key looks missing."""
        key_map = {
            "openai": ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            "anthropic": ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY),
            "azure": ("AZURE_OPENAI_API_KEY", self.AZURE_OPENAI_API_KEY),
        }
        if self.PROVIDER in key_map:
            env_name, value = key_map[self.PROVIDER]
            if not value:
                import warnings

                warnings.warn(
                    f"AIFORGE_{env_name} is not set but provider is {self.PROVIDER!r}. "
                    "LLM calls will fail until the key is provided.",
                    stacklevel=2,
                )
        return self

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "AIForgeSettings":
        """Load settings from a YAML file with optional code-level overrides."""
        yaml_data = _load_yaml_config(Path(path))
        # Set env vars temporarily so Pydantic Settings picks them up
        old_env: dict[str, str | None] = {}
        for k, v in yaml_data.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        try:
            instance = cls(**overrides)
        finally:
            for k, old in old_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
        return instance


@lru_cache(maxsize=1)
def get_settings() -> AIForgeSettings:
    """Return the cached global settings singleton.

    The first call reads YAML + env vars. Module constructors that accept
    kwargs create a local patched copy via ``AIForgeSettings(**kwargs)``.
    """
    yaml_path = Path("aiforge.yaml")
    if yaml_path.exists():
        return AIForgeSettings.from_yaml(yaml_path)
    return AIForgeSettings()
