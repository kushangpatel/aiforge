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

import contextlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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

# ---------------------------------------------------------------------------
# YAML settings source
# ---------------------------------------------------------------------------

#: Path checked automatically on every settings load.
_DEFAULT_YAML_PATH = Path("aiforge.yaml")

#: Temporarily overridden by AIForgeSettings.from_yaml() for one instantiation.
_YAML_PATH_OVERRIDE: Path | None = None


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Pydantic-settings v2 source that loads from an ``aiforge.yaml`` file.

    Keys in the YAML file are expected to match the field names (without the
    ``AIFORGE_`` prefix, e.g. ``model: gpt-4o``).  They are uppercased and
    matched against the settings fields directly.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_path: Path = _DEFAULT_YAML_PATH,
    ) -> None:
        super().__init__(settings_cls)
        self._yaml_file = yaml_path
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        """Read and normalise the YAML file, returning an empty dict if absent."""
        if not self._yaml_file.exists():
            return {}
        with self._yaml_file.open() as fh:
            raw = yaml.safe_load(fh) or {}
        # Normalise keys: strip AIFORGE_ prefix if present, then uppercase.
        normalised: dict[str, Any] = {}
        for k, v in raw.items():
            key = k.upper()
            if key.startswith("AIFORGE_"):
                key = key[len("AIFORGE_") :]
            normalised[key] = v
        return normalised

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # field_name is already the canonical name (without prefix).
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_name, field_info in self.settings_cls.model_fields.items():
            value, _, found = self.get_field_value(field_info, field_name)
            if found:
                with contextlib.suppress(Exception):
                    value = self.prepare_field_value(field_name, field_info, value, value)
                result[field_name] = value
        return result


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


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
    PINECONE_ENVIRONMENT: str | None = (
        None  # Deprecated — unused by factory (v3 SDK resolves endpoint automatically)
    )

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
    # Request cache
    # ------------------------------------------------------------------
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # seconds
    CACHE_BACKEND: str = "memory"  # "memory" | "redis"
    CACHE_MAX_ENTRIES: int = 1000
    CACHE_REDIS_URL: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------
    WEBHOOK_ENABLED: bool = True
    WEBHOOK_TIMEOUT: float = 10.0  # seconds
    WEBHOOK_ALLOW_INSECURE: bool = False  # allow http:// URLs (dev only)

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    CORS_ORIGINS: str = "*"  # Comma-separated origins, or "*" for open access

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
    # Custom pricing overrides
    # ------------------------------------------------------------------
    CUSTOM_PRICING: dict[str, dict[str, float]] = {}
    CUSTOM_CONTEXT_WINDOWS: dict[str, int] = {}

    # ------------------------------------------------------------------
    # MCP (Model Context Protocol)
    # ------------------------------------------------------------------
    MCP_SERVERS: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        # Built-in providers are always accepted.
        builtin = {"openai", "anthropic", "ollama", "azure"}
        if v in builtin:
            return v
        # For custom providers, check the runtime registry.
        # Import lazily to avoid circular deps (config ← providers).
        try:
            from aiforge.providers import _REGISTRY, _ensure_registered

            _ensure_registered()
            if v in _REGISTRY:
                return v
        except ImportError:
            pass
        raise ValueError(
            f"AIFORGE_PROVIDER must be a built-in ({sorted(builtin)}) "
            f"or registered via register_provider(). Got {v!r}"
        )

    @field_validator("VECTOR_STORE")
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("AIFORGE_VECTOR_STORE must not be empty")
        return v

    @field_validator("CACHE_BACKEND")
    @classmethod
    def validate_cache_backend(cls, v: str) -> str:
        allowed = {"memory", "redis"}
        if v not in allowed:
            raise ValueError(f"AIFORGE_CACHE_BACKEND must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("DEFAULT_QUALITY")
    @classmethod
    def validate_default_quality(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v not in allowed:
            raise ValueError(f"AIFORGE_DEFAULT_QUALITY must be one of {allowed}, got {v!r}")
        return v

    @field_validator("TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("AIFORGE_TEMPERATURE must be between 0.0 and 2.0")
        return v

    @model_validator(mode="after")
    def validate_provider_embedding_combo(self) -> AIForgeSettings:
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
    def validate_api_keys_present(self) -> AIForgeSettings:
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
    # ------------------------------------------------------------------
    # Custom settings sources
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority order (highest → lowest): init > env > .env > yaml > defaults."""
        yaml_path = _YAML_PATH_OVERRIDE if _YAML_PATH_OVERRIDE is not None else _DEFAULT_YAML_PATH
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_path=yaml_path),
            file_secret_settings,
        )

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> AIForgeSettings:
        """Load settings from a specific YAML file with optional code-level overrides."""
        global _YAML_PATH_OVERRIDE  # noqa: PLW0603
        _YAML_PATH_OVERRIDE = Path(path)
        try:
            return cls(**overrides)
        finally:
            _YAML_PATH_OVERRIDE = None


@lru_cache(maxsize=1)
def get_settings() -> AIForgeSettings:
    """Return the cached global settings singleton.

    The first call reads YAML + env vars. Module constructors that accept
    kwargs create a local patched copy via ``AIForgeSettings(**kwargs)``.
    """
    settings = AIForgeSettings()

    apply_runtime_overrides(settings)

    return settings


def apply_runtime_overrides(settings: AIForgeSettings) -> None:
    """Apply pricing/context runtime tables from a settings object."""
    from aiforge.core.pricing import (
        apply_context_window_overrides,
        apply_pricing_overrides,
    )

    if settings.CUSTOM_PRICING:
        apply_pricing_overrides(settings.CUSTOM_PRICING)
    if settings.CUSTOM_CONTEXT_WINDOWS:
        apply_context_window_overrides(settings.CUSTOM_CONTEXT_WINDOWS)
