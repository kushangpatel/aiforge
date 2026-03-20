"""Config-driven vector store factory.

Resolves the configured ``VECTOR_STORE`` setting to a concrete
:class:`~aiforge.core.vectorstore.VectorStoreAdapter` instance by looking
up the registered adapter class in the plugin registry.

Which backend to use (``chroma``, ``pgvector``, …) comes from
``aiforge.yaml`` / ``AIFORGE_VECTOR_STORE``.  Credentials and connection
details come from ``.env`` / env vars (never from YAML).
"""

from __future__ import annotations

import structlog
from typing import Any

from aiforge.core.types import ConfigError
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.infra.config import AIForgeSettings, get_settings

logger = structlog.get_logger(__name__)

# Credentials that should be set for each backend (warn if missing)
_REQUIRED_CREDENTIALS: dict[str, list[tuple[str, str]]] = {
    "pgvector": [("PGVECTOR_DSN", "AIFORGE_PGVECTOR_DSN")],
    "pinecone": [("PINECONE_API_KEY", "AIFORGE_PINECONE_API_KEY")],
    "qdrant": [("QDRANT_URL", "AIFORGE_QDRANT_URL")],
}


def _check_credentials(store_name: str, cfg: AIForgeSettings) -> None:
    """Raise :class:`ConfigError` if required credentials are missing."""
    for attr, env_name in _REQUIRED_CREDENTIALS.get(store_name, []):
        if not getattr(cfg, attr, None):
            raise ConfigError(
                f"{env_name} must be set when AIFORGE_VECTOR_STORE={store_name!r}"
            )


def build_vectorstore(
    *,
    collection_name: str = "default",
    persist_dir: str | None = None,
    settings: AIForgeSettings | None = None,
    **overrides: Any,
) -> VectorStoreAdapter:
    """Instantiate the vector store configured in settings.

    Args:
        collection_name: Logical collection / table name.
        persist_dir: Override for ChromaDB's persist directory.  Ignored by
            backends that don't use it.
        settings: Optional pre-built settings instance.  Falls back to the
            global singleton.
        **overrides: Extra kwargs forwarded to the adapter constructor
            (e.g. custom timeouts, pool sizes).

    Returns:
        A ready-to-use :class:`VectorStoreAdapter` instance.

    Raises:
        KeyError: If the configured vector store name is not registered.
        ConfigError: If required credentials are missing for the backend.
    """
    from aiforge.registry import get_vectorstore

    cfg = settings or get_settings()
    store_name = cfg.VECTOR_STORE.strip().lower()

    logger.debug(
        "vectorstore.resolve",
        store=store_name,
        collection=collection_name,
    )

    adapter_cls = get_vectorstore(store_name)  # KeyError if unknown
    _check_credentials(store_name, cfg)

    # Build kwargs based on the backend
    kwargs: dict[str, Any] = {}

    if store_name == "chroma":
        kwargs["persist_dir"] = persist_dir or cfg.CHROMA_PERSIST_DIR
        kwargs["collection_name"] = collection_name

    elif store_name == "pgvector":
        kwargs["dsn"] = cfg.PGVECTOR_DSN
        # Prefer the function argument; fall back to config if arg is default
        if collection_name != "default":
            kwargs["collection_name"] = collection_name
        else:
            kwargs["collection_name"] = cfg.PGVECTOR_COLLECTION or collection_name

    elif store_name == "pinecone":
        kwargs["api_key"] = cfg.PINECONE_API_KEY
        if cfg.PINECONE_INDEX:
            kwargs["index_name"] = cfg.PINECONE_INDEX
        if cfg.PINECONE_ENVIRONMENT:
            kwargs["environment"] = cfg.PINECONE_ENVIRONMENT

    elif store_name == "qdrant":
        kwargs["url"] = cfg.QDRANT_URL
        if cfg.QDRANT_API_KEY:
            kwargs["api_key"] = cfg.QDRANT_API_KEY
        kwargs["collection_name"] = cfg.QDRANT_COLLECTION or collection_name

    # For unknown/custom adapters, pass collection_name + overrides and let
    # the adapter's __init__ accept what it needs.
    if store_name not in {"chroma", "pgvector", "pinecone", "qdrant"}:
        kwargs["collection_name"] = collection_name

    kwargs.update(overrides)

    logger.debug(
        "vectorstore.created",
        store=store_name,
        adapter=adapter_cls.__name__,
        collection=kwargs.get("collection_name", collection_name),
    )

    return adapter_cls(**kwargs)
