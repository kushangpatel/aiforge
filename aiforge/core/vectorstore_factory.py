"""Config-driven vector store factory.

Resolves the configured ``VECTOR_STORE`` setting to a concrete
:class:`~aiforge.core.vectorstore.VectorStoreAdapter` instance by looking
up the registered adapter class in the plugin registry.

Which backend to use (``chroma``, ``pgvector``, …) comes from
``aiforge.yaml`` / ``AIFORGE_VECTOR_STORE``.  Credentials and connection
details come from ``.env`` / env vars (never from YAML).
"""

from __future__ import annotations

from typing import Any

from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.infra.config import AIForgeSettings, get_settings


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
    """
    from aiforge.registry import get_vectorstore

    cfg = settings or get_settings()
    store_name = cfg.VECTOR_STORE.strip().lower()

    adapter_cls = get_vectorstore(store_name)  # KeyError if unknown

    # Build kwargs based on the backend
    kwargs: dict[str, Any] = {}

    if store_name == "chroma":
        kwargs["persist_dir"] = persist_dir or cfg.CHROMA_PERSIST_DIR
        kwargs["collection_name"] = collection_name

    elif store_name == "pgvector":
        if cfg.PGVECTOR_DSN:
            kwargs["dsn"] = cfg.PGVECTOR_DSN
        kwargs["collection_name"] = collection_name

    elif store_name == "pinecone":
        if cfg.PINECONE_API_KEY:
            kwargs["api_key"] = cfg.PINECONE_API_KEY
        if cfg.PINECONE_INDEX:
            kwargs["index_name"] = cfg.PINECONE_INDEX
        if cfg.PINECONE_ENVIRONMENT:
            kwargs["environment"] = cfg.PINECONE_ENVIRONMENT

    elif store_name == "qdrant":
        if cfg.QDRANT_URL:
            kwargs["url"] = cfg.QDRANT_URL
        if cfg.QDRANT_API_KEY:
            kwargs["api_key"] = cfg.QDRANT_API_KEY
        kwargs["collection_name"] = cfg.QDRANT_COLLECTION or collection_name

    # For unknown/custom adapters, pass collection_name + overrides and let
    # the adapter's __init__ accept what it needs.
    if store_name not in {"chroma", "pgvector", "pinecone", "qdrant"}:
        kwargs["collection_name"] = collection_name

    kwargs.update(overrides)

    return adapter_cls(**kwargs)
