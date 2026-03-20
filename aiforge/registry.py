"""AIForge plugin registry — register and resolve modules, providers,
and vector store adapters by name."""

from __future__ import annotations

from typing import Any

from aiforge.core.types import BaseModule
from aiforge.core.vectorstore import VectorStoreAdapter

# ---------------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------------

_module_registry: dict[str, type[BaseModule]] = {}
_vectorstore_registry: dict[str, type[VectorStoreAdapter]] = {}


def _validate_name(name: str) -> str:
    """Normalise and validate a registry name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Registry name must be a non-empty string")
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


def register_module(name: str, cls: type[BaseModule]) -> None:
    """Register a module class under *name*.

    Example::

        from aiforge.registry import register_module

        class SentimentAnalyzer(BaseModule):
            def run(self, text, **kw): ...
            async def arun(self, text, **kw): ...

        register_module("sentiment", SentimentAnalyzer)
    """
    key = _validate_name(name)
    if not (isinstance(cls, type) and issubclass(cls, BaseModule)):
        raise TypeError(f"{cls!r} must be a subclass of BaseModule")
    _module_registry[key] = cls


def get_module(name: str) -> type[BaseModule]:
    """Look up a registered module class by name.

    Raises:
        KeyError: if *name* is not registered.
    """
    key = _validate_name(name)
    if key not in _module_registry:
        raise KeyError(
            f"Module {key!r} not registered. "
            f"Available: {sorted(_module_registry)}"
        )
    return _module_registry[key]


def list_modules() -> list[str]:
    """Return names of all registered modules."""
    return sorted(_module_registry)


# ---------------------------------------------------------------------------
# Vector store registration
# ---------------------------------------------------------------------------


def register_vectorstore(name: str, cls: type[VectorStoreAdapter]) -> None:
    """Register a vector store adapter under *name*.

    Example::

        from aiforge.registry import register_vectorstore

        class PgVectorAdapter(VectorStoreAdapter):
            ...

        register_vectorstore("pgvector", PgVectorAdapter)
    """
    key = _validate_name(name)
    if not (isinstance(cls, type) and issubclass(cls, VectorStoreAdapter)):
        raise TypeError(f"{cls!r} must be a subclass of VectorStoreAdapter")
    _vectorstore_registry[key] = cls


def get_vectorstore(name: str) -> type[VectorStoreAdapter]:
    """Look up a registered vector store adapter class by name.

    Raises:
        KeyError: if *name* is not registered.
    """
    key = _validate_name(name)
    if key not in _vectorstore_registry:
        raise KeyError(
            f"Vector store {key!r} not registered. "
            f"Available: {sorted(_vectorstore_registry)}"
        )
    return _vectorstore_registry[key]


def list_vectorstores() -> list[str]:
    """Return names of all registered vector store adapters."""
    return sorted(_vectorstore_registry)


# ---------------------------------------------------------------------------
# Provider registration (delegates to aiforge.providers)
# ---------------------------------------------------------------------------
# The provider subsystem in aiforge.providers has a richer API (lazy loading,
# settings-based resolution).  We re-export the registration and listing
# functions here so that registry.py is the unified plugin entry point.

from aiforge.providers import (  # noqa: E402
    get_provider,
    list_providers,
    register_provider,
)
