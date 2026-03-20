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
    if not (isinstance(cls, type) and issubclass(cls, BaseModule)):
        raise TypeError(f"{cls!r} must be a subclass of BaseModule")
    _module_registry[name] = cls


def get_module(name: str) -> type[BaseModule]:
    """Look up a registered module class by name.

    Raises:
        KeyError: if *name* is not registered.
    """
    if name not in _module_registry:
        raise KeyError(
            f"Module {name!r} not registered. "
            f"Available: {sorted(_module_registry)}"
        )
    return _module_registry[name]


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
    if not (isinstance(cls, type) and issubclass(cls, VectorStoreAdapter)):
        raise TypeError(f"{cls!r} must be a subclass of VectorStoreAdapter")
    _vectorstore_registry[name] = cls


def get_vectorstore(name: str) -> type[VectorStoreAdapter]:
    """Look up a registered vector store adapter class by name.

    Raises:
        KeyError: if *name* is not registered.
    """
    if name not in _vectorstore_registry:
        raise KeyError(
            f"Vector store {name!r} not registered. "
            f"Available: {sorted(_vectorstore_registry)}"
        )
    return _vectorstore_registry[name]


def list_vectorstores() -> list[str]:
    """Return names of all registered vector store adapters."""
    return sorted(_vectorstore_registry)
