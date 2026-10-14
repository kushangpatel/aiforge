"""Unit tests for config-driven vector store factory."""

from __future__ import annotations

from typing import Any, cast

import pytest

from aiforge.core.types import ConfigError
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.core.vectorstore_factory import build_vectorstore
from aiforge.infra.config import AIForgeSettings
from aiforge.registry import register_vectorstore

# ---------------------------------------------------------------------------
# Helpers — minimal concrete stub
# ---------------------------------------------------------------------------


class _StubStore(VectorStoreAdapter):
    """Minimal concrete vector store that records __init__ kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def add_documents(self, texts, embeddings, metadatas, ids):
        pass

    def query(self, embedding, top_k=5, filter=None, include_embeddings=False):
        return []

    def delete_collection(self, name):
        pass

    def delete_by_source(self, source):
        return 0

    def list_all_metadata(self):
        return []

    def list_collections(self):
        return []

    def collection_stats(self, name):
        return {}


class _PgStub(_StubStore):
    """pgvector-like stub."""

    def __init__(
        self, *, dsn: str | None = None, collection_name: str = "default", **kw: Any
    ) -> None:
        super().__init__(dsn=dsn, collection_name=collection_name, **kw)
        self.dsn = dsn
        self.collection_name = collection_name


class _PineconeStub(_StubStore):
    """Pinecone-like stub."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str = "default",
        **kw: Any,
    ) -> None:
        super().__init__(api_key=api_key, index_name=index_name, namespace=namespace, **kw)
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace


class _QdrantStub(_StubStore):
    """Qdrant-like stub."""

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "default",
        **kw: Any,
    ) -> None:
        super().__init__(url=url, api_key=api_key, collection_name=collection_name, **kw)
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildVectorstoreChromaDefault:
    """Default behaviour: VECTOR_STORE=chroma → ChromaVectorStore."""

    def test_default_returns_chroma(self, tmp_path: Any) -> None:
        """With default config, build_vectorstore returns a ChromaVectorStore."""
        from aiforge.core.vectorstore import ChromaVectorStore

        settings = AIForgeSettings(
            VECTOR_STORE="chroma",
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma_data"),
        )
        store = build_vectorstore(settings=settings)
        assert isinstance(store, ChromaVectorStore)

    def test_chroma_uses_persist_dir_from_settings(self, tmp_path: Any) -> None:
        from aiforge.core.vectorstore import ChromaVectorStore

        custom_dir = str(tmp_path / "custom_chroma")
        settings = AIForgeSettings(
            VECTOR_STORE="chroma",
            CHROMA_PERSIST_DIR=custom_dir,
        )
        store = build_vectorstore(settings=settings)
        assert isinstance(store, ChromaVectorStore)
        assert store._persist_dir == custom_dir

    def test_chroma_persist_dir_override(self, tmp_path: Any) -> None:
        """persist_dir kwarg overrides settings.CHROMA_PERSIST_DIR."""
        from aiforge.core.vectorstore import ChromaVectorStore

        override_dir = str(tmp_path / "override")
        settings = AIForgeSettings(
            VECTOR_STORE="chroma",
            CHROMA_PERSIST_DIR=str(tmp_path / "ignored"),
        )
        store = build_vectorstore(persist_dir=override_dir, settings=settings)
        assert isinstance(store, ChromaVectorStore)
        assert store._persist_dir == override_dir

    def test_chroma_collection_name(self, tmp_path: Any) -> None:
        from aiforge.core.vectorstore import ChromaVectorStore

        settings = AIForgeSettings(
            VECTOR_STORE="chroma",
            CHROMA_PERSIST_DIR=str(tmp_path / "data"),
        )
        store = build_vectorstore(collection_name="my_docs", settings=settings)
        assert isinstance(store, ChromaVectorStore)
        assert store._collection_name == "my_docs"


class TestBuildVectorstorePgvector:
    """VECTOR_STORE=pgvector routes to registered pgvector adapter."""

    def test_pgvector_passes_dsn(self) -> None:
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(
            VECTOR_STORE="pgvector",
            PGVECTOR_DSN="postgresql://u:p@localhost/db",
        )
        store = build_vectorstore(settings=settings)
        assert isinstance(store, _PgStub)
        assert store.dsn == "postgresql://u:p@localhost/db"
        assert store.collection_name == "default"

    def test_pgvector_collection_name(self) -> None:
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(
            VECTOR_STORE="pgvector",
            PGVECTOR_DSN="postgresql://u:p@localhost/db",
        )
        store = build_vectorstore(collection_name="docs", settings=settings)
        assert cast(Any, store).collection_name == "docs"

    def test_pgvector_no_dsn_raises(self) -> None:
        """When DSN not set, ConfigError is raised early."""
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(VECTOR_STORE="pgvector")
        with pytest.raises(ConfigError, match="AIFORGE_PGVECTOR_DSN"):
            build_vectorstore(settings=settings)

    def test_pgvector_uses_config_collection(self) -> None:
        """PGVECTOR_COLLECTION from settings is used as the collection name."""
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(
            VECTOR_STORE="pgvector",
            PGVECTOR_DSN="postgresql://u:p@localhost/db",
            PGVECTOR_COLLECTION="my_table",
        )
        store = build_vectorstore(settings=settings)
        assert cast(Any, store).collection_name == "my_table"

    def test_pgvector_config_collection_overridden_by_arg(self) -> None:
        """collection_name arg takes precedence when PGVECTOR_COLLECTION is empty."""
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(
            VECTOR_STORE="pgvector",
            PGVECTOR_DSN="postgresql://u:p@localhost/db",
            PGVECTOR_COLLECTION="",
        )
        store = build_vectorstore(collection_name="from_arg", settings=settings)
        assert cast(Any, store).collection_name == "from_arg"


class TestBuildVectorstorePinecone:
    """VECTOR_STORE=pinecone routes to registered Pinecone adapter."""

    def test_pinecone_passes_credentials(self) -> None:
        register_vectorstore("pinecone", _PineconeStub)
        settings = AIForgeSettings(
            VECTOR_STORE="pinecone",
            PINECONE_API_KEY="pk-test",
            PINECONE_INDEX="my-idx",
            PINECONE_ENVIRONMENT="us-east-1",
        )
        store = build_vectorstore(settings=settings)
        assert isinstance(store, _PineconeStub)
        assert store.api_key == "pk-test"
        assert store.index_name == "my-idx"
        assert store.namespace == "default"


class TestBuildVectorstoreQdrant:
    """VECTOR_STORE=qdrant routes to registered Qdrant adapter."""

    def test_qdrant_passes_credentials(self) -> None:
        register_vectorstore("qdrant", _QdrantStub)
        settings = AIForgeSettings(
            VECTOR_STORE="qdrant",
            QDRANT_URL="http://localhost:6333",
            QDRANT_API_KEY="qd-key",
            QDRANT_COLLECTION="my_col",
        )
        store = build_vectorstore(settings=settings)
        assert isinstance(store, _QdrantStub)
        assert store.url == "http://localhost:6333"
        assert store.api_key == "qd-key"
        assert store.collection_name == "my_col"

    def test_qdrant_falls_back_to_arg_collection(self) -> None:
        register_vectorstore("qdrant", _QdrantStub)
        settings = AIForgeSettings(
            VECTOR_STORE="qdrant",
            QDRANT_URL="http://localhost:6333",
        )
        store = build_vectorstore(collection_name="fallback", settings=settings)
        assert cast(Any, store).collection_name == "fallback"

    def test_qdrant_no_url_raises(self) -> None:
        """ConfigError raised when QDRANT_URL is missing."""
        register_vectorstore("qdrant", _QdrantStub)
        settings = AIForgeSettings(VECTOR_STORE="qdrant")
        with pytest.raises(ConfigError, match="AIFORGE_QDRANT_URL"):
            build_vectorstore(settings=settings)


class TestBuildVectorstoreCustom:
    """A fully custom adapter registered at runtime."""

    def test_custom_adapter(self) -> None:
        register_vectorstore("myvec", _StubStore)
        settings = AIForgeSettings(VECTOR_STORE="myvec")
        store = build_vectorstore(collection_name="c1", settings=settings)
        assert isinstance(store, _StubStore)
        assert cast(_StubStore, store).init_kwargs["collection_name"] == "c1"

    def test_overrides_forwarded(self) -> None:
        register_vectorstore("myvec", _StubStore)
        settings = AIForgeSettings(VECTOR_STORE="myvec")
        store = build_vectorstore(collection_name="c1", settings=settings, timeout=30)
        assert cast(_StubStore, store).init_kwargs["timeout"] == 30


class TestBuildVectorstoreUnregistered:
    """Requesting an unregistered adapter raises KeyError."""

    def test_unregistered_raises(self) -> None:
        settings = AIForgeSettings(VECTOR_STORE="doesnotexist")
        with pytest.raises(KeyError, match="doesnotexist"):
            build_vectorstore(settings=settings)


class TestBuildVectorstoreFromGlobalSettings:
    """build_vectorstore falls back to get_settings() when no settings= passed."""

    def test_uses_global_settings(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.infra.config import get_settings

        monkeypatch.setenv("AIFORGE_VECTOR_STORE", "chroma")
        monkeypatch.setenv("AIFORGE_CHROMA_PERSIST_DIR", str(tmp_path / "global"))
        get_settings.cache_clear()

        from aiforge.core.vectorstore import ChromaVectorStore

        store = build_vectorstore()
        assert isinstance(store, ChromaVectorStore)


class TestConfigField:
    """VECTOR_STORE config field wired correctly."""

    def test_default_is_chroma(self) -> None:
        settings = AIForgeSettings()
        assert settings.VECTOR_STORE == "chroma"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.infra.config import get_settings

        monkeypatch.setenv("AIFORGE_VECTOR_STORE", "pgvector")
        # Ensure no yaml file interferes (yaml sets vector_store: chroma)
        monkeypatch.delenv("AIFORGE_VECTOR_STORE", raising=False)
        monkeypatch.setenv("AIFORGE_VECTOR_STORE", "pgvector")
        get_settings.cache_clear()
        # Direct constructor bypasses yaml → reads env directly
        settings = AIForgeSettings()
        assert settings.VECTOR_STORE == "pgvector"

    def test_pgvector_dsn_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiforge.infra.config import get_settings

        monkeypatch.setenv("AIFORGE_PGVECTOR_DSN", "postgresql://u:p@h/d")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.PGVECTOR_DSN == "postgresql://u:p@h/d"


class TestPublicExport:
    """build_vectorstore is importable from top-level aiforge package."""

    def test_importable(self) -> None:
        from aiforge import build_vectorstore as bvs

        assert callable(bvs)


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------


class TestCredentialValidation:
    """Required credentials raise ConfigError early."""

    def test_pgvector_missing_dsn(self) -> None:
        register_vectorstore("pgvector", _PgStub)
        settings = AIForgeSettings(VECTOR_STORE="pgvector")
        with pytest.raises(ConfigError, match="AIFORGE_PGVECTOR_DSN"):
            build_vectorstore(settings=settings)

    def test_pinecone_missing_api_key(self) -> None:
        register_vectorstore("pinecone", _PineconeStub)
        settings = AIForgeSettings(VECTOR_STORE="pinecone")
        with pytest.raises(ConfigError, match="AIFORGE_PINECONE_API_KEY"):
            build_vectorstore(settings=settings)

    def test_qdrant_missing_url(self) -> None:
        register_vectorstore("qdrant", _QdrantStub)
        settings = AIForgeSettings(VECTOR_STORE="qdrant")
        with pytest.raises(ConfigError, match="AIFORGE_QDRANT_URL"):
            build_vectorstore(settings=settings)

    def test_chroma_no_credential_check(self, tmp_path: Any) -> None:
        """Chroma needs no credentials — no ConfigError raised."""
        settings = AIForgeSettings(
            VECTOR_STORE="chroma",
            CHROMA_PERSIST_DIR=str(tmp_path / "data"),
        )
        store = build_vectorstore(settings=settings)
        assert store is not None

    def test_custom_no_credential_check(self) -> None:
        """Custom adapters have no credential requirements in the factory."""
        register_vectorstore("myvec", _StubStore)
        settings = AIForgeSettings(VECTOR_STORE="myvec")
        store = build_vectorstore(settings=settings)
        assert store is not None


# ---------------------------------------------------------------------------
# VECTOR_STORE validator
# ---------------------------------------------------------------------------


class TestVectorStoreValidator:
    """VECTOR_STORE field validator normalises and rejects empty strings."""

    def test_normalises_to_lowercase(self) -> None:
        settings = AIForgeSettings(VECTOR_STORE="Chroma")
        assert settings.VECTOR_STORE == "chroma"

    def test_strips_whitespace(self) -> None:
        settings = AIForgeSettings(VECTOR_STORE="  pgvector  ")
        assert settings.VECTOR_STORE == "pgvector"

    def test_rejects_empty(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="must not be empty"):
            AIForgeSettings(VECTOR_STORE="")

    def test_rejects_whitespace_only(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="must not be empty"):
            AIForgeSettings(VECTOR_STORE="   ")
