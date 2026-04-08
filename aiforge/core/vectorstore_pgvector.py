"""pgvector (PostgreSQL) vector store adapter.

Each AIForge "collection" maps to a separate table in the configured
PostgreSQL database.  Uses ``psycopg`` (v3) with the ``pgvector``
extension for vector similarity search.

The adapter auto-creates the ``vector`` extension and collection table
on first use if they do not already exist.
"""

from __future__ import annotations

import structlog
from typing import Any

from aiforge.core.vectorstore import VectorStoreAdapter

logger = structlog.get_logger(__name__)

_UPSERT_BATCH = 500  # Postgres can handle larger batches than SaaS APIs


class PgVectorStore(VectorStoreAdapter):
    """PostgreSQL + pgvector backed vector store.

    Args:
        dsn:             PostgreSQL connection string
                         (e.g. ``postgresql://user:pass@host:5432/db``).
        collection_name: Table name for this collection.
        vector_size:     Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        dsn: str,
        collection_name: str = "default",
        vector_size: int = 1536,
    ) -> None:
        import psycopg

        self._dsn = dsn
        self._collection_name = self._sanitise_name(collection_name)
        self._vector_size = vector_size
        self._conn = psycopg.connect(dsn, autocommit=True)

        # Ensure pgvector extension and collection table exist
        self._ensure_schema()

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        import json
        from psycopg import sql as pgsql

        table = self._collection_name
        stmt = pgsql.SQL("""
            INSERT INTO {table} (id, text, metadata, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE
                SET text      = EXCLUDED.text,
                    metadata  = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
        """).format(table=pgsql.Identifier(table))

        with self._conn.cursor() as cur:
            for i in range(0, len(ids), _UPSERT_BATCH):
                batch = list(
                    zip(
                        ids[i : i + _UPSERT_BATCH],
                        texts[i : i + _UPSERT_BATCH],
                        [json.dumps(m) for m in metadatas[i : i + _UPSERT_BATCH]],
                        [self._vec_literal(e) for e in embeddings[i : i + _UPSERT_BATCH]],
                    )
                )
                cur.executemany(stmt, batch)

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        import json
        from psycopg import sql as pgsql

        table = self._collection_name
        vec = self._vec_literal(embedding)

        if include_embeddings:
            cols_sql = pgsql.SQL("id, text, metadata, embedding <=> %s::vector AS distance, embedding")
        else:
            cols_sql = pgsql.SQL("id, text, metadata, embedding <=> %s::vector AS distance")

        params: list[Any] = [vec]
        where_sql = pgsql.SQL("")
        if filter:
            where_sql = pgsql.SQL(" WHERE metadata @> %s::jsonb")
            params.append(json.dumps(filter))

        params.append(vec)  # ORDER BY
        params.append(top_k)
        stmt = pgsql.SQL(
            "SELECT {cols} FROM {table}{where} ORDER BY embedding <=> %s::vector LIMIT %s"
        ).format(cols=cols_sql, table=pgsql.Identifier(table), where=where_sql)

        with self._conn.cursor() as cur:
            cur.execute(stmt, params)
            rows = cur.fetchall()

        hits: list[dict[str, Any]] = []
        for row in rows:
            hit: dict[str, Any] = {
                "id": row[0],
                "text": row[1],
                "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                "distance": float(row[3]),
            }
            if include_embeddings:
                hit["embeddings"] = row[4]
            hits.append(hit)
        return hits

    def delete_collection(self, name: str) -> None:
        from psycopg import sql as pgsql

        safe = self._sanitise_name(name)
        with self._conn.cursor() as cur:
            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(pgsql.Identifier(safe)))

    def list_collections(self) -> list[str]:
        sql = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE 'aiforge_%%'
        """
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]

    def collection_stats(self, name: str) -> dict[str, Any]:
        from psycopg import sql as pgsql

        safe = self._sanitise_name(name)
        with self._conn.cursor() as cur:
            cur.execute(pgsql.SQL("SELECT COUNT(*) FROM {}").format(pgsql.Identifier(safe)))
            row = cur.fetchone()
            assert row is not None, f"COUNT(*) returned no rows for table {safe!r}"
            count = row[0]
        return {
            "name": name,
            "chunks": count,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the pgvector extension and collection table if needed."""
        from psycopg import sql as pgsql

        table = self._collection_name
        idx_name = f"{table}_embedding_idx"
        with self._conn.cursor() as cur:
            cur.execute(pgsql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
            cur.execute(pgsql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id        TEXT PRIMARY KEY,
                    text      TEXT NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding vector({size})
                )
            """).format(
                table=pgsql.Identifier(table),
                size=pgsql.Literal(self._vector_size),
            ))
            cur.execute(pgsql.SQL("""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """).format(
                idx=pgsql.Identifier(idx_name),
                table=pgsql.Identifier(table),
            ))

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    @staticmethod
    def _sanitise_name(name: str) -> str:
        """Ensure the collection name is safe for use as a SQL identifier.

        Prefixes with ``aiforge_`` so we can enumerate collections later and
        only strips non-alphanumeric characters.
        """
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        if not clean.startswith("aiforge_"):
            clean = f"aiforge_{clean}"
        return clean

    @staticmethod
    def _vec_literal(vec: list[float]) -> str:
        """Convert a Python list of floats to pgvector literal ``'[1,2,3]'``."""
        return "[" + ",".join(str(v) for v in vec) + "]"
