"""
Hybrid retrieval: an in-memory BM25 keyword index, Reciprocal Rank Fusion,
and a :class:`Retriever` that combines them with a vector store.

Retrieval modes
---------------
``"vector"``   Dense similarity search only (the default, unchanged behaviour).
``"keyword"``  Okapi BM25 over the chunk texts only; no query embedding.
``"hybrid"``   Both, fused with Reciprocal Rank Fusion (RRF).

Persistence
-----------
The BM25 index is **not** written to disk.  It is rebuilt in memory from the
vector store's own documents (:meth:`VectorStoreAdapter.list_all_documents`)
the first time a keyword or hybrid query runs, and rebuilt again whenever the
store's chunk count changes or the owning module writes to the store.  The
vector store therefore stays the single source of truth, so the two indexes
cannot drift apart across restarts.  Stores that cannot list their documents
fall back to vector retrieval with a logged warning.

The index lives in process memory and suits collections up to roughly a few
hundred thousand chunks.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from aiforge.core.vectorstore import VectorStoreAdapter
    from aiforge.infra.config import AIForgeSettings

RetrievalMode = Literal["vector", "hybrid", "keyword"]
RETRIEVAL_MODES: tuple[str, ...] = ("vector", "hybrid", "keyword")

_logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case *text* and split it into alphanumeric tokens.

    Underscores and punctuation are separators, so ``auth_middleware`` yields
    ``["auth", "middleware"]``.  No stemming or stop-word removal: BM25's IDF
    term already discounts very common words.
    """
    return _TOKEN_RE.findall(text.lower())


def retrieval_settings(module: object) -> AIForgeSettings:
    """The module's settings, or the global settings if the module has none yet."""
    from aiforge.infra.config import AIForgeSettings, get_settings

    cfg = getattr(module, "settings", None)
    return cfg if isinstance(cfg, AIForgeSettings) else get_settings()


def validate_retrieval_mode(mode: str) -> RetrievalMode:
    """Return *mode* normalised, or raise :class:`ValueError` if unknown."""
    norm = (mode or "").strip().lower()
    if norm not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {list(RETRIEVAL_MODES)}, got {mode!r}")
    return norm  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Metadata filters (subset of the Chroma / Mongo-style ``where`` syntax)
# ---------------------------------------------------------------------------

_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda v, x: v == x,
    "$ne": lambda v, x: v != x,
    "$gt": lambda v, x: v is not None and v > x,
    "$gte": lambda v, x: v is not None and v >= x,
    "$lt": lambda v, x: v is not None and v < x,
    "$lte": lambda v, x: v is not None and v <= x,
    "$in": lambda v, x: v in x,
    "$nin": lambda v, x: v not in x,
    "$contains": lambda v, x: v is not None and x in v,
}


class UnsupportedFilterError(ValueError):
    """Raised when a metadata filter uses an operator the keyword index cannot evaluate."""


def check_filter(filter: dict[str, Any] | None) -> None:
    """Raise :class:`UnsupportedFilterError` if *filter* cannot be evaluated locally."""
    if not filter:
        return
    for key, cond in filter.items():
        if key in ("$and", "$or"):
            if not isinstance(cond, list):
                raise UnsupportedFilterError(f"{key} expects a list of filters")
            for sub in cond:
                check_filter(sub)
        elif key.startswith("$"):
            raise UnsupportedFilterError(f"Unsupported filter operator {key!r}")
        elif isinstance(cond, dict):
            for op in cond:
                if op not in _COMPARATORS:
                    raise UnsupportedFilterError(f"Unsupported filter operator {op!r}")


def match_filter(metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Return whether *metadata* satisfies *filter*.

    Supports field equality (``{"lang": "py"}``), the comparison operators
    ``$eq $ne $gt $gte $lt $lte $in $nin $contains`` and the logical
    operators ``$and`` / ``$or``.  Call :func:`check_filter` first to
    reject unsupported operators.
    """
    if not filter:
        return True
    for key, cond in filter.items():
        if key == "$and":
            if not all(match_filter(metadata, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(match_filter(metadata, sub) for sub in cond):
                return False
        elif isinstance(cond, dict):
            value = metadata.get(key)
            for op, operand in cond.items():
                try:
                    if not _COMPARATORS[op](value, operand):
                        return False
                except TypeError:
                    return False
        elif metadata.get(key) != cond:
            return False
    return True


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """Pure-Python Okapi BM25 index over a set of text chunks.

    Score of document *d* for query *q*::

        sum over t in q of  IDF(t) * tf(t,d) * (k1 + 1)
                            / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

        IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    The ``+ 1`` inside the logarithm keeps IDF positive for terms that occur
    in more than half the documents (the Lucene variant).

    Args:
        k1: Term-frequency saturation.  Typical range 1.2-2.0.
        b:  Length normalisation, between 0 (none) and 1 (full).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._texts: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._texts)

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Add or replace documents (upsert semantics, keyed by id)."""
        metas = metadatas if metadatas is not None else [{}] * len(ids)
        for doc_id, text, meta in zip(ids, texts, metas, strict=True):
            if doc_id in self._texts:
                self.remove([doc_id])
            tokens = tokenize(text)
            self._texts[doc_id] = text
            self._metadata[doc_id] = dict(meta or {})
            self._lengths[doc_id] = len(tokens)
            self._total_length += len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, {})[doc_id] = tf

    def remove(self, ids: Sequence[str]) -> None:
        """Remove documents by id.  Unknown ids are ignored."""
        for doc_id in ids:
            text = self._texts.pop(doc_id, None)
            if text is None:
                continue
            self._metadata.pop(doc_id, None)
            self._total_length -= self._lengths.pop(doc_id, 0)
            for term in set(tokenize(text)):
                posting = self._postings.get(term)
                if posting is not None:
                    posting.pop(doc_id, None)
                    if not posting:
                        del self._postings[term]

    def idf(self, term: str) -> float:
        """Inverse document frequency of *term* (0.0 for an empty index)."""
        n = len(self._texts)
        if n == 0:
            return 0.0
        df = len(self._postings.get(term, {}))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """Return ``{doc_id: bm25_score}`` for every document matching a query term."""
        n = len(self._texts)
        if n == 0:
            return {}
        avgdl = self._total_length / n if self._total_length else 1.0
        out: dict[str, float] = {}
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                norm = 1.0 - self.b + self.b * self._lengths[doc_id] / avgdl
                out[doc_id] = out.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * norm
                )
        return out

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* best BM25 matches, best first.

        Each hit has ``id``, ``text``, ``metadata`` and ``bm25`` (raw score).
        Documents with a zero score (no query term) are never returned.
        """
        scored = self.scores(query)
        if filter:
            scored = {d: s for d, s in scored.items() if match_filter(self._metadata[d], filter)}
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            {
                "id": doc_id,
                "text": self._texts[doc_id],
                "metadata": dict(self._metadata[doc_id]),
                "bm25": score,
            }
            for doc_id, score in ranked
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[dict[str, Any]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse several ranked hit lists with Reciprocal Rank Fusion.

    ``rrf(d) = sum_i  w_i / (k + rank_i(d))`` with 1-based ranks; documents
    missing from a list contribute nothing for it.  Hits are matched by their
    ``id`` key.  The returned dicts merge the fields of every occurrence
    (earlier lists win on conflicts) and carry the fused value under ``rrf``.
    Ties keep first-seen order.

    Args:
        rankings: Hit lists, each sorted best first.
        k:        Rank damping constant (60 in the original paper).
        weights:  Optional per-list weights; defaults to 1.0 each.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[str, dict[str, Any]] = {}
    totals: dict[str, float] = {}
    for weight, ranking in zip(ws, rankings, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            doc_id = hit["id"]
            merged = fused.setdefault(doc_id, {})
            for key, value in hit.items():
                merged.setdefault(key, value)
            totals[doc_id] = totals.get(doc_id, 0.0) + weight / (k + rank)
    order = sorted(fused, key=lambda d: -totals[d])  # stable: ties keep first-seen order
    out = []
    for doc_id in order:
        hit = fused[doc_id]
        hit["rrf"] = totals[doc_id]
        out.append(hit)
    return out


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Vector, keyword or hybrid retrieval over one vector store collection.

    Every hit returned by :meth:`retrieve` is a dict with ``id``, ``text``,
    ``metadata`` and ``score`` in ``[0, 1]``.  Vector hits also keep
    ``distance`` (and ``embeddings`` when requested).  The meaning of
    ``score`` depends on the mode:

    * ``vector``: ``1 - cosine distance`` (unchanged).
    * ``keyword``: BM25 score relative to the best hit (best hit = 1.0).
    * ``hybrid``: RRF score relative to the best possible fused score
      (a document ranked first by both retrievers = 1.0).

    Args:
        store:      The vector store to query.
        collection: Collection name, used to detect external writes via
                    :meth:`VectorStoreAdapter.collection_stats`.
        mode:       ``"vector"``, ``"hybrid"`` or ``"keyword"``.
        rrf_k:      RRF damping constant.
    """

    def __init__(
        self,
        store: VectorStoreAdapter,
        collection: str,
        mode: str = "vector",
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.collection = collection
        self.mode: RetrievalMode = validate_retrieval_mode(mode)
        self.rrf_k = rrf_k
        self._index: BM25Index | None = None
        self._keyword_supported: bool | None = None
        self._lock = threading.Lock()

    # -- index lifecycle -------------------------------------------------

    def invalidate(self) -> None:
        """Drop the cached BM25 index; it is rebuilt on the next keyword query."""
        with self._lock:
            self._index = None

    def keyword_index(self) -> BM25Index | None:
        """Return an up-to-date BM25 index, or ``None`` if the store can't list documents."""
        with self._lock:
            if self._keyword_supported is False:
                return None
            if self._index is not None and len(self._index) == self._store_count():
                return self._index
            try:
                docs = self.store.list_all_documents()
            except NotImplementedError:
                self._keyword_supported = False
                _logger.warning(
                    "retrieval.keyword_unsupported",
                    store=type(self.store).__name__,
                    msg="Vector store cannot list documents; falling back to vector retrieval",
                )
                return None
            index = BM25Index()
            index.add(
                [d["id"] for d in docs],
                [d.get("text") or "" for d in docs],
                [d.get("metadata") or {} for d in docs],
            )
            self._keyword_supported = True
            self._index = index
            _logger.debug("retrieval.bm25_built", collection=self.collection, docs=len(index))
            return index

    def _store_count(self) -> int:
        try:
            return int(self.store.collection_stats(self.collection).get("chunks", 0))
        except Exception:  # noqa: BLE001 — a failing stats call must not break retrieval
            return -1

    # -- mode resolution -------------------------------------------------

    def effective_mode(self, filter: dict[str, Any] | None = None) -> RetrievalMode:
        """The mode a query with *filter* will actually run in.

        Returns ``"vector"`` (and logs a warning) when keyword retrieval is
        unavailable for this store or *filter* uses an operator the keyword
        index cannot evaluate.  Callers use this to decide whether a query
        embedding is needed.
        """
        if self.mode == "vector":
            return "vector"
        try:
            check_filter(filter)
        except UnsupportedFilterError as exc:
            _logger.warning("retrieval.filter_unsupported", error=str(exc), mode=self.mode)
            return "vector"
        if self.keyword_index() is None:
            return "vector"
        return self.mode

    # -- querying --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        embedding: list[float] | None = None,
        filter: dict[str, Any] | None = None,
        include_embeddings: bool = False,
        candidate_k: int | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to *top_k* hits for *query*, best first.

        Args:
            query:              Query text (used by the keyword index).
            top_k:              Number of hits to return.
            embedding:          Query embedding; required for vector and
                                hybrid mode.
            filter:             Metadata filter (Chroma ``where`` syntax).
            include_embeddings: Ask the vector store for raw vectors.
            candidate_k:        Hits fetched from each retriever before
                                fusion; defaults to *top_k*.
            mode:               Mode to run in; defaults to
                                :meth:`effective_mode`.
        """
        run_mode = mode or self.effective_mode(filter)
        fetch = max(top_k, candidate_k or top_k)

        if run_mode == "vector":
            if embedding is None:
                raise ValueError("embedding is required for vector retrieval")
            hits = self.store.query(
                embedding=embedding,
                top_k=top_k,
                filter=filter,
                include_embeddings=include_embeddings,
            )
            for h in hits:
                h["score"] = _clamp(1.0 - float(h.get("distance", 1.0)))
            return hits

        index = self.keyword_index()
        keyword_hits = index.search(query, top_k=fetch, filter=filter) if index else []

        if run_mode == "keyword":
            best = keyword_hits[0]["bm25"] if keyword_hits else 0.0
            for h in keyword_hits:
                h["score"] = _clamp(h["bm25"] / best) if best > 0 else 0.0
            return keyword_hits[:top_k]

        if embedding is None:
            raise ValueError("embedding is required for hybrid retrieval")
        vector_hits = self.store.query(
            embedding=embedding,
            top_k=fetch,
            filter=filter,
            include_embeddings=include_embeddings,
        )
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self.rrf_k)
        best_possible = 2.0 / (self.rrf_k + 1)
        for h in fused:
            h["score"] = _clamp(h["rrf"] / best_possible)
        return fused[:top_k]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))
