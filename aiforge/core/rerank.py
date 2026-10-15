"""
Rerankers: re-order retrieved candidate chunks by relevance to the query.

A reranker is any object with a ``rerank(query, candidates, top_n)`` method
(see :class:`Reranker`).  Two implementations ship with AIForge:

* :class:`LLMReranker` scores every candidate in **one** batched LLM call
  using the provider's native structured output, and falls back to the
  original order on any failure.
* :class:`ScoreFunctionReranker` wraps any ``(query, texts) -> scores``
  callable, e.g. a cross-encoder you load yourself::

      from sentence_transformers import CrossEncoder

      ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
      reranker = ScoreFunctionReranker(
          lambda q, docs: ce.predict([(q, d) for d in docs]).tolist()
      )
      qa = DocQA(collection="docs", reranker=reranker)

Candidates are the hit dicts produced by retrieval (``id``, ``text``,
``metadata``, ``score``, …).  Rerankers return them re-ordered, best first,
truncated to ``top_n``, with the new relevance under ``rerank_score``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aiforge.core.provider import LLMProvider
    from aiforge.core.types import BaseModule, CompletionResult

_logger = structlog.get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    """Protocol every reranker implements."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Return up to *top_n* of *candidates*, most relevant first.

        Implementations should set ``rerank_score`` on each returned hit and
        must not raise for ordinary scoring failures: returning the original
        order is the expected fallback.
        """
        ...


# ---------------------------------------------------------------------------
# Score-function reranker (cross-encoders, custom models)
# ---------------------------------------------------------------------------


class ScoreFunctionReranker:
    """Rerank with any ``(query, texts) -> scores`` callable.

    Higher scores mean more relevant; any numeric scale is fine.  If the
    function raises or returns the wrong number of scores, the original
    order is kept.

    Args:
        score_fn: Called once per rerank with the query and all candidate
                  texts; returns one float per text.
    """

    def __init__(self, score_fn: Callable[[str, list[str]], Sequence[float]]) -> None:
        self.score_fn = score_fn

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        try:
            scores = [float(s) for s in self.score_fn(query, [c["text"] for c in candidates])]
            if len(scores) != len(candidates):
                raise ValueError(f"expected {len(candidates)} scores, got {len(scores)}")
        except Exception as exc:  # noqa: BLE001 — fall back to retrieval order
            _logger.warning("rerank.score_fn_failed", error=str(exc))
            return [dict(c) for c in candidates[:top_n]]
        return _apply_scores(candidates, dict(enumerate(scores)), top_n)


# ---------------------------------------------------------------------------
# LLM reranker
# ---------------------------------------------------------------------------


class ChunkScore(BaseModel):
    """Relevance score for one candidate chunk."""

    index: int = Field(description="The candidate's index as shown in the prompt")
    score: float = Field(description="Relevance from 0.0 (irrelevant) to 1.0 (answers it)")


class RerankScores(BaseModel):
    """Structured output schema for :class:`LLMReranker`."""

    scores: list[ChunkScore]


RERANK_PROMPT = """\
Score how relevant each candidate passage is to the query.

Query: {query}

Candidates:
{candidates}

Return one entry per candidate with its index and a relevance score between \
0.0 (irrelevant) and 1.0 (directly answers the query). Judge only relevance \
to the query, not writing quality."""


class LLMReranker:
    """Score candidates for relevance with one structured-output LLM call.

    The call asks for ``{"scores": [{"index": i, "score": s}, …]}`` through
    the provider's native ``response_format``.  Providers without native
    structured output are asked for the same JSON in plain text.  On any
    error (call failure, unparsable output) the original order is returned.

    Args:
        llm:       Where to send the call.  A module (e.g. a
                   :class:`~aiforge.modules.qa.module.DocQA`) routes it
                   through ``_call_llm`` so caching, rate limits and cost
                   tracking apply; an :class:`~aiforge.core.provider.LLMProvider`
                   is called directly.
        max_chars: Each candidate's text is truncated to this many
                   characters in the prompt.
    """

    def __init__(self, llm: BaseModule | LLMProvider, *, max_chars: int = 1500) -> None:
        self.llm = llm
        self.max_chars = max_chars

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        hits, _ = self.rerank_with_result(query, candidates, top_n)
        return hits

    def rerank_with_result(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int,
    ) -> tuple[list[dict[str, Any]], CompletionResult | None]:
        """Like :meth:`rerank`, also returning the LLM call's result (for usage/cost).

        The result is ``None`` when no call was made or the call raised.
        """
        if not candidates:
            return [], None
        result: CompletionResult | None = None
        try:
            result = self._complete(self._messages(query, candidates))
            scores = self._parse(result, len(candidates))
        except Exception as exc:  # noqa: BLE001 — fall back to retrieval order
            _logger.warning("rerank.llm_failed", error=str(exc), candidates=len(candidates))
            return [dict(c) for c in candidates[:top_n]], result
        return _apply_scores(candidates, scores, top_n), result

    # -- internals -------------------------------------------------------

    def _messages(self, query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        blocks = "\n\n".join(
            f"[{i}] {(c.get('text') or '')[: self.max_chars]}" for i, c in enumerate(candidates)
        )
        prompt = RERANK_PROMPT.format(query=query, candidates=blocks)
        return [{"role": "user", "content": prompt}]

    def _complete(self, messages: list[dict[str, Any]]) -> CompletionResult:
        call_llm = getattr(self.llm, "_call_llm", None)
        if call_llm is not None:
            structured = self.llm._structured_kwargs(RerankScores)  # type: ignore[union-attr]
            if not structured:
                messages = _with_json_instruction(messages)
            result: CompletionResult = call_llm(messages, temperature=0.0, **structured)
            return result
        provider: LLMProvider = self.llm  # type: ignore[assignment]
        if getattr(provider, "supports_structured_output", False):
            return provider.complete(messages, temperature=0.0, response_format=RerankScores)
        return provider.complete(_with_json_instruction(messages), temperature=0.0)

    @staticmethod
    def _parse(result: CompletionResult, n: int) -> dict[int, float]:
        payload: Any = result.parsed
        if payload is None:
            text = result.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{") :] if "{" in text else text
            payload = json.loads(text)
        parsed = RerankScores.model_validate(payload)
        scores = {
            s.index: max(0.0, min(1.0, float(s.score))) for s in parsed.scores if 0 <= s.index < n
        }
        if not scores:
            raise ValueError("reranker returned no valid scores")
        return scores


def apply_reranker(
    reranker: Reranker,
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_n: int,
) -> tuple[list[dict[str, Any]], CompletionResult | None]:
    """Run *reranker*, returning ``(hits, llm_result_or_None)``.

    Exceptions raised by a custom reranker are logged and the original order
    is kept, so a broken reranker never breaks retrieval.
    """
    if isinstance(reranker, LLMReranker):
        return reranker.rerank_with_result(query, candidates, top_n)
    try:
        return list(reranker.rerank(query, list(candidates), top_n))[:top_n], None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("rerank.failed", reranker=type(reranker).__name__, error=str(exc))
        return [dict(c) for c in candidates[:top_n]], None


def _with_json_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schema = json.dumps(RerankScores.model_json_schema())
    suffix = f"\n\nRespond with JSON only, matching this schema:\n{schema}"
    out = [dict(m) for m in messages]
    out[-1]["content"] = f"{out[-1]['content']}{suffix}"
    return out


def _apply_scores(
    candidates: Sequence[dict[str, Any]],
    scores: dict[int, float],
    top_n: int,
) -> list[dict[str, Any]]:
    """Order *candidates* by *scores* (index → score), unscored ones last in original order."""
    scored = sorted(scores, key=lambda i: (-scores[i], i))
    unscored = [i for i in range(len(candidates)) if i not in scores]
    out: list[dict[str, Any]] = []
    for i in [*scored, *unscored][:top_n]:
        hit = dict(candidates[i])
        hit["rerank_score"] = scores.get(i)
        out.append(hit)
    return out
