"""Summarizer module — text summarisation with auto stuff/map-reduce strategy."""

from __future__ import annotations

import time
from typing import Any

from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.types import BaseModule
from aiforge.modules.summarize.prompts import (
    SUMMARIZE_BULLET_POINTS,
    SUMMARIZE_CONCISE,
    SUMMARIZE_DETAILED,
    SUMMARIZE_EXECUTIVE,
    SUMMARIZE_MAP,
    SUMMARIZE_REDUCE,
    SUMMARIZE_SYSTEM,
)
from aiforge.modules.summarize.schemas import SummarizeResponse, SummarizeStyle

_STYLE_PROMPTS: dict[str, str] = {
    "concise": SUMMARIZE_CONCISE,
    "detailed": SUMMARIZE_DETAILED,
    "bullet_points": SUMMARIZE_BULLET_POINTS,
    "executive": SUMMARIZE_EXECUTIVE,
}

# Context-window usage thresholds
_STUFF_THRESHOLD_RATIO = 0.8   # auto-select map_reduce above this
_CHUNK_SIZE_RATIO = 0.6        # each chunk ≈ 60 % of context window
_CHUNK_OVERLAP = 200           # character overlap between chunks


def _count_tokens(text: str, model: str) -> int:
    """Estimate token count for *text* using tiktoken.

    Falls back to a ``len(text) // 4`` heuristic when tiktoken doesn't
    recognise the model (e.g. Ollama / Anthropic models).
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def _split_text(text: str, chunk_size: int, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping character-based chunks.

    Uses the core TextSplitter when available, otherwise falls back to a
    simple character-based split.
    """
    try:
        from aiforge.core.text_splitter import TextSplitter
        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text, ext=".txt")
    except Exception:  # noqa: BLE001
        # Simple fallback — stride must always be >= 1 to avoid infinite loop
        stride = max(1, chunk_size - overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + chunk_size])
            start += stride
        return chunks or [text]


class Summarizer(BaseModule):
    """Text summarisation with automatic stuff / map-reduce strategy selection.

    When the input text fits within the model context window (≤ 80 % of
    max tokens), the ``stuff`` strategy sends everything in one call.
    Otherwise ``map_reduce`` splits the text into overlapping chunks,
    summarises each independently, then combines the partial summaries.

    Args:
        style:       Summary style — ``"concise"``, ``"detailed"``,
                     ``"bullet_points"``, or ``"executive"``.
        max_length:  Target summary length in words.
        temperature: LLM sampling temperature (0.0–2.0).
        **kwargs:    Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        style: SummarizeStyle = "concise",
        max_length: int = 500,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if style not in _STYLE_PROMPTS:
            raise ValueError(f"style must be one of {list(_STYLE_PROMPTS)}")
        self.style: SummarizeStyle = style
        self.max_length: int = max_length
        self.temperature: float = temperature
        self._system_prompt_override: str | None = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        *,
        strategy: str | None = None,
        system_prompt: str | None = None,
    ) -> SummarizeResponse:
        """Summarize *text* synchronously.

        Args:
            text:     Source text to summarise.
            strategy: Force ``"stuff"`` or ``"map_reduce"``.  ``None`` auto-
                      selects based on token count vs. context window.
            system_prompt: Per-call system prompt override.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        chosen = self._choose_strategy(text, strategy)
        resolved = self._resolve_system_prompt(system_prompt)
        if chosen == "stuff":
            return self._stuff(text, t0, resolved)
        return self._map_reduce(text, t0, resolved)

    async def asummarize(
        self,
        text: str,
        *,
        strategy: str | None = None,
        system_prompt: str | None = None,
    ) -> SummarizeResponse:
        """Async version of :meth:`summarize`."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        chosen = self._choose_strategy(text, strategy)
        resolved = self._resolve_system_prompt(system_prompt)
        if chosen == "stuff":
            return await self._astuff(text, t0, resolved)
        return await self._amap_reduce(text, t0, resolved)

    def run(self, text: str, **kwargs: Any) -> SummarizeResponse:
        """Pipeline entry point (sync)."""
        return self.summarize(text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> SummarizeResponse:
        """Pipeline entry point (async)."""
        return await self.asummarize(text, **kwargs)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _choose_strategy(self, text: str, explicit: str | None) -> str:
        """Return ``'stuff'`` or ``'map_reduce'``."""
        if explicit is not None:
            if explicit not in ("stuff", "map_reduce"):
                raise ValueError("strategy must be 'stuff' or 'map_reduce'")
            return explicit
        from aiforge.core.pricing import get_context_window
        ctx = get_context_window(self.settings.MODEL)
        token_count = _count_tokens(text, self.settings.MODEL)
        threshold = int(ctx * _STUFF_THRESHOLD_RATIO)
        return "map_reduce" if token_count > threshold else "stuff"

    # ------------------------------------------------------------------
    # Prompt override resolution
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self, call_override: str | None = None) -> str:
        return resolve_prompt(
            call_override=call_override,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("summarize", {}).get("system"),
            default=SUMMARIZE_SYSTEM,
        )

    # ------------------------------------------------------------------
    # Stuff strategy (single call)
    # ------------------------------------------------------------------

    def _build_stuff_messages(self, text: str, system: str) -> list[dict[str, str]]:
        """Construct messages for a single-call summarisation."""
        template = _STYLE_PROMPTS[self.style]
        user_content = template.format(text=text, max_length=self.max_length)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _stuff(self, text: str, t0: float, system: str) -> SummarizeResponse:
        messages = self._build_stuff_messages(text, system)
        llm = self._call_llm(messages, temperature=self.temperature)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        return SummarizeResponse(
            summary=llm.text.strip(),
            strategy_used="stuff",
            chunks_processed=1,
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def _astuff(self, text: str, t0: float, system: str) -> SummarizeResponse:
        messages = self._build_stuff_messages(text, system)
        llm = await self._acall_llm(messages, temperature=self.temperature)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        return SummarizeResponse(
            summary=llm.text.strip(),
            strategy_used="stuff",
            chunks_processed=1,
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Map-reduce strategy
    # ------------------------------------------------------------------

    def _get_chunk_size(self) -> int:
        from aiforge.core.pricing import get_context_window
        ctx = get_context_window(self.settings.MODEL)
        return int(ctx * _CHUNK_SIZE_RATIO)

    def _build_map_messages(self, chunk: str, system: str) -> list[dict[str, str]]:
        user_content = SUMMARIZE_MAP.format(chunk=chunk)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _build_reduce_messages(self, summaries: str, system: str) -> list[dict[str, str]]:
        user_content = SUMMARIZE_REDUCE.format(
            summaries=summaries, max_length=self.max_length,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _map_reduce(self, text: str, t0: float, system: str) -> SummarizeResponse:
        chunk_size = self._get_chunk_size()
        chunks = _split_text(text, chunk_size)

        total_in = total_out = 0
        total_cost = 0.0
        model = ""
        chunk_summaries: list[str] = []

        # Map phase
        for chunk in chunks:
            messages = self._build_map_messages(chunk, system)
            llm = self._call_llm(messages, temperature=self.temperature)
            from aiforge.core.pricing import calculate_cost
            cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += cost
            model = llm.model
            chunk_summaries.append(llm.text.strip())

        # Reduce phase
        combined = "\n\n".join(chunk_summaries)
        messages = self._build_reduce_messages(combined, system)
        llm = self._call_llm(messages, temperature=self.temperature)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        total_in += llm.input_tokens
        total_out += llm.output_tokens
        total_cost += cost
        model = llm.model

        return SummarizeResponse(
            summary=llm.text.strip(),
            strategy_used="map_reduce",
            chunks_processed=len(chunks),
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def _amap_reduce(self, text: str, t0: float, system: str) -> SummarizeResponse:
        chunk_size = self._get_chunk_size()
        chunks = _split_text(text, chunk_size)

        total_in = total_out = 0
        total_cost = 0.0
        model = ""
        chunk_summaries: list[str] = []

        # Map phase
        for chunk in chunks:
            messages = self._build_map_messages(chunk, system)
            llm = await self._acall_llm(messages, temperature=self.temperature)
            from aiforge.core.pricing import calculate_cost
            cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += cost
            model = llm.model
            chunk_summaries.append(llm.text.strip())

        # Reduce phase (async)
        combined = "\n\n".join(chunk_summaries)
        messages = self._build_reduce_messages(combined, system)
        llm = await self._acall_llm(messages, temperature=self.temperature)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        total_in += llm.input_tokens
        total_out += llm.output_tokens
        total_cost += cost
        model = llm.model

        return SummarizeResponse(
            summary=llm.text.strip(),
            strategy_used="map_reduce",
            chunks_processed=len(chunks),
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
