"""
Classifier module \u2014 zero/few-shot text classification.

Usage::

    from aiforge import Classifier

    # Zero-shot, single label
    c = Classifier(labels=["positive", "negative", "neutral"])
    result = c.classify("I love this product!")
    print(result.label)      # "positive"
    print(result.rationale)  # "Text expresses strong positive sentiment."

    # Few-shot, single label
    from aiforge.modules.classify.schemas import ClassifyExample
    c = Classifier(
        labels=["sports", "politics", "technology"],
        examples=[
            ClassifyExample(text="The team scored in overtime.", label="sports"),
            ClassifyExample(text="The senator voted on the bill.", label="politics"),
        ],
    )
    result = c.classify("Apple released a new chip design.")
    print(result.label)  # "technology"

    # Multi-label
    c = Classifier(labels=["sports", "finance", "technology"], multi_label=True)
    result = c.classify("Nike's stock surged after their new shoe launch.")
    print([s.label for s in result.scores])  # ["finance", "sports"]
"""

from __future__ import annotations

import json
import time
from typing import Any

from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.types import BaseModule
from aiforge.modules.classify.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER_FEW_SHOT,
    CLASSIFY_USER_MULTI_FEW_SHOT,
    CLASSIFY_USER_MULTI_ZERO_SHOT,
    CLASSIFY_USER_ZERO_SHOT,
    SENTIMENT_SYSTEM,
    SENTIMENT_USER,
    SENTIMENT_USER_ASPECTS,
)
from aiforge.modules.classify.schemas import (
    AspectSentiment,
    ClassifyExample,
    ClassifyResponse,
    EmotionScore,
    LabelScore,
    SentimentResponse,
)

_LABELS_MAX = 100


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown code fences from LLM output before JSON parsing."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip().rsplit("```", 1)[0]
    return text.strip()


def _fmt_labels(labels: list[str]) -> str:
    """Format label list as a bulleted block for prompt insertion."""
    return "\n".join(f"  - {lbl}" for lbl in labels)


def _fmt_examples(examples: list[ClassifyExample]) -> str:
    """Format few-shot examples as a readable block for prompt insertion.

    Escapes backslashes and double-quotes in example text so the prompt
    structure is not corrupted by user-supplied content (M1 / C2).
    """
    lines = []
    for ex in examples:
        safe_text = ex.text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  Text: "{safe_text}"\n  Label: {ex.label}')
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class Classifier(BaseModule):
    """Zero/few-shot text classifier backed by an LLM.

    Args:
        labels:      Non-empty list of label strings forming the taxonomy.
                     Duplicates are silently removed (order preserved).
        examples:    Optional few-shot examples. Each must reference a label
                     present in *labels*.
        multi_label: When ``True``, the LLM may return multiple applicable
                     labels instead of a single best match.
        **kwargs:    Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        labels: list[str],
        examples: list[ClassifyExample] | None = None,
        multi_label: bool = False,
        temperature: float = 0.0,
        min_confidence: float = 0.0,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not labels:
            raise ValueError("labels must be non-empty")
        if len(labels) > _LABELS_MAX:
            raise ValueError(f"labels must have at most {_LABELS_MAX} entries")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.0, 1.0]")
        # Preserve order, remove duplicates
        deduped = list(dict.fromkeys(labels))
        if len(deduped) != len(labels):
            self.logger.warning(
                "classify.duplicate_labels_removed",
                original=len(labels),
                deduped=len(deduped),
            )
        self.labels: list[str] = deduped
        self.examples: list[ClassifyExample] = examples or []
        self.multi_label: bool = multi_label
        self.temperature: float = temperature
        self.min_confidence: float = min_confidence
        self._system_prompt_override: str | None = system_prompt
        # Validate that all examples reference known labels
        label_set = set(self.labels)
        bad = [ex.label for ex in self.examples if ex.label not in label_set]
        if bad:
            raise ValueError(f"Example labels not in taxonomy: {bad}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str, *, system_prompt: str | None = None) -> ClassifyResponse:
        """Classify *text* synchronously and return a :class:`ClassifyResponse`."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        llm = self._call_llm(self._build_messages(text, system_prompt=system_prompt), temperature=self.temperature)
        result = self._parse_response(llm.text)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        return ClassifyResponse(
            label=result.label,
            scores=result.scores,
            rationale=result.rationale,
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aclassify(self, text: str, *, system_prompt: str | None = None) -> ClassifyResponse:
        """Async version of :meth:`classify`."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        llm = await self._acall_llm(self._build_messages(text, system_prompt=system_prompt), temperature=self.temperature)
        result = self._parse_response(llm.text)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        return ClassifyResponse(
            label=result.label,
            scores=result.scores,
            rationale=result.rationale,
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def run(self, text: str, **kwargs: Any) -> ClassifyResponse:
        """Pipeline entry point — delegates to :meth:`classify`."""
        sp = kwargs.pop("system_prompt", None)
        return self.classify(text, system_prompt=sp)

    async def arun(self, text: str, **kwargs: Any) -> ClassifyResponse:
        """Async pipeline entry point — delegates to :meth:`aclassify`."""
        sp = kwargs.pop("system_prompt", None)
        return await self.aclassify(text, system_prompt=sp)

    # ------------------------------------------------------------------
    # Sentiment API
    # ------------------------------------------------------------------

    def sentiment(
        self,
        text: str,
        *,
        aspects: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> SentimentResponse:
        """Analyze sentiment of *text* synchronously.

        Args:
            text:          Input text to analyze.
            aspects:       Optional list of aspects to evaluate individually
                           (e.g. ``["price", "quality", "service"]``).
            system_prompt: Override the system prompt for this call.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        messages = self._build_sentiment_messages(text, aspects, system_prompt)
        llm = self._call_llm(messages, temperature=self.temperature)
        result = self._parse_sentiment(llm.text, aspects)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        result.cost = cost
        result.input_tokens = llm.input_tokens
        result.output_tokens = llm.output_tokens
        result.model = llm.model
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    async def asentiment(
        self,
        text: str,
        *,
        aspects: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> SentimentResponse:
        """Async version of :meth:`sentiment`."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        t0 = time.monotonic()
        messages = self._build_sentiment_messages(text, aspects, system_prompt)
        llm = await self._acall_llm(messages, temperature=self.temperature)
        result = self._parse_sentiment(llm.text, aspects)
        from aiforge.core.pricing import calculate_cost
        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
        result.cost = cost
        result.input_tokens = llm.input_tokens
        result.output_tokens = llm.output_tokens
        result.model = llm.model
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    def _build_sentiment_messages(
        self,
        text: str,
        aspects: list[str] | None,
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        """Construct messages for a sentiment analysis call."""
        resolved_system = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("classify", {}).get("sentiment_system"),
            default=SENTIMENT_SYSTEM,
        )
        if aspects:
            aspects_list = "\n".join(f"  - {a}" for a in aspects)
            user_content = SENTIMENT_USER_ASPECTS.format(aspects_list=aspects_list, text=text)
        else:
            user_content = SENTIMENT_USER.format(text=text)
        return [
            {"role": "system", "content": resolved_system},
            {"role": "user", "content": user_content},
        ]

    def _parse_sentiment(self, llm_text: str, aspects: list[str] | None) -> SentimentResponse:
        """Parse LLM JSON output into a :class:`SentimentResponse`."""
        try:
            data = json.loads(_extract_json(llm_text))
        except json.JSONDecodeError:
            self.logger.warning("classify.sentiment_json_parse_failed", raw_head=llm_text[:200])
            return SentimentResponse()

        overall = str(data.get("overall", "neutral")).lower()
        if overall not in ("positive", "negative", "neutral", "mixed"):
            overall = "neutral"

        try:
            score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
        except (ValueError, TypeError):
            score = 0.0

        emotions: list[EmotionScore] = []
        for e in data.get("emotions", []):
            try:
                intensity = max(0.0, min(1.0, float(e.get("intensity", 1.0))))
            except (ValueError, TypeError):
                intensity = 1.0
            emotions.append(EmotionScore(emotion=str(e.get("emotion", "")), intensity=intensity))

        aspect_results: list[AspectSentiment] = []
        for a in data.get("aspects", []):
            a_sent = str(a.get("sentiment", "neutral")).lower()
            if a_sent not in ("positive", "negative", "neutral", "mixed"):
                a_sent = "neutral"
            try:
                a_score = max(-1.0, min(1.0, float(a.get("score", 0.0))))
            except (ValueError, TypeError):
                a_score = 0.0
            aspect_results.append(
                AspectSentiment(aspect=str(a.get("aspect", "")), sentiment=a_sent, score=a_score)
            )

        return SentimentResponse(
            overall=overall,
            score=score,
            aspects=aspect_results,
            emotions=emotions,
            rationale=str(data.get("rationale", "")),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, text: str, *, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Construct the system + user message list for the LLM call."""
        resolved_system = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("classify", {}).get("system"),
            default=CLASSIFY_SYSTEM,
        )
        labels_list = _fmt_labels(self.labels)
        if self.multi_label:
            if self.examples:
                user_content = CLASSIFY_USER_MULTI_FEW_SHOT.format(
                    labels_list=labels_list,
                    examples_block=_fmt_examples(self.examples),
                    text=text,
                )
            else:
                user_content = CLASSIFY_USER_MULTI_ZERO_SHOT.format(
                    labels_list=labels_list,
                    text=text,
                )
        else:
            if self.examples:
                user_content = CLASSIFY_USER_FEW_SHOT.format(
                    labels_list=labels_list,
                    examples_block=_fmt_examples(self.examples),
                    text=text,
                )
            else:
                user_content = CLASSIFY_USER_ZERO_SHOT.format(
                    labels_list=labels_list,
                    text=text,
                )
        return [
            {"role": "system", "content": resolved_system},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, llm_text: str) -> ClassifyResponse:
        """Parse LLM JSON output into a :class:`ClassifyResponse`.

        Gracefully degrades on malformed output — returns
        ``ClassifyResponse(label=None)`` rather than raising.

        Confidence values are clamped to ``[0.0, 1.0]`` (C1).
        Labels are matched case-insensitively as a fallback (I6).
        Results below ``self.min_confidence`` return ``label=None`` (M6).
        """
        try:
            data = json.loads(_extract_json(llm_text))
        except json.JSONDecodeError:
            self.logger.warning("classify.json_parse_failed", raw_head=llm_text[:200])
            self.logger.debug("classify.json_parse_failed_full", raw=llm_text)
            return ClassifyResponse(label=None, scores=[], rationale="")

        rationale: str = str(data.get("rationale", ""))
        label_set = set(self.labels)

        def _resolve(raw: str) -> str | None:
            """Exact match, then case-insensitive fallback."""
            if raw in label_set:
                return raw
            matched = next((lbl for lbl in self.labels if lbl.lower() == raw.lower()), None)
            if matched:
                self.logger.warning(
                    "classify.label_case_mismatch", returned=raw, matched=matched
                )
            return matched

        if self.multi_label:
            raw_entries: list[dict] = data.get("labels", [])
            scores: list[LabelScore] = []
            for entry in raw_entries:
                lbl = str(entry.get("label", ""))
                try:
                    conf = max(0.0, min(1.0, float(entry.get("confidence", 1.0))))
                except (ValueError, TypeError):
                    conf = 1.0
                resolved = _resolve(lbl)
                if resolved is not None:
                    if conf >= self.min_confidence:
                        scores.append(LabelScore(label=resolved, confidence=conf))
                else:
                    self.logger.warning("classify.unknown_label_ignored", label=lbl)
            scores.sort(key=lambda s: s.confidence, reverse=True)
            top = scores[0].label if scores else None
            return ClassifyResponse(label=top, scores=scores, rationale=rationale)
        else:
            raw_label = str(data.get("label", ""))
            try:
                confidence = max(0.0, min(1.0, float(data.get("confidence", 1.0))))
            except (ValueError, TypeError):
                confidence = 1.0
            resolved = _resolve(raw_label)
            if resolved is None:
                self.logger.warning(
                    "classify.label_not_in_taxonomy",
                    label=raw_label,
                    taxonomy=self.labels,
                )
                return ClassifyResponse(label=None, scores=[], rationale=rationale)
            if confidence < self.min_confidence:
                self.logger.warning(
                    "classify.confidence_below_threshold",
                    label=resolved,
                    confidence=confidence,
                    min_confidence=self.min_confidence,
                )
                return ClassifyResponse(label=None, scores=[], rationale=rationale)
            return ClassifyResponse(
                label=resolved,
                scores=[LabelScore(label=resolved, confidence=confidence)],
                rationale=rationale,
            )
