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

from aiforge.core.types import BaseModule
from aiforge.modules.classify.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER_FEW_SHOT,
    CLASSIFY_USER_MULTI_FEW_SHOT,
    CLASSIFY_USER_MULTI_ZERO_SHOT,
    CLASSIFY_USER_ZERO_SHOT,
)
from aiforge.modules.classify.schemas import ClassifyExample, ClassifyResponse, LabelScore

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
    """Format few-shot examples as a readable block for prompt insertion."""
    lines = []
    for ex in examples:
        lines.append(f'  Text: "{ex.text}"\n  Label: {ex.label}')
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not labels:
            raise ValueError("labels must be non-empty")
        if len(labels) > _LABELS_MAX:
            raise ValueError(f"labels must have at most {_LABELS_MAX} entries")
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
        # Validate that all examples reference known labels
        label_set = set(self.labels)
        bad = [ex.label for ex in self.examples if ex.label not in label_set]
        if bad:
            raise ValueError(f"Example labels not in taxonomy: {bad}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> ClassifyResponse:
        """Classify *text* synchronously and return a :class:`ClassifyResponse`."""
        t0 = time.monotonic()
        llm = self._call_llm(self._build_messages(text), temperature=0.0)
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

    async def aclassify(self, text: str) -> ClassifyResponse:
        """Async version of :meth:`classify`."""
        t0 = time.monotonic()
        llm = await self._acall_llm(self._build_messages(text), temperature=0.0)
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

    def run(self, text: str, **_kwargs: Any) -> ClassifyResponse:
        """Pipeline entry point \u2014 delegates to :meth:`classify`."""
        return self.classify(text)

    async def arun(self, text: str, **_kwargs: Any) -> ClassifyResponse:
        """Async pipeline entry point \u2014 delegates to :meth:`aclassify`."""
        return await self.aclassify(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        """Construct the system + user message list for the LLM call."""
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
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, llm_text: str) -> ClassifyResponse:
        """Parse LLM JSON output into a :class:`ClassifyResponse`.

        Gracefully degrades on malformed output: returns
        ``ClassifyResponse(label=None)`` rather than raising.
        """
        try:
            data = json.loads(_extract_json(llm_text))
        except json.JSONDecodeError:
            self.logger.warning("classify.json_parse_failed", raw=llm_text[:200])
            return ClassifyResponse(label=None, scores=[], rationale="")

        rationale: str = str(data.get("rationale", ""))
        label_set = set(self.labels)

        if self.multi_label:
            raw_entries: list[dict] = data.get("labels", [])
            scores: list[LabelScore] = []
            for entry in raw_entries:
                lbl = str(entry.get("label", ""))
                conf = float(entry.get("confidence", 1.0))
                if lbl in label_set:
                    scores.append(LabelScore(label=lbl, confidence=conf))
                else:
                    self.logger.warning("classify.unknown_label_ignored", label=lbl)
            scores.sort(key=lambda s: s.confidence, reverse=True)
            top = scores[0].label if scores else None
            return ClassifyResponse(label=top, scores=scores, rationale=rationale)
        else:
            raw_label = str(data.get("label", ""))
            confidence = float(data.get("confidence", 1.0))
            if raw_label not in label_set:
                self.logger.warning(
                    "classify.label_not_in_taxonomy",
                    label=raw_label,
                    taxonomy=self.labels,
                )
                return ClassifyResponse(label=None, scores=[], rationale=rationale)
            return ClassifyResponse(
                label=raw_label,
                scores=[LabelScore(label=raw_label, confidence=confidence)],
                rationale=rationale,
            )
