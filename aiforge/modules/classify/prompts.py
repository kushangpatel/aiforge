"""Prompt templates for the Classifier module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt (shared)
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You are a text classification engine. Assign the most accurate label from "
    "the provided taxonomy to the given text. Be precise and consistent. "
    "Always respond ONLY with valid JSON \u2014 no markdown, no extra prose."
)

# ---------------------------------------------------------------------------
# Single-label templates
# ---------------------------------------------------------------------------

CLASSIFY_USER_ZERO_SHOT = """Classify the following text into exactly one of these labels:
{labels_list}

Respond ONLY with valid JSON:
{{"label": "<chosen_label>", "confidence": <float 0.0-1.0>, "rationale": "<one sentence>"}}

Text:
<text>
{text}
</text>"""

CLASSIFY_USER_FEW_SHOT = """Classify the following text into exactly one of these labels:
{labels_list}

Examples:
{examples_block}

Respond ONLY with valid JSON:
{{"label": "<chosen_label>", "confidence": <float 0.0-1.0>, "rationale": "<one sentence>"}}

Text:
<text>
{text}
</text>"""

# ---------------------------------------------------------------------------
# Multi-label templates
# ---------------------------------------------------------------------------

CLASSIFY_USER_MULTI_ZERO_SHOT = """Classify the following text. Assign ALL applicable labels from:
{labels_list}

Respond ONLY with valid JSON (include only labels that genuinely apply; at least one):
{{"labels": [{{"label": "<label>", "confidence": <float 0.0-1.0>}}, ...], "rationale": "<one sentence>"}}

Text:
<text>
{text}
</text>"""

CLASSIFY_USER_MULTI_FEW_SHOT = """Classify the following text. Assign ALL applicable labels from:
{labels_list}

Examples:
{examples_block}

Respond ONLY with valid JSON (include only labels that genuinely apply; at least one):
{{"labels": [{{"label": "<label>", "confidence": <float 0.0-1.0>}}, ...], "rationale": "<one sentence>"}}

Text:
<text>
{text}
</text>"""

# ---------------------------------------------------------------------------
# Sentiment analysis prompts
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM = (
    "You are a sentiment analysis engine. Analyze text for emotional tone and polarity. "
    "Be precise and nuanced. "
    "Always respond ONLY with valid JSON — no markdown, no extra prose."
)

SENTIMENT_USER = """Analyze the sentiment of the following text.

Respond ONLY with valid JSON:
{{"overall": "<positive|negative|neutral|mixed>", "score": <float -1.0 to 1.0>, "emotions": [{{"emotion": "<emotion>", "intensity": <float 0.0-1.0>}}], "rationale": "<one sentence>"}}

Score guide: -1.0 = extremely negative, 0.0 = neutral, 1.0 = extremely positive.
Include 1-3 dominant emotions (e.g. joy, anger, sadness, fear, surprise, disgust, trust, anticipation).

Text:
<text>
{text}
</text>"""

SENTIMENT_USER_ASPECTS = """Analyze the sentiment of the following text, both overall and for each specified aspect.

Aspects to analyze:
{aspects_list}

Respond ONLY with valid JSON:
{{"overall": "<positive|negative|neutral|mixed>", "score": <float -1.0 to 1.0>, "aspects": [{{"aspect": "<aspect>", "sentiment": "<positive|negative|neutral|mixed>", "score": <float -1.0 to 1.0>}}], "emotions": [{{"emotion": "<emotion>", "intensity": <float 0.0-1.0>}}], "rationale": "<one sentence>"}}

Score guide: -1.0 = extremely negative, 0.0 = neutral, 1.0 = extremely positive.
Include 1-3 dominant emotions.

Text:
<text>
{text}
</text>"""
