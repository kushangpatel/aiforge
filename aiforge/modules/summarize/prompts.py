"""Prompt templates for the Summarizer module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = (
    "You are an expert text summariser. "
    "Produce clear, accurate summaries that preserve the key facts and tone "
    "of the original text. Respond ONLY with the summary — no preamble."
)

# ---------------------------------------------------------------------------
# Style-specific user prompts
# ---------------------------------------------------------------------------

SUMMARIZE_CONCISE = """Summarize the following text in {max_length} words or fewer.
Focus on the key points and main conclusions.

Text:
<text>
{text}
</text>

Summary:"""

SUMMARIZE_DETAILED = """Provide a detailed summary of the following text.
Include all major points, supporting evidence, and conclusions.
Target length: {max_length} words.

Text:
<text>
{text}
</text>

Detailed Summary:"""

SUMMARIZE_BULLET_POINTS = """Summarize the following text as bullet points.
Use {max_length} words or fewer total. Each bullet should be one key point.

Text:
<text>
{text}
</text>

Bullet Point Summary:"""

SUMMARIZE_EXECUTIVE = """Write an executive summary of the following text.
Focus on decisions, actions, and outcomes. {max_length} words or fewer.

Text:
<text>
{text}
</text>

Executive Summary:"""

# ---------------------------------------------------------------------------
# Map-reduce prompts
# ---------------------------------------------------------------------------

SUMMARIZE_MAP = """Summarize this section concisely, preserving all key facts:

<chunk>
{chunk}
</chunk>

Section Summary:"""

SUMMARIZE_REDUCE = """Combine these section summaries into one coherent summary
of {max_length} words or fewer. Remove any redundancy.

Section Summaries:
<summaries>
{summaries}
</summaries>

Combined Summary:"""
