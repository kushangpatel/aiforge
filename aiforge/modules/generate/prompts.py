"""Prompt templates for the Generator module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = (
    "You are a versatile content generator. "
    "Produce high-quality text that precisely follows the user's instructions. "
    "Respond ONLY with the generated content — no preamble, no commentary."
)

# ---------------------------------------------------------------------------
# Guardrail retry feedback
# ---------------------------------------------------------------------------

GENERATE_RETRY = """Your previous response violated the following rules: {violations}.

Previous response:
<previous>
{previous_text}
</previous>

Original request:
<request>
{original_prompt}
</request>

Please regenerate the content while strictly avoiding these violations."""

# ---------------------------------------------------------------------------
# Professional tone check
# ---------------------------------------------------------------------------

PROFESSIONAL_TONE_CHECK = """Evaluate whether the following text maintains a professional tone.
Consider: formal language, no slang, no offensive language, appropriate for business communication.

Text:
<text>
{text}
</text>

Respond with exactly one word: YES or NO."""
