"""Prompt templates for the Translator module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = (
    "You are an expert multilingual translator. "
    "Produce accurate, natural-sounding translations that preserve the meaning, "
    "tone, and nuance of the original text. "
    "Respond ONLY with the translated text — no preamble, no commentary."
)

# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

TRANSLATE_USER = """Translate the following text from {source_language} to {target_language}.

Formality: {formality}
{domain_instruction}

Text:
<text>
{text}
</text>

Translation:"""

TRANSLATE_USER_AUTO_DETECT = """Translate the following text to {target_language}.
Automatically detect the source language.

Formality: {formality}
{domain_instruction}

Text:
<text>
{text}
</text>

Translation:"""

# ---------------------------------------------------------------------------
# Language detection (used internally when source_language="auto")
# ---------------------------------------------------------------------------

DETECT_LANGUAGE = """Identify the language of the following text.
Respond with ONLY the language name in English (e.g. "English", "Spanish", "Japanese").

Text:
<text>
{text}
</text>

Language:"""
