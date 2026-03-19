"""Prompt templates for the Extractor module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You are a structured-data extraction engine. "
    "Given source text and a JSON Schema, extract the requested fields. "
    "Always respond ONLY with valid JSON — no markdown, no extra prose."
)

# ---------------------------------------------------------------------------
# First-attempt user prompt
# ---------------------------------------------------------------------------

EXTRACT_USER = """Extract structured data from the text below.
Return ONLY valid JSON matching this schema exactly.

Schema:
{schema_description}
{examples_section}
Text:
<text>
{text}
</text>

JSON Output:"""

# ---------------------------------------------------------------------------
# Retry prompt (sent when previous attempt failed validation)
# ---------------------------------------------------------------------------

EXTRACT_RETRY = """Your previous extraction attempt was invalid.

Error:
{error_message}

Please fix the JSON output to match the required schema exactly.

Schema:
{schema_description}

Original text:
<text>
{text}
</text>

Previous (invalid) output:
{previous_output}

Corrected JSON Output:"""
