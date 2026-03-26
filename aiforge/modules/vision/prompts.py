"""Prompt templates for the Vision module."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

VISION_SYSTEM_DESCRIBE = (
    "You are an expert image analyst. Describe the image in detail, including "
    "objects, people, text, colors, composition, and any notable features. "
    "Be thorough and precise."
)

VISION_SYSTEM_QA = (
    "You are an expert visual question-answering assistant. Analyze the image "
    "carefully and answer the user's question accurately based on what you see. "
    "If the answer is not visible in the image, say so."
)

VISION_SYSTEM_EXTRACT = (
    "You are a data extraction specialist. Extract all visible text, numbers, "
    "labels, and structured data from the image. Preserve the original formatting "
    "as closely as possible. Respond ONLY with the extracted content."
)

VISION_SYSTEM_CLASSIFY = (
    "You are an image classification engine. Categorize the image according to "
    "the provided labels. Be precise and consistent. "
    "Respond ONLY with valid JSON — no markdown, no extra prose."
)

# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

VISION_USER_DESCRIBE = "Describe this image in detail."

VISION_USER_DESCRIBE_FOCUS = "Describe this image, focusing on: {focus}"

VISION_USER_QA = "{question}"

VISION_USER_EXTRACT = "Extract all text and structured data from this image."

VISION_USER_CLASSIFY = """Classify this image into exactly one of these categories:
{labels_list}

Respond ONLY with valid JSON:
{{"label": "<chosen_label>", "confidence": <float 0.0-1.0>, "rationale": "<one sentence>"}}"""
