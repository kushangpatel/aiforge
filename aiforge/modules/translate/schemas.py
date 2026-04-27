"""Pydantic schemas for the Translator module."""

from __future__ import annotations

from typing import Literal

from aiforge.core.types import ModuleResponse

Formality = Literal["formal", "informal", "neutral"]


class TranslateResponse(ModuleResponse):
    """Response from Translator.translate() / .run()."""

    text: str = ""
    """The translated text."""

    source_language: str = ""
    """Detected or specified source language."""

    target_language: str = ""
    """Target language of the translation."""

    formality: Formality = "neutral"
    """Formality level used for translation."""

    def to_text(self) -> str:
        """Pipeline protocol — return the translated text for downstream modules."""
        return self.text
