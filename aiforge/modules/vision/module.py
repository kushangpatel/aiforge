"""
Vision module — image analysis, description, and visual Q&A.

Requires a vision-capable model (GPT-4o, Claude 3.5 Sonnet, etc.).

Usage::

    from aiforge import Vision

    # Describe an image
    v = Vision()
    result = v.analyze("product_photo.jpg")
    print(result.text)

    # Visual Q&A
    v = Vision(task="qa")
    result = v.analyze("chart.png", question="What is the highest value?")
    print(result.text)

    # Extract text from an image (OCR-like)
    v = Vision(task="extract")
    result = v.analyze("screenshot.png")
    print(result.text)

    # Classify an image
    v = Vision(task="classify")
    result = v.analyze("photo.jpg", labels=["landscape", "portrait", "product", "document"])
    print(result.text)
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from aiforge.core.prompt_override import resolve_prompt
from aiforge.core.types import BaseModule
from aiforge.modules.vision.prompts import (
    VISION_SYSTEM_CLASSIFY,
    VISION_SYSTEM_DESCRIBE,
    VISION_SYSTEM_EXTRACT,
    VISION_SYSTEM_QA,
    VISION_USER_CLASSIFY,
    VISION_USER_DESCRIBE,
    VISION_USER_DESCRIBE_FOCUS,
    VISION_USER_EXTRACT,
    VISION_USER_QA,
)
from aiforge.modules.vision.schemas import VisionResponse, VisionTask

_SUPPORTED_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _encode_image(source: str | Path) -> tuple[str, str]:
    """Read an image file and return ``(base64_data, mime_type)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in _SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type {mime!r}. Supported: {sorted(_SUPPORTED_MIME_TYPES)}"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return data, mime


def _build_image_content(
    source: str | Path,
    user_text: str,
) -> list[dict[str, Any]]:
    """Build a multimodal message content list with image + text."""
    source_str = str(source)
    # URL-based image
    if source_str.startswith(("http://", "https://")):
        return [
            {"type": "image_url", "image_url": {"url": source_str}},
            {"type": "text", "text": user_text},
        ]
    # File-based image — encode as data URI
    b64_data, mime = _encode_image(source)
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64_data}"},
        },
        {"type": "text", "text": user_text},
    ]


class Vision(BaseModule):
    """Image analysis module backed by a vision-capable LLM.

    Args:
        task:        Analysis mode — ``"describe"``, ``"qa"``, ``"extract"``,
                     or ``"classify"``.
        temperature: Sampling temperature for the LLM call.
        max_tokens:  Maximum tokens for the LLM response.
        **kwargs:    Forwarded to :class:`~aiforge.core.types.BaseModule`.
    """

    def __init__(
        self,
        task: VisionTask = "describe",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task: VisionTask = task
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt_override = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        source: str | Path,
        *,
        question: str | None = None,
        focus: str | None = None,
        labels: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> VisionResponse:
        """Analyze an image synchronously.

        Args:
            source:  Path to a local image file or an HTTP(S) URL.
            question: Question for ``task="qa"``.
            focus:   Focus area for ``task="describe"`` (e.g. ``"text"``).
            labels:  Classification labels for ``task="classify"``.
            system_prompt: Override the system prompt.
        """
        t0 = time.monotonic()
        system, user_text = self._resolve_prompts(
            question=question,
            focus=focus,
            labels=labels,
            system_prompt=system_prompt,
        )
        content = _build_image_content(source, user_text)

        llm = self._call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)

        return VisionResponse(
            text=llm.text.strip(),
            task=self.task,
            image_source=str(source),
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def aanalyze(
        self,
        source: str | Path,
        *,
        question: str | None = None,
        focus: str | None = None,
        labels: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> VisionResponse:
        """Async version of :meth:`analyze`."""
        t0 = time.monotonic()
        system, user_text = self._resolve_prompts(
            question=question,
            focus=focus,
            labels=labels,
            system_prompt=system_prompt,
        )
        content = _build_image_content(source, user_text)

        llm = await self._acall_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)

        return VisionResponse(
            text=llm.text.strip(),
            task=self.task,
            image_source=str(source),
            cost=cost,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # Pipeline interface -----------------------------------------------

    def run(self, text: str, **kwargs: Any) -> VisionResponse:
        """Pipeline entry point (sync).

        *text* is treated as the image source (path or URL).
        """
        return self.analyze(
            text,
            question=kwargs.get("question"),
            focus=kwargs.get("focus"),
            labels=kwargs.get("labels"),
            system_prompt=kwargs.get("system_prompt"),
        )

    async def arun(self, text: str, **kwargs: Any) -> VisionResponse:
        """Pipeline entry point (async)."""
        return await self.aanalyze(
            text,
            question=kwargs.get("question"),
            focus=kwargs.get("focus"),
            labels=kwargs.get("labels"),
            system_prompt=kwargs.get("system_prompt"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_prompts(
        self,
        *,
        question: str | None,
        focus: str | None,
        labels: list[str] | None,
        system_prompt: str | None,
    ) -> tuple[str, str]:
        """Return ``(system_prompt, user_text)`` based on the configured task."""
        task_systems = {
            "describe": VISION_SYSTEM_DESCRIBE,
            "qa": VISION_SYSTEM_QA,
            "extract": VISION_SYSTEM_EXTRACT,
            "classify": VISION_SYSTEM_CLASSIFY,
        }
        default_system = task_systems.get(self.task, VISION_SYSTEM_DESCRIBE)
        resolved_system = resolve_prompt(
            call_override=system_prompt,
            instance_override=self._system_prompt_override,
            config_override=self.settings.CUSTOM_PROMPTS.get("vision", {}).get("system"),
            default=default_system,
        )

        if self.task == "qa":
            if not question:
                raise ValueError("question is required for task='qa'")
            user_text = VISION_USER_QA.format(question=question)
        elif self.task == "extract":
            user_text = VISION_USER_EXTRACT
        elif self.task == "classify":
            if not labels:
                raise ValueError("labels are required for task='classify'")
            labels_list = "\n".join(f"  - {lbl}" for lbl in labels)
            user_text = VISION_USER_CLASSIFY.format(labels_list=labels_list)
        elif focus:
            user_text = VISION_USER_DESCRIBE_FOCUS.format(focus=focus)
        else:
            user_text = VISION_USER_DESCRIBE

        return resolved_system, user_text
