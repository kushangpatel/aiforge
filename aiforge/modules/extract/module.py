"""Extractor module — structured data extraction with schema validation + retry."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aiforge.core.types import BaseModule
from aiforge.modules.extract.prompts import EXTRACT_RETRY, EXTRACT_SYSTEM, EXTRACT_USER
from aiforge.modules.extract.schemas import ExtractExample, ExtractResponse

_MAX_RETRIES = 3


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


def _schema_to_description(schema: type[BaseModel] | dict[str, Any]) -> str:
    """Convert a Pydantic model or a dict JSON schema into a readable string."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return json.dumps(schema.model_json_schema(), indent=2)
    if isinstance(schema, dict):
        return json.dumps(schema, indent=2)
    raise TypeError(f"output_schema must be a Pydantic BaseModel subclass or dict, got {type(schema)}")


def _format_examples(examples: list[ExtractExample]) -> str:
    """Format few-shot examples for prompt insertion."""
    if not examples:
        return ""
    lines = ["\nExamples:"]
    for i, ex in enumerate(examples, 1):
        safe_input = ex.input.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f"  Example {i}:")
        lines.append(f'    Input: "{safe_input}"')
        lines.append(f"    Output: {json.dumps(ex.output)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class Extractor(BaseModule):
    """Structured data extraction backed by an LLM with schema validation + retry.

    Accepts a Pydantic model *or* a raw JSON-schema dict.  On validation
    failure the error message is fed back to the LLM for self-correction
    (up to ``max_retries`` attempts).
    """

    def __init__(
        self,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        examples: list[ExtractExample] | None = None,
        max_retries: int = _MAX_RETRIES,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if output_schema is not None:
            # Validate early so callers get an immediate error
            _schema_to_description(output_schema)
        self.output_schema: type[BaseModel] | dict[str, Any] | None = output_schema
        self.examples: list[ExtractExample] = examples or []
        self.max_retries: int = max(1, max_retries)
        self.temperature: float = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        examples: list[ExtractExample] | None = None,
    ) -> ExtractResponse:
        """Extract structured data from *text* synchronously."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        schema = output_schema or self.output_schema
        if schema is None:
            raise ValueError("output_schema must be provided either in constructor or extract() call")
        exs = examples if examples is not None else self.examples
        t0 = time.monotonic()
        return self._extract_loop(text, schema, exs, t0)

    async def aextract(
        self,
        text: str,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
        examples: list[ExtractExample] | None = None,
    ) -> ExtractResponse:
        """Async version of extract()."""
        if not text or not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        schema = output_schema or self.output_schema
        if schema is None:
            raise ValueError("output_schema must be provided either in constructor or extract() call")
        exs = examples if examples is not None else self.examples
        t0 = time.monotonic()
        return await self._aextract_loop(text, schema, exs, t0)

    def run(self, text: str, **kwargs: Any) -> ExtractResponse:
        """Pipeline entry point (sync)."""
        return self.extract(text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> ExtractResponse:
        """Pipeline entry point (async)."""
        return await self.aextract(text, **kwargs)

    # ------------------------------------------------------------------
    # Retry loops
    # ------------------------------------------------------------------

    def _extract_loop(
        self,
        text: str,
        schema: type[BaseModel] | dict[str, Any],
        examples: list[ExtractExample],
        t0: float,
    ) -> ExtractResponse:
        """Synchronous extraction with retry-on-validation-failure."""
        schema_desc = _schema_to_description(schema)
        examples_section = _format_examples(examples)
        total_in = total_out = 0
        total_cost = 0.0
        raw_json = ""
        error_msg = ""

        for attempt in range(self.max_retries):
            messages = self._build_messages(
                text, schema_desc, examples_section, attempt, raw_json, error_msg,
            )
            llm = self._call_llm(messages, temperature=self.temperature)
            from aiforge.core.pricing import calculate_cost
            cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += cost
            raw_json = llm.text

            parsed, err = self._validate(raw_json, schema)
            if parsed is not None:
                return ExtractResponse(
                    data=parsed,
                    raw_json=raw_json,
                    validation_passed=True,
                    retry_count=attempt,
                    cost=total_cost,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    model=llm.model,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            error_msg = err
            self.logger.warning(
                "extract.validation_failed",
                attempt=attempt + 1,
                error=error_msg[:200],
            )

        # All retries exhausted
        return ExtractResponse(
            data={},
            raw_json=raw_json,
            validation_passed=False,
            retry_count=self.max_retries - 1,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def _aextract_loop(
        self,
        text: str,
        schema: type[BaseModel] | dict[str, Any],
        examples: list[ExtractExample],
        t0: float,
    ) -> ExtractResponse:
        """Async extraction with retry-on-validation-failure."""
        schema_desc = _schema_to_description(schema)
        examples_section = _format_examples(examples)
        total_in = total_out = 0
        total_cost = 0.0
        raw_json = ""
        error_msg = ""

        for attempt in range(self.max_retries):
            messages = self._build_messages(
                text, schema_desc, examples_section, attempt, raw_json, error_msg,
            )
            llm = await self._acall_llm(messages, temperature=self.temperature)
            from aiforge.core.pricing import calculate_cost
            cost = calculate_cost(llm.model, llm.input_tokens, llm.output_tokens)
            total_in += llm.input_tokens
            total_out += llm.output_tokens
            total_cost += cost
            raw_json = llm.text

            parsed, err = self._validate(raw_json, schema)
            if parsed is not None:
                return ExtractResponse(
                    data=parsed,
                    raw_json=raw_json,
                    validation_passed=True,
                    retry_count=attempt,
                    cost=total_cost,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    model=llm.model,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            error_msg = err
            self.logger.warning(
                "extract.validation_failed",
                attempt=attempt + 1,
                error=error_msg[:200],
            )

        return ExtractResponse(
            data={},
            raw_json=raw_json,
            validation_passed=False,
            retry_count=self.max_retries - 1,
            cost=total_cost,
            input_tokens=total_in,
            output_tokens=total_out,
            model=llm.model,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        text: str,
        schema_desc: str,
        examples_section: str,
        attempt: int,
        previous_output: str,
        error_msg: str,
    ) -> list[dict[str, str]]:
        """Construct system + user messages for an extraction attempt."""
        if attempt == 0:
            user_content = EXTRACT_USER.format(
                schema_description=schema_desc,
                examples_section=examples_section,
                text=text,
            )
        else:
            user_content = EXTRACT_RETRY.format(
                error_message=error_msg,
                schema_description=schema_desc,
                text=text,
                previous_output=previous_output,
            )
        return [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _validate(
        raw_json: str,
        schema: type[BaseModel] | dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        """Parse LLM output and validate against schema.

        Returns ``(data_dict, "")`` on success or ``(None, error_message)`` on failure.
        """
        try:
            parsed = json.loads(_extract_json(raw_json))
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc}"

        if not isinstance(parsed, dict):
            return None, f"Expected a JSON object, got {type(parsed).__name__}"

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                validated = schema(**parsed)
                return validated.model_dump(), ""
            except PydanticValidationError as exc:
                return None, str(exc)

        # Raw dict schema — no further validation, accept the parsed dict
        return parsed, ""
