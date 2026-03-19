"""Unit tests for the Extractor module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from aiforge.modules.extract.module import (
    Extractor,
    _extract_json,
    _format_examples,
    _schema_to_description,
)
from aiforge.modules.extract.schemas import ExtractExample, ExtractResponse


# ---------------------------------------------------------------------------
# Test Pydantic models
# ---------------------------------------------------------------------------


class PersonSchema(BaseModel):
    name: str
    age: int


class InvoiceSchema(BaseModel):
    vendor: str
    total: float
    currency: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ext(
    output_schema: type[BaseModel] | dict | None = None,
    examples: list[ExtractExample] | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Extractor:
    """Instantiate an Extractor bypassing BaseModule provider initialisation."""
    e = object.__new__(Extractor)
    e.output_schema = output_schema
    e.examples = examples or []
    e.max_retries = max_retries
    e.temperature = temperature

    import structlog

    e.logger = structlog.get_logger()
    e._provider = None
    e._embedding_provider = None
    settings = MagicMock()
    settings.RATE_LIMIT_RPM = 500
    settings.RATE_LIMIT_TPM = 150_000
    e.settings = settings
    from aiforge.infra.cost_tracker import CostTracker

    e.cost_tracker = CostTracker.get_session()
    return e


def _fake_llm(text: str) -> MagicMock:
    """Create a fake CompletionResult."""
    r = MagicMock()
    r.text = text
    r.input_tokens = 20
    r.output_tokens = 10
    r.model = "gpt-4o-mini"
    return r


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_json_unchanged(self) -> None:
        raw = '{"name": "Alice"}'
        assert _extract_json(raw) == raw

    def test_strips_json_fence(self) -> None:
        raw = '```json\n{"name": "Alice"}\n```'
        assert _extract_json(raw) == '{"name": "Alice"}'

    def test_strips_plain_fence(self) -> None:
        raw = '```\n{"name": "Bob"}\n```'
        assert _extract_json(raw) == '{"name": "Bob"}'

    def test_whitespace_trimmed(self) -> None:
        raw = '   {"name": "Carol"}   '
        assert _extract_json(raw) == '{"name": "Carol"}'

    def test_multiline_json_in_fence(self) -> None:
        raw = '```json\n{\n  "name": "Dan",\n  "age": 25\n}\n```'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed == {"name": "Dan", "age": 25}

    def test_no_fence_with_extra_text_prefix(self) -> None:
        # LLM might return prose before JSON — _extract_json only strips fences
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw


# ---------------------------------------------------------------------------
# _schema_to_description helper
# ---------------------------------------------------------------------------


class TestSchemaToDescription:
    def test_pydantic_model_produces_json_schema(self) -> None:
        desc = _schema_to_description(PersonSchema)
        parsed = json.loads(desc)
        assert "properties" in parsed
        assert "name" in parsed["properties"]
        assert "age" in parsed["properties"]

    def test_dict_schema_passes_through(self) -> None:
        raw = {"type": "object", "properties": {"x": {"type": "number"}}}
        desc = _schema_to_description(raw)
        parsed = json.loads(desc)
        assert parsed == raw

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Pydantic BaseModel"):
            _schema_to_description("not a schema")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _format_examples helper
# ---------------------------------------------------------------------------


class TestFormatExamples:
    def test_empty_list_returns_empty_string(self) -> None:
        assert _format_examples([]) == ""

    def test_single_example_formatted(self) -> None:
        ex = ExtractExample(input="John is 30", output={"name": "John", "age": 30})
        result = _format_examples([ex])
        assert "Example 1:" in result
        assert "John is 30" in result
        assert '"name"' in result

    def test_multiple_examples_numbered(self) -> None:
        exs = [
            ExtractExample(input="A", output={"x": 1}),
            ExtractExample(input="B", output={"x": 2}),
        ]
        result = _format_examples(exs)
        assert "Example 1:" in result
        assert "Example 2:" in result

    def test_escapes_quotes(self) -> None:
        ex = ExtractExample(input='Say "hello"', output={"s": "hello"})
        result = _format_examples([ex])
        assert '\\"hello\\"' in result

    def test_escapes_backslashes(self) -> None:
        ex = ExtractExample(input="path\\to\\file", output={"p": "x"})
        result = _format_examples([ex])
        assert "path\\\\to\\\\file" in result


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestExtractorConstructor:
    def test_valid_pydantic_schema(self) -> None:
        e = Extractor(output_schema=PersonSchema)
        assert e.output_schema is PersonSchema
        assert e.examples == []
        assert e.max_retries == 3
        assert e.temperature == 0.0

    def test_valid_dict_schema(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        e = Extractor(output_schema=schema)
        assert e.output_schema == schema

    def test_no_schema_is_ok_at_construction(self) -> None:
        e = Extractor()
        assert e.output_schema is None

    def test_invalid_schema_type_raises_at_construction(self) -> None:
        with pytest.raises(TypeError):
            Extractor(output_schema="not a schema")  # type: ignore[arg-type]

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            Extractor(output_schema=PersonSchema, temperature=3.0)

    def test_examples_stored(self) -> None:
        exs = [ExtractExample(input="text", output={"name": "X", "age": 1})]
        e = Extractor(output_schema=PersonSchema, examples=exs)
        assert len(e.examples) == 1

    def test_max_retries_at_least_one(self) -> None:
        e = Extractor(output_schema=PersonSchema, max_retries=0)
        assert e.max_retries >= 1


# ---------------------------------------------------------------------------
# _validate static method
# ---------------------------------------------------------------------------


class TestValidateMethod:
    def test_valid_pydantic_returns_data(self) -> None:
        raw = '{"name": "Alice", "age": 30}'
        data, err = Extractor._validate(raw, PersonSchema)
        assert data == {"name": "Alice", "age": 30}
        assert err == ""

    def test_invalid_json_returns_error(self) -> None:
        data, err = Extractor._validate("{bad json", PersonSchema)
        assert data is None
        assert "Invalid JSON" in err

    def test_pydantic_validation_error_returns_message(self) -> None:
        raw = '{"name": "Alice"}'  # missing 'age'
        data, err = Extractor._validate(raw, PersonSchema)
        assert data is None
        assert err != ""

    def test_non_object_json_returns_error(self) -> None:
        raw = "[1, 2, 3]"
        data, err = Extractor._validate(raw, PersonSchema)
        assert data is None
        assert "JSON object" in err

    def test_dict_schema_accepts_any_dict(self) -> None:
        raw = '{"anything": "goes"}'
        schema = {"type": "object"}
        data, err = Extractor._validate(raw, schema)
        assert data == {"anything": "goes"}
        assert err == ""

    def test_fenced_json_extracted_before_validation(self) -> None:
        raw = '```json\n{"name": "Bob", "age": 25}\n```'
        data, err = Extractor._validate(raw, PersonSchema)
        assert data == {"name": "Bob", "age": 25}
        assert err == ""

    def test_extra_fields_accepted_by_pydantic(self) -> None:
        # Pydantic v2 ignores extra fields by default
        raw = '{"name": "Eve", "age": 28, "email": "eve@test.com"}'
        data, err = Extractor._validate(raw, PersonSchema)
        assert data is not None
        assert data["name"] == "Eve"

    def test_wrong_type_field_fails(self) -> None:
        raw = '{"name": "Alice", "age": "not a number"}'
        data, err = Extractor._validate(raw, PersonSchema)
        assert data is None
        assert err != ""


# ---------------------------------------------------------------------------
# extract() — sync, with mocked LLM
# ---------------------------------------------------------------------------


class TestExtract:
    def test_basic_extraction_first_attempt(self) -> None:
        e = _ext(output_schema=PersonSchema)
        valid_json = '{"name": "Alice", "age": 30}'
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid_json)):
            result = e.extract("Alice is 30 years old.")
        assert result.validation_passed is True
        assert result.data == {"name": "Alice", "age": 30}
        assert result.retry_count == 0
        assert result.cost >= 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.model == "gpt-4o-mini"

    def test_extraction_with_dict_schema(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        e = _ext(output_schema=schema)
        valid_json = '{"x": 42}'
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid_json)):
            result = e.extract("The value is 42.")
        assert result.validation_passed is True
        assert result.data == {"x": 42}

    def test_extraction_with_fenced_json(self) -> None:
        e = _ext(output_schema=PersonSchema)
        fenced = '```json\n{"name": "Bob", "age": 25}\n```'
        with patch.object(e, "_call_llm", return_value=_fake_llm(fenced)):
            result = e.extract("Bob is 25.")
        assert result.validation_passed is True
        assert result.data["name"] == "Bob"

    def test_retry_on_invalid_json_then_succeed(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=3)
        bad = _fake_llm("{bad json")
        good = _fake_llm('{"name": "Carol", "age": 40}')
        with patch.object(e, "_call_llm", side_effect=[bad, good]):
            result = e.extract("Carol is 40.")
        assert result.validation_passed is True
        assert result.retry_count == 1
        assert result.data["name"] == "Carol"

    def test_retry_on_validation_error_then_succeed(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=3)
        missing_age = _fake_llm('{"name": "Dan"}')
        complete = _fake_llm('{"name": "Dan", "age": 35}')
        with patch.object(e, "_call_llm", side_effect=[missing_age, complete]):
            result = e.extract("Dan is 35.")
        assert result.validation_passed is True
        assert result.retry_count == 1

    def test_all_retries_exhausted(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=2)
        bad = _fake_llm("{invalid")
        with patch.object(e, "_call_llm", return_value=bad):
            result = e.extract("Some text")
        assert result.validation_passed is False
        assert result.retry_count == 1  # max_retries - 1
        assert result.data == {}

    def test_empty_text_raises(self) -> None:
        e = _ext(output_schema=PersonSchema)
        with pytest.raises(ValueError, match="text must not be empty"):
            e.extract("")

    def test_whitespace_only_text_raises(self) -> None:
        e = _ext(output_schema=PersonSchema)
        with pytest.raises(ValueError, match="text must not be empty"):
            e.extract("   \n  ")

    def test_no_schema_raises(self) -> None:
        e = _ext(output_schema=None)
        with pytest.raises(ValueError, match="output_schema must be provided"):
            e.extract("Some text")

    def test_schema_override_at_call_time(self) -> None:
        e = _ext(output_schema=None)
        valid_json = '{"vendor": "Acme", "total": 100.0, "currency": "USD"}'
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid_json)):
            result = e.extract("Invoice from Acme for $100.", output_schema=InvoiceSchema)
        assert result.validation_passed is True
        assert result.data["vendor"] == "Acme"

    def test_examples_passed_at_call_time(self) -> None:
        e = _ext(output_schema=PersonSchema)
        valid_json = '{"name": "Eve", "age": 22}'
        call_examples = [ExtractExample(input="Frank is 50", output={"name": "Frank", "age": 50})]
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid_json)) as mock_call:
            result = e.extract("Eve is 22", examples=call_examples)
        assert result.validation_passed is True
        # Verify examples were included in the prompt
        prompt_messages = mock_call.call_args[0][0]
        user_content = prompt_messages[1]["content"]
        assert "Frank is 50" in user_content

    def test_cost_accumulates_across_retries(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=3)
        bad = _fake_llm("{bad")
        good = _fake_llm('{"name": "X", "age": 1}')
        with patch.object(e, "_call_llm", side_effect=[bad, bad, good]):
            result = e.extract("X is 1")
        # 3 calls total — cost should be > 0
        assert result.input_tokens == 60  # 20 * 3
        assert result.output_tokens == 30  # 10 * 3

    def test_latency_recorded(self) -> None:
        e = _ext(output_schema=PersonSchema)
        with patch.object(e, "_call_llm", return_value=_fake_llm('{"name": "A", "age": 1}')):
            result = e.extract("A is 1")
        assert result.latency_ms >= 0

    def test_complex_nested_schema(self) -> None:
        class LineItem(BaseModel):
            item: str
            qty: int

        class Order(BaseModel):
            customer: str
            items: list[LineItem]

        e = _ext(output_schema=Order)
        valid = '{"customer": "Acme", "items": [{"item": "Widget", "qty": 5}]}'
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid)):
            result = e.extract("Acme ordered 5 Widgets.")
        assert result.validation_passed is True
        assert result.data["customer"] == "Acme"
        assert len(result.data["items"]) == 1


# ---------------------------------------------------------------------------
# aextract() — async, with mocked LLM
# ---------------------------------------------------------------------------


class TestAextract:
    @pytest.mark.asyncio
    async def test_async_basic_extraction(self) -> None:
        e = _ext(output_schema=PersonSchema)
        valid_json = '{"name": "Alice", "age": 30}'
        with patch.object(e, "_acall_llm", new_callable=AsyncMock, return_value=_fake_llm(valid_json)):
            result = await e.aextract("Alice is 30 years old.")
        assert result.validation_passed is True
        assert result.data == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_async_retry_then_succeed(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=3)
        bad = _fake_llm("{invalid")
        good = _fake_llm('{"name": "Bob", "age": 25}')
        with patch.object(e, "_acall_llm", new_callable=AsyncMock, side_effect=[bad, good]):
            result = await e.aextract("Bob is 25.")
        assert result.validation_passed is True
        assert result.retry_count == 1

    @pytest.mark.asyncio
    async def test_async_all_retries_exhausted(self) -> None:
        e = _ext(output_schema=PersonSchema, max_retries=2)
        bad = _fake_llm("{bad")
        with patch.object(e, "_acall_llm", new_callable=AsyncMock, return_value=bad):
            result = await e.aextract("Some text")
        assert result.validation_passed is False
        assert result.data == {}

    @pytest.mark.asyncio
    async def test_async_empty_text_raises(self) -> None:
        e = _ext(output_schema=PersonSchema)
        with pytest.raises(ValueError, match="text must not be empty"):
            await e.aextract("")

    @pytest.mark.asyncio
    async def test_async_no_schema_raises(self) -> None:
        e = _ext(output_schema=None)
        with pytest.raises(ValueError, match="output_schema must be provided"):
            await e.aextract("Some text")


# ---------------------------------------------------------------------------
# run() / arun() — pipeline entry points
# ---------------------------------------------------------------------------


class TestPipelineInterface:
    def test_run_delegates_to_extract(self) -> None:
        e = _ext(output_schema=PersonSchema)
        valid_json = '{"name": "A", "age": 1}'
        with patch.object(e, "_call_llm", return_value=_fake_llm(valid_json)):
            result = e.run("A is 1")
        assert isinstance(result, ExtractResponse)
        assert result.validation_passed is True

    @pytest.mark.asyncio
    async def test_arun_delegates_to_aextract(self) -> None:
        e = _ext(output_schema=PersonSchema)
        valid_json = '{"name": "B", "age": 2}'
        with patch.object(e, "_acall_llm", new_callable=AsyncMock, return_value=_fake_llm(valid_json)):
            result = await e.arun("B is 2")
        assert isinstance(result, ExtractResponse)
        assert result.validation_passed is True


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_first_attempt_uses_extract_prompt(self) -> None:
        e = _ext(output_schema=PersonSchema)
        msgs = e._build_messages(
            text="Some text",
            schema_desc='{"properties": {}}',
            examples_section="",
            attempt=0,
            previous_output="",
            error_msg="",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Extract structured data" in msgs[1]["content"]

    def test_retry_attempt_uses_retry_prompt(self) -> None:
        e = _ext(output_schema=PersonSchema)
        msgs = e._build_messages(
            text="Some text",
            schema_desc='{"properties": {}}',
            examples_section="",
            attempt=1,
            previous_output='{"bad": true}',
            error_msg="Missing field 'name'",
        )
        assert "previous extraction attempt was invalid" in msgs[1]["content"]
        assert "Missing field 'name'" in msgs[1]["content"]
        assert '{"bad": true}' in msgs[1]["content"]

    def test_examples_included_in_first_attempt(self) -> None:
        e = _ext(output_schema=PersonSchema)
        examples_section = "\nExamples:\n  Example 1:\n    Input: test\n    Output: {}"
        msgs = e._build_messages(
            text="Some text",
            schema_desc="{}",
            examples_section=examples_section,
            attempt=0,
            previous_output="",
            error_msg="",
        )
        assert "Example 1:" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# ExtractResponse — schema & to_text
# ---------------------------------------------------------------------------


class TestExtractResponse:
    def test_to_text_with_data(self) -> None:
        r = ExtractResponse(
            data={"name": "Alice", "age": 30},
            raw_json='{"name":"Alice","age":30}',
            validation_passed=True,
        )
        text = r.to_text()
        parsed = json.loads(text)
        assert parsed["name"] == "Alice"

    def test_to_text_empty_data(self) -> None:
        r = ExtractResponse(data={}, raw_json="", validation_passed=False)
        assert r.to_text() == "{}"

    def test_default_values(self) -> None:
        r = ExtractResponse()
        assert r.data == {}
        assert r.raw_json == ""
        assert r.validation_passed is False
        assert r.retry_count == 0
        assert r.cost == 0.0

    def test_inherits_module_response_fields(self) -> None:
        r = ExtractResponse(
            data={"x": 1},
            raw_json='{"x": 1}',
            validation_passed=True,
            cost=0.001,
            input_tokens=20,
            output_tokens=10,
            model="gpt-4o-mini",
            latency_ms=150.0,
        )
        assert r.cost == 0.001
        assert r.model == "gpt-4o-mini"
        assert r.latency_ms == 150.0


# ---------------------------------------------------------------------------
# ExtractExample — schema
# ---------------------------------------------------------------------------


class TestExtractExample:
    def test_valid_example(self) -> None:
        ex = ExtractExample(input="John is 30", output={"name": "John", "age": 30})
        assert ex.input == "John is 30"
        assert ex.output == {"name": "John", "age": 30}

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation
            ExtractExample(input="", output={"x": 1})

    def test_empty_output_ok(self) -> None:
        ex = ExtractExample(input="text", output={})
        assert ex.output == {}
