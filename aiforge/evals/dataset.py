"""Evaluation datasets: test cases loaded from Python, JSON, JSONL, CSV or YAML."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

_CASE_FIELDS = {"id", "input", "expected", "kwargs", "metadata"}


class EvalCase(BaseModel):
    """One test case: an input, an optional expected value and per-case call kwargs."""

    id: str = ""
    """Stable identifier used to match cases across reports (defaults to the row index)."""

    input: Any
    """Value passed to the target (text for AI Forge modules)."""

    expected: Any = None
    """Reference value that scorers compare the output against."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra keyword arguments forwarded to the target for this case only."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Free-form tags carried through to the report."""

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, value: Any) -> str:
        return "" if value is None else str(value)


class Dataset:
    """An ordered collection of :class:`EvalCase` objects.

    Example::

        ds = Dataset.load("tests/evals/tickets.jsonl")
        ds = Dataset.from_list([{"input": "I love it", "expected": "positive"}])
    """

    def __init__(self, cases: Iterable[EvalCase | dict[str, Any]], name: str = "") -> None:
        self.name = name
        self.cases: list[EvalCase] = [_to_case(c, i) for i, c in enumerate(cases)]
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate case ids in dataset: {dupes}")

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        return self.cases[index]

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, rows: Iterable[EvalCase | dict[str, Any]], name: str = "") -> Dataset:
        """Build a dataset from dicts (``input``, ``expected``, ``kwargs``, ``metadata``, ``id``)."""
        return cls(rows, name=name)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        """Load one JSON object per line; blank lines are skipped."""
        p = Path(path)
        rows: list[dict[str, Any]] = []
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from exc
        return cls(rows, name=p.stem)

    @classmethod
    def from_json(cls, path: str | Path) -> Dataset:
        """Load a JSON list of cases, or an object with a ``cases`` list."""
        p = Path(path)
        return cls(_rows_from_document(json.loads(p.read_text(encoding="utf-8")), p), name=p.stem)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Dataset:
        """Load a YAML list of cases, or a mapping with a ``cases`` list."""
        import yaml

        p = Path(path)
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(_rows_from_document(doc, p), name=p.stem)

    @classmethod
    def from_csv(cls, path: str | Path) -> Dataset:
        """Load a CSV with an ``input`` column.

        ``expected`` is optional; values starting with ``[`` or ``{`` are parsed
        as JSON. ``kwargs`` and ``metadata`` columns hold JSON objects. Any other
        column is copied into the case's ``metadata``.
        """
        p = Path(path)
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "input" not in reader.fieldnames:
                raise ValueError(f"{p}: CSV must have an 'input' column")
            for lineno, raw in enumerate(reader, start=2):
                row: dict[str, Any] = {"input": raw["input"]}
                metadata: dict[str, Any] = {}
                for key, value in raw.items():
                    if key in (None, "input") or value is None:
                        continue
                    if key == "expected":
                        if value != "":
                            row["expected"] = _maybe_json(value)
                    elif key in ("kwargs", "metadata"):
                        if value.strip():
                            try:
                                parsed = json.loads(value)
                            except json.JSONDecodeError as exc:
                                raise ValueError(
                                    f"{p}:{lineno}: column {key!r} must be a JSON object"
                                ) from exc
                            if key == "kwargs":
                                row["kwargs"] = parsed
                            else:
                                metadata.update(parsed)
                    elif key == "id":
                        if value:
                            row["id"] = value
                    else:
                        metadata[key] = value
                row["metadata"] = metadata
                rows.append(row)
        return cls(rows, name=p.stem)

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load a dataset, picking the parser from the file extension."""
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".jsonl":
            return cls.from_jsonl(p)
        if suffix == ".json":
            return cls.from_json(p)
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(p)
        if suffix == ".csv":
            return cls.from_csv(p)
        raise ValueError(
            f"Unsupported dataset format {suffix!r} (use .jsonl, .json, .yaml or .csv)"
        )


def _maybe_json(value: str) -> Any:
    stripped = value.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _rows_from_document(doc: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and "cases" in doc:
        doc = doc["cases"]
    if not isinstance(doc, list):
        raise ValueError(f"{path}: expected a list of cases or a mapping with a 'cases' list")
    return doc


def _to_case(row: EvalCase | dict[str, Any], index: int) -> EvalCase:
    if isinstance(row, EvalCase):
        case = row
    elif isinstance(row, dict):
        if "input" not in row:
            raise ValueError(f"Case {index} has no 'input' field")
        unknown = set(row) - _CASE_FIELDS
        if unknown:
            raise ValueError(f"Case {index} has unknown fields {sorted(unknown)}")
        case = EvalCase(**row)
    else:
        raise TypeError(f"Case {index} must be a dict or EvalCase, got {type(row).__name__}")
    if not case.id:
        case = case.model_copy(update={"id": str(index)})
    return case
