"""Pipeline request and result schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_serializer

from aiforge.core.types import ModuleResponse


class PipelineRequest(BaseModel):
    """Input to a pipeline execution."""

    text: str
    context: dict[str, Any] | None = None  # Extra metadata passed to each step


class PipelineResult(BaseModel):
    """Aggregated result from running all pipeline steps."""

    steps: dict[str, ModuleResponse]
    total_cost: float = 0.0
    total_tokens: int = 0
    halted: bool = False  # True if a halt_on_flag condition triggered early exit
    halted_at: str | None = None  # Name of step that triggered the halt

    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        """Custom serializer so concrete ModuleResponse subclasses round-trip correctly."""
        base = handler(self)
        # Re-serialize steps using each subclass's own model_dump so extra fields
        # (e.g. .summary, .label) are preserved in JSON output.
        base["steps"] = {name: resp.model_dump() for name, resp in self.steps.items()}
        return base

    @property
    def last(self) -> ModuleResponse | None:
        """Return the final step's response."""
        if not self.steps:
            return None
        return list(self.steps.values())[-1]

    def get(self, step_name: str) -> ModuleResponse | None:
        """Get a step result by name."""
        return self.steps.get(step_name)
