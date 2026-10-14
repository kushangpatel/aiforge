"""
Pipeline composition engine.

Chains multiple AIForge modules sequentially, with optional conditional
routing via :class:`Router` steps.

Example::

    from aiforge import Pipeline, Extractor, Classifier, Summarizer

    pipe = Pipeline([
        ("extract",   Extractor(output_schema=TicketSchema)),
        ("classify",  Classifier(labels=["critical", "high", "medium", "low"])),
        ("summarize", Summarizer(style="concise")),
    ])
    result = pipe.run("Long customer complaint...")
    print(result["classify"].label)
    print(result["summarize"].summary)
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from aiforge.pipeline.schemas import PipelineResult

if TYPE_CHECKING:
    from aiforge.core.types import BaseModule, ModuleResponse
    from aiforge.pipeline.router import Router


class Pipeline:
    """Sequential pipeline with optional conditional routing.

    Steps are a list of ``(name, module_or_router)`` tuples.  Each step
    receives the ``to_text()`` output of the previous step as its input.
    """

    def __init__(
        self,
        steps: list[tuple[str, BaseModule | Router]],
        *,
        halt_on_flag: bool = True,
    ) -> None:
        """
        Args:
            steps:        List of ``(step_name, module)`` pairs.
            halt_on_flag: If ``True`` (default), stop execution when a
                          :class:`Router` branch returns ``None``.
        """
        self._steps = steps
        self._halt_on_flag = halt_on_flag
        # Validate no duplicate step names
        names = [name for name, _ in steps]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            raise ValueError(f"Duplicate step names: {sorted(set(dupes))}")

    def run(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline synchronously.

        Args:
            text:    Initial input text.
            context: Optional metadata passed through (not used by modules
                     directly, but available for custom routing logic).

        Returns:
            :class:`PipelineResult` with per-step responses and aggregates.
        """
        from aiforge.pipeline.router import Router

        results: dict[str, ModuleResponse] = {}
        current_input = text
        previous_response: ModuleResponse | None = None
        halted = False
        halted_at: str | None = None

        for name, step in self._steps:
            if isinstance(step, Router):
                # Routing step — read label from previous response
                if previous_response is None or not hasattr(previous_response, "label"):
                    from aiforge.core.types import PipelineError

                    raise PipelineError(
                        f"Router step {name!r} requires the previous step to return "
                        "a response with a .label attribute (e.g. ClassifyResponse)."
                    )

                label = str(previous_response.label)
                response, was_halted = step.resolve(label, current_input)

                if was_halted and self._halt_on_flag:
                    halted = True
                    halted_at = name
                    break

                if response is not None:
                    results[name] = response
                    previous_response = response
                    try:
                        current_input = response.to_text()
                    except NotImplementedError:
                        current_input = current_input  # keep previous
            else:
                # Standard module step
                module: BaseModule = step
                response = module.run(current_input, **(context or {}))
                results[name] = response
                previous_response = response
                with contextlib.suppress(NotImplementedError):
                    current_input = response.to_text()

        total_cost = sum(r.cost for r in results.values())
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results.values())

        return PipelineResult(
            steps=results,
            total_cost=total_cost,
            total_tokens=total_tokens,
            halted=halted,
            halted_at=halted_at,
        )

    async def arun(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute the pipeline asynchronously."""
        from aiforge.pipeline.router import Router

        results: dict[str, ModuleResponse] = {}
        current_input = text
        previous_response: ModuleResponse | None = None
        halted = False
        halted_at: str | None = None

        for name, step in self._steps:
            if isinstance(step, Router):
                if previous_response is None or not hasattr(previous_response, "label"):
                    from aiforge.core.types import PipelineError

                    raise PipelineError(
                        f"Router step {name!r} requires previous step to have .label"
                    )

                label = str(previous_response.label)
                response, was_halted = await step.resolve_async(label, current_input)

                if was_halted and self._halt_on_flag:
                    halted = True
                    halted_at = name
                    break

                if response is not None:
                    results[name] = response
                    previous_response = response
                    with contextlib.suppress(NotImplementedError):
                        current_input = response.to_text()
            else:
                module: BaseModule = step
                response = await module.arun(current_input, **(context or {}))
                results[name] = response
                previous_response = response
                with contextlib.suppress(NotImplementedError):
                    current_input = response.to_text()

        total_cost = sum(r.cost for r in results.values())
        total_tokens = sum(r.input_tokens + r.output_tokens for r in results.values())

        return PipelineResult(
            steps=results,
            total_cost=total_cost,
            total_tokens=total_tokens,
            halted=halted,
            halted_at=halted_at,
        )
