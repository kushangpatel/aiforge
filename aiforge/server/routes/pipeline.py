"""Pipeline endpoint (/pipeline)."""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class PipelineStepDef(BaseModel):
    """Definition for a single pipeline step."""
    name: str
    module: str
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineRunRequest(BaseModel):
    steps: list[PipelineStepDef]
    text: str
    context: dict[str, Any] | None = None


@router.post("")
async def run_pipeline(body: PipelineRunRequest) -> dict[str, Any]:
    """Build and execute a pipeline from step definitions."""
    from aiforge.pipeline.pipeline import Pipeline
    from aiforge.registry import get_module

    if not body.steps:
        raise HTTPException(status_code=422, detail="steps must not be empty")

    try:
        built_steps: list[tuple[str, Any]] = []
        for step_def in body.steps:
            module_cls = get_module(step_def.module)
            module_instance = module_cls(**step_def.config)
            built_steps.append((step_def.name, module_instance))

        pipe = Pipeline(built_steps)
        result = pipe.run(body.text, context=body.context)
        return result.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
