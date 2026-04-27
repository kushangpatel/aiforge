"""Prompt override resolution for all modules."""

from __future__ import annotations


def resolve_prompt(
    *,
    call_override: str | None = None,
    instance_override: str | None = None,
    config_override: str | None = None,
    default: str,
) -> str:
    """Return the highest-priority prompt that is non-None and non-empty.

    Resolution order:
        1. call_override      (per-call ``system_prompt=`` argument)
        2. instance_override   (constructor ``system_prompt=`` argument)
        3. config_override     (``aiforge.yaml`` ``custom_prompts:`` section)
        4. default             (module's ``prompts.py`` constant)
    """
    for candidate in (call_override, instance_override, config_override):
        if candidate is not None and candidate.strip():
            return candidate
    return default
