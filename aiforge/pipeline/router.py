"""
Conditional router for pipeline branching.

A ``Router`` step reads the ``label`` attribute of the previous step's
response and dispatches to the appropriate module branch::

    pipe = Pipeline([
        ("classify", Classifier(labels=["support", "sales", "spam"])),
        ("route", Router({
            "support": Summarizer(style="bullet_points"),
            "sales":   Generator(template="..."),
            "spam":    None,   # drop — pipeline halts here
        })),
    ])
"""

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiforge.core.types import BaseModule, ModuleResponse


class Router:
    """Conditional routing step.

    Routes the incoming text to one of the modules in *branches* based on
    the ``label`` attribute of the previous step's result.  If the label
    does not match any branch, a :exc:`KeyError` is raised unless a
    ``default`` branch is provided.

    Setting a branch to ``None`` drops the input and halts the pipeline.
    """

    def __init__(
        self,
        branches: dict[str, BaseModule | None],
        *,
        default: BaseModule | EllipsisType | None = ...,
    ) -> None:
        """
        Args:
            branches: Mapping of label → module (or ``None`` to halt).
            default:  Module to use if no branch matches.  If not provided
                      and the label doesn’t match, raises ``KeyError``.
        """
        self._branches = branches
        self._has_default = not isinstance(default, EllipsisType)
        self._default: BaseModule | None = (
            None if isinstance(default, EllipsisType) else default
        )

    def resolve(
        self,
        label: str,
        text: str,
        **kwargs: Any,
    ) -> tuple[ModuleResponse | None, bool]:
        """Resolve *label* to a branch and execute it.

        Returns:
            ``(response, halted)`` where *halted* is True when the branch
            is ``None`` (drop case).

        Raises:
            KeyError: If *label* is not in branches and no default is set.
        """
        module = self._branches.get(label)
        if module is None and label not in self._branches:
            # Label not in branches at all
            if self._has_default:
                module = self._default
            else:
                raise KeyError(
                    f"Router has no branch for label {label!r}. "
                    f"Available: {list(self._branches.keys())}"
                )

        if module is None:
            # Explicit None branch — halt pipeline
            return None, True

        return module.run(text, **kwargs), False

    async def resolve_async(
        self,
        label: str,
        text: str,
        **kwargs: Any,
    ) -> tuple[ModuleResponse | None, bool]:
        """Async version of :meth:`resolve` — calls ``module.arun()`` instead."""
        module = self._branches.get(label)
        if module is None and label not in self._branches:
            if self._has_default:
                module = self._default
            else:
                raise KeyError(
                    f"Router has no branch for label {label!r}. "
                    f"Available: {list(self._branches.keys())}"
                )

        if module is None:
            return None, True

        return await module.arun(text, **kwargs), False
