"""
:class:`Guard` — composes input and output checks and wraps callables.

Usage::

    from aiforge import Summarizer
    from aiforge.guardrails import Guard, PIIGuard, PromptInjectionGuard, MaxLengthGuard

    guard = Guard(
        input=[PIIGuard(), PromptInjectionGuard(), MaxLengthGuard(max_tokens=4000)],
        output=[PIIGuard()],
        on_fail="redact",
    )

    guard.check_input("email me at a@b.co").sanitized_text   # "email me at [EMAIL_1]"
    safe = guard.wrap(Summarizer())
    safe.run(document)                                          # GuardrailViolation if blocked
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from pydantic import BaseModel, Field

from aiforge.core.types import GuardrailViolation
from aiforge.guardrails.checks import Check, GuardContext, OnFail, Stage, Violation
from aiforge.guardrails.pii import PIIVault, rehydrate

_log = structlog.get_logger("aiforge.guardrails")

_TEXT_KWARGS = ("text", "prompt", "query", "question", "input", "message")


class GuardResult(BaseModel):
    """Outcome of running one stage of a :class:`Guard`.

    ``passed`` is False only when a violation *blocked*; violations that were
    redacted or warned on are still listed.
    """

    passed: bool = True
    stage: Stage = "input"
    violations: list[Violation] = Field(default_factory=list)
    sanitized_text: str = ""
    original_text: str = ""
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Placeholder -> original value for PII redacted in this stage.",
    )

    @property
    def blocked(self) -> bool:
        return not self.passed

    @property
    def modified(self) -> bool:
        return self.sanitized_text != self.original_text

    def raise_for_violation(self) -> None:
        """Raise :class:`GuardrailViolation` if this result blocked."""
        if not self.passed:
            blocking = [v for v in self.violations if v.action == "blocked"]
            kinds = ", ".join(sorted({v.type for v in blocking})) or "violation"
            raise GuardrailViolation(
                f"Guardrail blocked {self.stage}: {kinds}", stage=self.stage, result=self
            )


class Guard:
    """A set of input and output checks with a shared failure policy.

    Args:
        input:   Checks run on text going *into* the model.
        output:  Checks run on text coming *out*.
        on_fail: ``"block"`` (fail the call), ``"redact"`` (repair where the
                 check can, block where it cannot) or ``"warn"`` (log and let
                 it through).  Each check may override it with its own
                 ``on_fail``.  Defaults to ``settings.GUARDRAILS_ON_FAIL``.
    """

    def __init__(
        self,
        input: Sequence[Check] = (),  # noqa: A002 — mirrors the spoken API
        output: Sequence[Check] = (),
        *,
        on_fail: OnFail | None = None,
    ) -> None:
        self.input: list[Check] = list(input)
        self.output: list[Check] = list(output)
        if on_fail is None:
            try:
                from aiforge.infra.config import get_settings

                on_fail = get_settings().GUARDRAILS_ON_FAIL
            except Exception:  # noqa: BLE001
                on_fail = "block"
        if on_fail not in ("block", "redact", "warn"):
            raise ValueError(f"on_fail must be 'block', 'redact' or 'warn', got {on_fail!r}")
        self.on_fail: OnFail = on_fail

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check_input(self, text: str, *, vault: PIIVault | None = None) -> GuardResult:
        """Run the input checks on *text*."""
        return self._run_sync(self.input, text, GuardContext("input", vault or PIIVault()))

    def check_output(self, text: str, *, vault: PIIVault | None = None) -> GuardResult:
        """Run the output checks on *text*."""
        return self._run_sync(self.output, text, GuardContext("output", vault or PIIVault()))

    async def acheck_input(self, text: str, *, vault: PIIVault | None = None) -> GuardResult:
        """Async :meth:`check_input`."""
        return await self._run_async(self.input, text, GuardContext("input", vault or PIIVault()))

    async def acheck_output(self, text: str, *, vault: PIIVault | None = None) -> GuardResult:
        """Async :meth:`check_output`."""
        return await self._run_async(self.output, text, GuardContext("output", vault or PIIVault()))

    def _run_sync(
        self, checks: list[Check], text: str, ctx: GuardContext, *, fix_only: bool = False
    ) -> GuardResult:
        state = _RunState(text, ctx)
        for check in checks:
            if state.blocked:
                break
            if fix_only and (check.on_fail or self.on_fail) != "redact":
                continue  # nothing to repair; avoid re-running (possibly LLM) checks
            self._apply(check, check.validate(state.text, ctx), state, fix_only)
        return state.result()

    async def _run_async(
        self, checks: list[Check], text: str, ctx: GuardContext, *, fix_only: bool = False
    ) -> GuardResult:
        state = _RunState(text, ctx)
        for check in checks:
            if state.blocked:
                break
            self._apply(check, await check.avalidate(state.text, ctx), state, fix_only)
        return state.result()

    def _apply(
        self, check: Check, found: list[Violation], state: _RunState, fix_only: bool
    ) -> None:
        if not found:
            return
        mode = check.on_fail or self.on_fail
        if fix_only:
            # Re-applying repairs to parts of an output that already passed.
            if mode == "redact":
                fixed = check.fix(state.text, found, state.ctx)
                state.text = fixed if fixed is not None else state.text
            return
        if mode == "redact":
            fixed = check.fix(state.text, found, state.ctx)
            if fixed is not None:
                state.text = fixed
                state.add(found, "redacted")
                return
            mode = "block"
        if mode == "warn":
            _log.warning(
                "guardrails.violation",
                stage=state.ctx.stage,
                check=check.name,
                types=sorted({v.subtype or v.type for v in found}),
            )
            state.add(found, "warned")
            return
        state.add(found, "blocked")
        state.blocked = True

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap(self, target: Any, *, rehydrate: bool = True) -> GuardedCallable:
        """Return a guarded version of a module or callable.

        The input text is taken from the first positional ``str`` argument, a
        ``text=`` / ``prompt=`` / ``query=`` style keyword, or the user turns of
        an OpenAI-style message list.  Outputs that are strings, or pydantic
        models (e.g. any ``ModuleResponse``), are checked and sanitised.

        Args:
            target:    A module with ``run`` / ``arun``, or any sync or async
                       callable.
            rehydrate: Restore input PII placeholders (``[EMAIL_1]``) in the
                       output, so callers see their own data back.
        """
        return GuardedCallable(self, target, rehydrate=rehydrate)


class _RunState:
    def __init__(self, text: str, ctx: GuardContext) -> None:
        self.original = text
        self.text = text
        self.ctx = ctx
        self.blocked = False
        self.violations: list[Violation] = []
        self._before = set(ctx.vault.mapping)

    def add(self, found: list[Violation], action: Any) -> None:
        self.violations += [v.model_copy(update={"action": action}) for v in found]

    def result(self) -> GuardResult:
        return GuardResult(
            passed=not self.blocked,
            stage=self.ctx.stage,
            violations=self.violations,
            sanitized_text=self.text,
            original_text=self.original,
            mapping={k: v for k, v in self.ctx.vault.mapping.items() if k not in self._before},
        )


# ---------------------------------------------------------------------------
# Guarded callable
# ---------------------------------------------------------------------------


class GuardedCallable:
    """A module or function behind a :class:`Guard`.

    Call it like the original (``guarded(...)`` / ``guarded.run(...)``) or
    await ``guarded.acall(...)`` / ``guarded.arun(...)``.  When the wrapped
    function is a coroutine function, ``guarded(...)`` returns an awaitable.
    Other attributes are forwarded to the wrapped object.
    """

    def __init__(self, guard: Guard, target: Any, *, rehydrate: bool = True) -> None:
        self.guard = guard
        self.target = target
        self.rehydrate = rehydrate
        run, arun = getattr(target, "run", None), getattr(target, "arun", None)
        if callable(run) and callable(arun) and not inspect.isfunction(target):
            self._sync: Callable[..., Any] | None = run
            self._async: Callable[..., Awaitable[Any]] | None = arun
        elif inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
            getattr(target, "__call__", None)  # noqa: B004
        ):
            self._sync, self._async = None, target
        elif callable(target):
            self._sync, self._async = target, None
        else:
            raise TypeError(f"Cannot guard {type(target).__name__}: not callable")
        if inspect.isfunction(target) or inspect.ismethod(target):
            functools.update_wrapper(self, target)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    # -- public entry points ------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._sync is None:
            return self.acall(*args, **kwargs)
        return self.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Guard the input, call the target synchronously, guard the output."""
        if self._sync is None:
            raise TypeError("Wrapped target is async-only; use `await guarded.acall(...)`")
        vault = PIIVault()
        args, kwargs, mapping = self._guard_args_sync(args, kwargs, vault)
        out = self._sync(*args, **kwargs)
        return self._guard_output_sync(out, vault, mapping)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async :meth:`run` (sync targets run in a worker thread)."""
        vault = PIIVault()
        args, kwargs, mapping = await self._guard_args_async(args, kwargs, vault)
        if self._async is not None:
            out = await self._async(*args, **kwargs)
        else:
            assert self._sync is not None
            out = await asyncio.to_thread(self._sync, *args, **kwargs)
        return await self._guard_output_async(out, vault, mapping)

    arun = acall

    # -- input ---------------------------------------------------------------

    @staticmethod
    def _locate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[tuple[str, Any, str]]:
        """Find the guarded text slots as ``(where, key, text)`` triples."""
        if args and isinstance(args[0], str):
            return [("arg", 0, args[0])]
        for k in _TEXT_KWARGS:
            if isinstance(kwargs.get(k), str):
                return [("kw", k, kwargs[k])]
        msgs = args[0] if args else kwargs.get("messages")
        if isinstance(msgs, list):
            return [
                ("msg", i, m["content"])
                for i, m in enumerate(msgs)
                if isinstance(m, dict)
                and m.get("role") == "user"
                and isinstance(m.get("content"), str)
            ]
        return []

    @staticmethod
    def _replace(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        slots: list[tuple[str, Any, str]],
        new: list[str],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        a, kw = list(args), dict(kwargs)
        for (where, key, _), text in zip(slots, new, strict=True):
            if where == "arg":
                a[key] = text
            elif where == "kw":
                kw[key] = text
            else:
                container = a if a and isinstance(a[0], list) else None
                msgs = list(container[0] if container else kw["messages"])
                msgs[key] = {**msgs[key], "content": text}
                if container is not None:
                    a[0] = msgs
                else:
                    kw["messages"] = msgs
        return tuple(a), kw

    def _guard_args_sync(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], vault: PIIVault
    ) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
        slots = self._locate(args, kwargs)
        results = [self.guard.check_input(t, vault=vault) for _, _, t in slots]
        return self._finish_input(args, kwargs, slots, results)

    async def _guard_args_async(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], vault: PIIVault
    ) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
        slots = self._locate(args, kwargs)
        results = [await self.guard.acheck_input(t, vault=vault) for _, _, t in slots]
        return self._finish_input(args, kwargs, slots, results)

    def _finish_input(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        slots: list[tuple[str, Any, str]],
        results: list[GuardResult],
    ) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
        mapping: dict[str, str] = {}
        for r in results:
            r.raise_for_violation()
            mapping.update(r.mapping)
        if any(r.modified for r in results):
            args, kwargs = self._replace(args, kwargs, slots, [r.sanitized_text for r in results])
        return args, kwargs, mapping

    # -- output --------------------------------------------------------------

    def _guard_output_sync(self, out: Any, vault: PIIVault, mapping: dict[str, str]) -> Any:
        text = _primary_text(out)
        if text is None:
            return out
        result = self.guard.check_output(text, vault=vault)
        result.raise_for_violation()
        return self._finish_output(out, result, vault, mapping)

    async def _guard_output_async(self, out: Any, vault: PIIVault, mapping: dict[str, str]) -> Any:
        text = _primary_text(out)
        if text is None:
            return out
        result = await self.guard.acheck_output(text, vault=vault)
        result.raise_for_violation()
        return self._finish_output(out, result, vault, mapping)

    def _finish_output(
        self, out: Any, result: GuardResult, vault: PIIVault, mapping: dict[str, str]
    ) -> Any:
        restore = mapping if self.rehydrate else {}
        if isinstance(out, str):
            return rehydrate(result.sanitized_text, restore)
        if not (result.modified or restore) or not isinstance(out, BaseModel):
            return out
        # Apply the repairs field by field so the model keeps its shape.
        ctx_vault = vault
        update: dict[str, Any] = {}
        for name in type(out).model_fields:
            value = getattr(out, name)
            if not isinstance(value, str) or not value:
                continue
            new = value
            if result.modified:
                new = self.guard._run_sync(
                    self.guard.output, value, GuardContext("output", ctx_vault), fix_only=True
                ).sanitized_text
            new = rehydrate(new, restore)
            if new != value:
                update[name] = new
        return out.model_copy(update=update) if update else out


def _primary_text(out: Any) -> str | None:
    if isinstance(out, str):
        return out
    to_text = getattr(out, "to_text", None)
    if callable(to_text):
        try:
            value = to_text()
        except NotImplementedError:
            value = None
        if isinstance(value, str):
            return value
    text = getattr(out, "text", None)
    return text if isinstance(text, str) else None
