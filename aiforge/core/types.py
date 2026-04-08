"""
Core types: error hierarchy, BaseModule, CompletionResult, ModuleResponse.
Every module and provider builds on these foundations.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aiforge.core.model_router import QualityTier
    from aiforge.infra.config import AIForgeSettings
    from aiforge.infra.cost_tracker import CostTracker

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class AIForgeError(Exception):
    """Base exception for all AIForge errors."""


class ProviderError(AIForgeError):
    """LLM / embedding provider errors."""


class RateLimitError(ProviderError):
    """Provider rate limit exceeded."""


class APITimeoutError(ProviderError):
    """Provider request timed out."""


class APIConnectionError(ProviderError):
    """Cannot connect to provider API."""


class VectorStoreError(AIForgeError):
    """Vector database errors."""


class ValidationError(AIForgeError):
    """Schema / Pydantic validation failures."""


class PipelineError(AIForgeError):
    """Pipeline execution errors."""


class ConfigError(AIForgeError):
    """Invalid or missing configuration."""


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


class CompletionResult(BaseModel):
    """Normalised result returned by every LLMProvider.complete() call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = Field(
        default="stop",
        description="stop | length | tool_calls | content_filter",
    )
    raw_response: dict[str, Any] | None = None
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModuleResponse(BaseModel):
    """Base response returned by every AIForge module."""

    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0

    def to_text(self) -> str:
        """Extract the primary text payload for pipeline chaining.

        Subclasses should override this to return their main text field.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement to_text() for pipeline chaining."
        )


class BatchResult(BaseModel):
    """Result from BaseModule.run_batch()."""

    results: list[ModuleResponse]
    total_cost: float = 0.0
    total_tokens: int = 0
    failed: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# BaseModule
# ---------------------------------------------------------------------------


class BaseModule(ABC):
    """Base class for all AIForge modules.

    Concrete modules inherit this and implement ``run()`` / ``arun()``.
    All cross-cutting concerns (retry, rate-limiting, cost tracking, logging)
    are handled by ``_call_llm()`` and ``_call_embedding()`` provided here.
    """

    def __init__(
        self, *, quality: str | None = None, provider: str | None = None, **kwargs: Any
    ) -> None:  # noqa: ANN401
        """Initialise the module.

        Args:
            provider: Optional provider name (e.g. ``"openai"``,
                ``"internal_gateway"``).  Must be a built-in provider or
                previously registered via ``register_provider()``.  When
                omitted, falls back to ``settings.PROVIDER``.
            quality: Optional quality tier (``"high"``, ``"medium"``,
                ``"low"``).  When provided (and ``MODEL_ROUTER_ENABLED``
                is true), the model router selects the cheapest model
                meeting this tier for the configured provider.  Falls
                back to ``DEFAULT_QUALITY`` if omitted.
            **kwargs: Forwarded to ``AIForgeSettings`` as overrides.
        """
        # Map the friendly provider= kwarg to the settings field name.
        if provider is not None:
            kwargs["PROVIDER"] = provider
        # Lazy imports to avoid circular deps at class-definition time.
        import structlog

        from aiforge.infra.config import AIForgeSettings, get_settings
        from aiforge.infra.cost_tracker import CostTracker

        explicit_model = "MODEL" in kwargs

        # Use the global singleton when no overrides are provided so that all
        # module instances share one consistent settings object.
        self.settings: AIForgeSettings = AIForgeSettings(**kwargs) if kwargs else get_settings()

        # Fall back to configured DEFAULT_QUALITY when caller omits quality=.
        if quality is None and self.settings.MODEL_ROUTER_ENABLED and not explicit_model:
            quality = self.settings.DEFAULT_QUALITY

        # Model routing: quality= selects the cheapest model meeting the tier.
        if quality is not None and not explicit_model and self.settings.MODEL_ROUTER_ENABLED:
            from aiforge.core.model_router import ModelRouter

            try:
                selected = ModelRouter(
                    custom_models=self.settings.CUSTOM_MODELS or None,
                ).select(
                    quality=cast("QualityTier", quality),
                    provider=self.settings.PROVIDER,
                    prefer_local=self.settings.PREFER_LOCAL,
                )
                # Avoid mutating the shared singleton — create a private copy.
                self.settings = self.settings.model_copy(update={"MODEL": selected})
            except ValueError:
                pass  # No matching model — keep current settings.MODEL

        self.cost_tracker: CostTracker = CostTracker.get_session()
        self.logger = structlog.get_logger(module=self.__class__.__name__)

        # Provider is resolved lazily on first use so that modules that don't
        # make LLM calls (e.g. pure vector-store search) start without hitting
        # the provider initialisation overhead.
        self._provider: Any | None = None
        self._embedding_provider: Any | None = None

    # ------------------------------------------------------------------
    # Provider access (lazy)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from aiforge.providers import get_provider

            self._provider = get_provider(self.settings)
        return self._provider

    @property
    def embedding_provider(self) -> Any:
        if self._embedding_provider is None:
            from aiforge.providers import get_embedding_provider

            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, text: str, **kwargs: Any) -> ModuleResponse:
        """Synchronous execution."""

    @abstractmethod
    async def arun(self, text: str, **kwargs: Any) -> ModuleResponse:
        """Async execution."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def run_stream(self, text: str, **kwargs: Any) -> Iterator[str]:
        """Streaming — override in subclass if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming.")

    async def arun_stream(self, text: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Async streaming — override in subclass if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support async streaming.")
        # Unreachable — but required to make Python treat this as an async generator.
        # See: https://peps.python.org/pep-0525/
        yield  # type: ignore[misc]  # pragma: no cover

    def run_batch(
        self,
        items: list[str],
        concurrency: int = 10,
        max_batch: int = 100,
    ) -> BatchResult:
        """Batch execution with concurrent async calls."""
        import asyncio as _asyncio

        async def _run_all() -> list[ModuleResponse | Exception]:
            sem = _asyncio.Semaphore(concurrency)

            async def _run_one(item: str) -> ModuleResponse | Exception:
                async with sem:
                    try:
                        return await self.arun(item)
                    except Exception as exc:  # noqa: BLE001
                        return exc

            return await _asyncio.gather(*[_run_one(t) for t in items[:max_batch]])

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    raw = future.result()
            else:
                raw = loop.run_until_complete(_run_all())
        except RuntimeError:
            raw = asyncio.run(_run_all())

        results: list[ModuleResponse] = []
        failed: list[dict[str, Any]] = []
        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                failed.append({"index": i, "text": items[i], "error": str(r)})
            else:
                results.append(r)

        return BatchResult(
            results=results,
            total_cost=sum(r.cost for r in results),
            total_tokens=sum(r.input_tokens + r.output_tokens for r in results),
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Internal helpers used by module implementations
    # ------------------------------------------------------------------

    _cache: Any = None  # class-level singleton — set via set_cache()

    @classmethod
    def set_cache(cls, cache: Any) -> None:
        """Set the shared request cache for all modules."""
        cls._cache = cache

    def _get_cache(self) -> Any:
        """Return the active cache (auto-initialise from settings if needed)."""
        if not self.settings.CACHE_ENABLED:
            return None
        if BaseModule._cache is not None:
            return BaseModule._cache
        from aiforge.infra.cache import MemoryBackend, RequestCache

        backend_name = self.settings.CACHE_BACKEND
        if backend_name == "redis":
            from aiforge.infra.cache import RedisBackend

            backend = RedisBackend(url=self.settings.CACHE_REDIS_URL)
        else:
            backend = MemoryBackend(max_entries=self.settings.CACHE_MAX_ENTRIES)

        BaseModule._cache = RequestCache(
            backend=backend,
            ttl_seconds=self.settings.CACHE_TTL,
            enabled=True,
        )
        return BaseModule._cache

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Synchronous LLM call with cache + retry + rate-limiting + cost tracking."""
        from aiforge.infra.rate_limiter import RateLimiter

        use_cache = kwargs.pop("use_cache", True)
        cache = self._get_cache()
        cache_key: str | None = None

        if cache and cache.enabled and use_cache:
            cache_key = cache.build_key(
                model=self.settings.MODEL,
                messages=messages,
                temperature=kwargs.get("temperature", self.settings.TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", self.settings.MAX_TOKENS),
                module=self.__class__.__name__.lower(),
                provider=self.settings.PROVIDER,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return CompletionResult(**cached)

        RateLimiter.get_instance(
            rpm=self.settings.RATE_LIMIT_RPM,
            tpm=self.settings.RATE_LIMIT_TPM,
        ).acquire(tokens=0)  # acquire 1 request slot; TPM accounted post-hoc
        result = self.provider.complete(messages, **kwargs)

        # Post-hoc TPM accounting — now we know the actual token count
        RateLimiter.get_instance(
            rpm=self.settings.RATE_LIMIT_RPM,
            tpm=self.settings.RATE_LIMIT_TPM,
        ).record_tokens(result.input_tokens + result.output_tokens)

        self._record_cost(result)

        if cache and cache_key is not None:
            cache.put(cache_key, result.model_dump())

        return result

    async def _acall_llm(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Async LLM call with cache + retry + rate-limiting + cost tracking."""
        import asyncio

        from aiforge.infra.rate_limiter import RateLimiter

        use_cache = kwargs.pop("use_cache", True)
        cache = self._get_cache()
        cache_key: str | None = None

        if cache and cache.enabled and use_cache:
            cache_key = cache.build_key(
                model=self.settings.MODEL,
                messages=messages,
                temperature=kwargs.get("temperature", self.settings.TEMPERATURE),
                max_tokens=kwargs.get("max_tokens", self.settings.MAX_TOKENS),
                module=self.__class__.__name__.lower(),
                provider=self.settings.PROVIDER,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return CompletionResult(**cached)

        limiter = RateLimiter.get_instance(
            rpm=self.settings.RATE_LIMIT_RPM,
            tpm=self.settings.RATE_LIMIT_TPM,
        )
        # Run blocking acquire in a thread pool to avoid blocking the event loop
        await asyncio.get_event_loop().run_in_executor(None, limiter.acquire, 0)
        result = await self.provider.complete_async(messages, **kwargs)

        # Post-hoc TPM accounting
        await asyncio.get_event_loop().run_in_executor(
            None,
            limiter.record_tokens,
            result.input_tokens + result.output_tokens,
        )

        self._record_cost(result)

        if cache and cache_key is not None:
            cache.put(cache_key, result.model_dump())

        return result

    def _call_embedding(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding call with retry."""
        return self.embedding_provider.embed(texts)

    async def _acall_embedding(self, texts: list[str]) -> list[list[float]]:
        """Async embedding call with retry."""
        return await self.embedding_provider.embed_async(texts)

    def _record_cost(self, result: CompletionResult) -> None:
        """Track token usage and cost in the session cost tracker."""
        from aiforge.core.pricing import calculate_cost

        cost = calculate_cost(result.model, result.input_tokens, result.output_tokens)
        self.cost_tracker.record(
            module=self.__class__.__name__.lower(),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=cost,
        )
