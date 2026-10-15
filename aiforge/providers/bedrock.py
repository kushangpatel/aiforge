"""
Amazon Bedrock provider: Converse API chat models (Claude, Nova, …) + Bedrock embeddings.

Requires:
    pip install "aiforge-kit[bedrock]"
    AIFORGE_AWS_REGION (or the AWS default region), optionally AIFORGE_AWS_PROFILE

Credentials come from the standard AWS chain (environment variables, shared
config/credentials files, SSO, instance or container roles). AIForge never
reads or stores AWS secrets itself.

Prompt caching uses Bedrock ``cachePoint`` blocks on models that support them
(Claude 3.7 and later, Amazon Nova):

* ``PROMPT_CACHE="system"`` (default) puts a cache point after the system prompt.
* ``PROMPT_CACHE="auto"`` lets ``langchain-aws`` place cache points on the
  system prompt, the tools and the end of the conversation.
* Content blocks marked with ``cache_control`` (e.g. ``messages.system(..., cache=True)``)
  are turned into cache points in both modes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from aiforge.core.messages import CACHE_BREAKPOINT
from aiforge.core.types import ProviderError
from aiforge.providers._errors import wrap_sdk_error
from aiforge.providers._langchain import LangChainEmbeddingProvider, LangChainLLMProvider
from aiforge.providers._utils import supports_sampling_params

if TYPE_CHECKING:
    from aiforge.infra.config import AIForgeSettings

_MISSING = 'langchain-aws is not installed. Run: pip install "aiforge-kit[bedrock]"'

CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}

# Claude 3.0 / 3.5 models predate Bedrock prompt caching.
_OLD_CLAUDE_RE = re.compile(r"anthropic\.claude-(3-(?!7)|instant|v2)")


def supports_prompt_cache(model_id: str) -> bool:
    """Whether a Bedrock model accepts ``cachePoint`` blocks (Claude 3.7+ and Nova)."""
    name = model_id.lower()
    if "amazon.nova" in name:
        return True
    return "anthropic.claude" in name and not _OLD_CLAUDE_RE.search(name)


def _session_kwargs(settings: AIForgeSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if settings.AWS_REGION:
        kwargs["region_name"] = settings.AWS_REGION
    if settings.AWS_PROFILE:
        kwargs["credentials_profile_name"] = settings.AWS_PROFILE
    return kwargs


def _to_cache_points(content: Any, enabled: bool) -> tuple[Any, bool]:
    """Replace ``cache_control`` markers on content blocks with Bedrock cache points.

    Returns the new content and whether any marker was found. When caching is
    not *enabled* for the model the markers are only stripped.
    """
    if not isinstance(content, list):
        return content, False
    out: list[Any] = []
    found = False
    for block in content:
        if isinstance(block, dict) and "cache_control" in block:
            found = True
            out.append({k: v for k, v in block.items() if k != "cache_control"})
            if enabled:
                out.append(dict(CACHE_POINT))
        else:
            out.append(block)
    return out, found


class BedrockProvider(LangChainLLMProvider):
    """LLM provider backed by the Amazon Bedrock Converse API (``langchain-aws``)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._settings = settings
        self._model_name = settings.MODEL
        self._cache = supports_prompt_cache(settings.MODEL)
        self._llm = ChatBedrockConverse(
            model=settings.MODEL,
            temperature=settings.TEMPERATURE if supports_sampling_params(settings.MODEL) else None,
            max_tokens=settings.MAX_TOKENS,
            **_session_kwargs(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)

    def _request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        call_kwargs = dict(super()._request_kwargs(kwargs))
        if (
            self._cache
            and self._settings.PROMPT_CACHE == "auto"
            and "cache_control" not in call_kwargs
        ):
            call_kwargs["cache_control"] = dict(CACHE_BREAKPOINT)
        return call_kwargs

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Turn ``cache_control`` markers into cache points; mark the system prompt.

        With ``PROMPT_CACHE="system"`` a cache point follows the last system
        message, so tools and system prompt form a cached prefix. Prompts below
        the model's minimum cacheable length are simply not cached.
        """
        prepared: list[dict[str, Any]] = []
        marked = False
        for msg in messages:
            content, found = _to_cache_points(msg.get("content"), self._cache)
            marked = marked or found
            prepared.append({**msg, "content": content} if found else msg)

        if not self._cache or marked or self._settings.PROMPT_CACHE != "system":
            return prepared
        idx = next(
            (i for i in range(len(prepared) - 1, -1, -1) if prepared[i].get("role") == "system"),
            None,
        )
        if idx is None:
            return prepared
        content = prepared[idx].get("content") or ""
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        prepared[idx] = {**prepared[idx], "content": [*blocks, dict(CACHE_POINT)]}
        return prepared


class BedrockEmbeddingProvider(LangChainEmbeddingProvider):
    """Embedding provider backed by Bedrock embeddings (Titan Text Embeddings V2 …)."""

    def __init__(self, settings: AIForgeSettings) -> None:
        try:
            from langchain_aws import BedrockEmbeddings
        except ImportError as exc:
            raise ImportError(_MISSING) from exc

        self._model_name = settings.EMBEDDING_MODEL
        self._embeddings = BedrockEmbeddings(
            model_id=settings.EMBEDDING_MODEL,
            **_session_kwargs(settings),
        )

    def _wrap_error(self, exc: Exception) -> ProviderError:
        return wrap_sdk_error(exc)
