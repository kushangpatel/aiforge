"""AIForge core layer — provider-agnostic abstractions."""

from aiforge.core.doc_loader import DocLoader, IngestStats
from aiforge.core.pricing import (
    MODEL_CONTEXT_WINDOWS,
    PRICING,
    calculate_cost,
    get_context_window,
    get_model_pricing,
)
from aiforge.core.provider import EmbeddingProvider, LLMProvider
from aiforge.core.text_splitter import TextSplitter
from aiforge.core.types import (
    AIForgeError,
    APIConnectionError,
    APITimeoutError,
    BaseModule,
    BatchResult,
    CompletionResult,
    ConfigError,
    ModuleResponse,
    PipelineError,
    ProviderError,
    RateLimitError,
    ValidationError,
    VectorStoreError,
)
from aiforge.core.vectorstore import ChromaVectorStore, VectorStoreAdapter

__all__ = [
    # Types
    "AIForgeError",
    "ProviderError",
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "VectorStoreError",
    "ValidationError",
    "PipelineError",
    "ConfigError",
    "CompletionResult",
    "ModuleResponse",
    "BatchResult",
    "BaseModule",
    # Abstractions
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStoreAdapter",
    "ChromaVectorStore",
    # Utilities
    "DocLoader",
    "IngestStats",
    "TextSplitter",
    # Pricing
    "PRICING",
    "MODEL_CONTEXT_WINDOWS",
    "calculate_cost",
    "get_model_pricing",
    "get_context_window",
]
