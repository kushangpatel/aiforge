"""
AIForge — Plug-and-play AI toolkit for product teams.

Quick start::

    from aiforge import Summarizer, DocQA, Classifier, Extractor
    from aiforge import SemanticSearch, Generator, ChatAgent, Moderator
    from aiforge import Pipeline, CostTracker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiforge.__version__ import __version__
from aiforge.core.model_router import MODEL_CATALOG, ModelRouter, ModelSpec
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
from aiforge.core.vectorstore import VectorStoreAdapter
from aiforge.core.vectorstore_factory import build_vectorstore
from aiforge.infra.config import AIForgeSettings, get_settings
from aiforge.infra.cost_tracker import CostTracker
from aiforge.infra.logging import configure_logging
from aiforge.infra.tracing import setup_tracing
from aiforge.pipeline.pipeline import Pipeline
from aiforge.pipeline.router import Router
from aiforge.providers import get_embedding_provider
from aiforge.registry import (
    get_module,
    get_provider,
    get_vectorstore,
    list_modules,
    list_providers,
    list_vectorstores,
    register_module,
    register_provider,
    register_vectorstore,
)

# Initialise logging as early as possible (uses defaults; can be reconfigured later)
configure_logging(
    level=get_settings().LOG_LEVEL,
    fmt=get_settings().LOG_FORMAT,
)
# Initialise tracing if enabled in settings
setup_tracing(get_settings())

# ---------------------------------------------------------------------------
# Module & response schema imports
# ---------------------------------------------------------------------------
# At type-checking time (mypy, pyright, Pylance) the real classes are always
# visible so IDE auto-complete, type narrowing, and hover docs work.
# At runtime the try/except blocks provide graceful degradation when an
# optional dependency (e.g. streamlit, pinecone) is missing.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    # Modules
    from aiforge.modules.chat.module import ChatAgent as ChatAgent

    # Response schemas
    from aiforge.modules.chat.schemas import ChatResponse as ChatResponse
    from aiforge.modules.classify.module import Classifier as Classifier
    from aiforge.modules.classify.schemas import ClassifyResponse as ClassifyResponse
    from aiforge.modules.classify.schemas import SentimentResponse as SentimentResponse
    from aiforge.modules.extract.module import Extractor as Extractor
    from aiforge.modules.extract.schemas import ExtractResponse as ExtractResponse
    from aiforge.modules.generate.module import Generator as Generator
    from aiforge.modules.generate.schemas import GenerateResponse as GenerateResponse
    from aiforge.modules.moderate.module import Moderator as Moderator
    from aiforge.modules.moderate.schemas import ModerateResponse as ModerateResponse
    from aiforge.modules.qa.module import DocQA as DocQA
    from aiforge.modules.qa.schemas import IngestResponse as IngestResponse
    from aiforge.modules.qa.schemas import QAResponse as QAResponse
    from aiforge.modules.search.module import SemanticSearch as SemanticSearch
    from aiforge.modules.search.schemas import IndexResponse as IndexResponse
    from aiforge.modules.search.schemas import SearchResponse as SearchResponse
    from aiforge.modules.summarize.module import Summarizer as Summarizer
    from aiforge.modules.summarize.schemas import SummarizeResponse as SummarizeResponse
    from aiforge.modules.translate.module import Translator as Translator
    from aiforge.modules.translate.schemas import TranslateResponse as TranslateResponse
    from aiforge.modules.vision.module import Vision as Vision
    from aiforge.modules.vision.schemas import VisionResponse as VisionResponse
else:
    try:
        from aiforge.modules.summarize.module import Summarizer
    except ImportError:
        Summarizer = None

    try:
        from aiforge.modules.qa.module import DocQA
    except ImportError:
        DocQA = None

    try:
        from aiforge.modules.extract.module import Extractor
    except ImportError:
        Extractor = None

    try:
        from aiforge.modules.classify.module import Classifier
    except ImportError:
        Classifier = None

    try:
        from aiforge.modules.search.module import SemanticSearch
    except ImportError:
        SemanticSearch = None

    try:
        from aiforge.modules.generate.module import Generator
    except ImportError:
        Generator = None

    try:
        from aiforge.modules.chat.module import ChatAgent
    except ImportError:
        ChatAgent = None

    try:
        from aiforge.modules.moderate.module import Moderator
    except ImportError:
        Moderator = None

    try:
        from aiforge.modules.translate.module import Translator
    except ImportError:
        Translator = None

    try:
        from aiforge.modules.vision.module import Vision
    except ImportError:
        Vision = None

    # Response schemas — always importable (no optional deps)
    from aiforge.modules.chat.schemas import ChatResponse
    from aiforge.modules.classify.schemas import ClassifyResponse, SentimentResponse
    from aiforge.modules.extract.schemas import ExtractResponse
    from aiforge.modules.generate.schemas import GenerateResponse
    from aiforge.modules.moderate.schemas import ModerateResponse
    from aiforge.modules.qa.schemas import IngestResponse, QAResponse
    from aiforge.modules.search.schemas import IndexResponse, SearchResponse
    from aiforge.modules.summarize.schemas import SummarizeResponse
    from aiforge.modules.translate.schemas import TranslateResponse
    from aiforge.modules.vision.schemas import VisionResponse

# ---------------------------------------------------------------------------
# Auto-register built-in modules and vector stores
# ---------------------------------------------------------------------------

_BUILTIN_MODULES: dict[str, type | None] = {
    "summarize": Summarizer,
    "qa": DocQA,
    "extract": Extractor,
    "classify": Classifier,
    "search": SemanticSearch,
    "generate": Generator,
    "chat": ChatAgent,
    "moderate": Moderator,
    "translate": Translator,
    "vision": Vision,
}

for _name, _cls in _BUILTIN_MODULES.items():
    if _cls is not None:
        register_module(_name, _cls)

# Built-in vector stores
try:
    from aiforge.core.vectorstore import ChromaVectorStore  # noqa: E402

    register_vectorstore("chroma", ChromaVectorStore)
except Exception:  # chromadb may not be installed
    ChromaVectorStore = None  # type: ignore[assignment]

try:
    from aiforge.core.vectorstore_pinecone import PineconeVectorStore  # noqa: E402

    register_vectorstore("pinecone", PineconeVectorStore)
except Exception:  # pinecone may not be installed
    PineconeVectorStore = None  # type: ignore[assignment]

try:
    from aiforge.core.vectorstore_qdrant import QdrantVectorStore  # noqa: E402

    register_vectorstore("qdrant", QdrantVectorStore)
except Exception:  # qdrant-client may not be installed
    QdrantVectorStore = None  # type: ignore[assignment]

try:
    from aiforge.core.vectorstore_pgvector import PgVectorStore  # noqa: E402

    register_vectorstore("pgvector", PgVectorStore)
except Exception:  # psycopg may not be installed
    PgVectorStore = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    # Core types
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
    # Config & infra
    "AIForgeSettings",
    "get_settings",
    "CostTracker",
    # Model router
    "ModelRouter",
    "ModelSpec",
    "MODEL_CATALOG",
    # Providers
    "get_provider",
    "get_embedding_provider",
    "register_provider",
    "list_providers",
    # Registry (plugin system)
    "register_module",
    "get_module",
    "list_modules",
    "register_vectorstore",
    "get_vectorstore",
    "list_vectorstores",
    "VectorStoreAdapter",
    "ChromaVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "PgVectorStore",
    "build_vectorstore",
    # Pipeline
    "Pipeline",
    "Router",
    # Modules
    "Summarizer",
    "DocQA",
    "Extractor",
    "Classifier",
    "SemanticSearch",
    "Generator",
    "ChatAgent",
    "Moderator",
    "Translator",
    "Vision",
    # Response schemas
    "SummarizeResponse",
    "QAResponse",
    "IngestResponse",
    "ExtractResponse",
    "ClassifyResponse",
    "SentimentResponse",
    "SearchResponse",
    "IndexResponse",
    "GenerateResponse",
    "ChatResponse",
    "ModerateResponse",
    "TranslateResponse",
    "VisionResponse",
]
