"""
AIForge — Plug-and-play AI toolkit for product teams.

Quick start::

    from aiforge import Summarizer, DocQA, Classifier, Extractor
    from aiforge import SemanticSearch, Generator, ChatAgent, Moderator
    from aiforge import Pipeline, CostTracker
"""

from aiforge.__version__ import __version__
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
from aiforge.core.model_router import MODEL_CATALOG, ModelRouter, ModelSpec
from aiforge.core.vectorstore import VectorStoreAdapter
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
# Module imports (lazy — will be populated as modules are implemented)
# ---------------------------------------------------------------------------

# These are currently stub placeholders.  Each will be replaced with the
# real implementation as the respective module is built out.

try:
    from aiforge.modules.summarize.module import Summarizer
except ImportError:
    Summarizer = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.qa.module import DocQA
except ImportError:
    DocQA = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.extract.module import Extractor
except ImportError:
    Extractor = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.classify.module import Classifier
except ImportError:
    Classifier = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.search.module import SemanticSearch
except ImportError:
    SemanticSearch = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.generate.module import Generator
except ImportError:
    Generator = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.chat.module import ChatAgent
except ImportError:
    ChatAgent = None  # type: ignore[assignment,misc]

try:
    from aiforge.modules.moderate.module import Moderator
except ImportError:
    Moderator = None  # type: ignore[assignment,misc]

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
}

for _name, _cls in _BUILTIN_MODULES.items():
    if _cls is not None:
        register_module(_name, _cls)

# Built-in vector stores
try:
    from aiforge.core.vectorstore import ChromaVectorStore  # noqa: E402

    register_vectorstore("chroma", ChromaVectorStore)
except Exception:  # chromadb may not be installed
    ChromaVectorStore = None  # type: ignore[assignment,misc]

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
]
