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
from aiforge.infra.config import AIForgeSettings, get_settings
from aiforge.infra.cost_tracker import CostTracker
from aiforge.infra.logging import configure_logging
from aiforge.infra.tracing import setup_tracing
from aiforge.pipeline.pipeline import Pipeline
from aiforge.pipeline.router import Router
from aiforge.providers import get_embedding_provider, get_provider, register_provider

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
    # Providers
    "get_provider",
    "get_embedding_provider",
    "register_provider",
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
