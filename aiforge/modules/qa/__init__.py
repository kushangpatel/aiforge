"""DocQA module — document question answering with RAG and agentic refinement."""

from aiforge.modules.qa.module import DocQA
from aiforge.modules.qa.schemas import IngestResponse, QAResponse

__all__ = [
    "DocQA",
    "IngestResponse",
    "QAResponse",
]
