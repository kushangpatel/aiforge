"""Classifier module — zero/few-shot text classification and sentiment analysis."""

from aiforge.modules.classify.module import Classifier
from aiforge.modules.classify.schemas import (
    AspectSentiment,
    ClassifyExample,
    ClassifyResponse,
    EmotionScore,
    LabelScore,
    SentimentResponse,
)

__all__ = [
    "Classifier",
    "ClassifyExample",
    "ClassifyResponse",
    "LabelScore",
    "SentimentResponse",
    "AspectSentiment",
    "EmotionScore",
]
