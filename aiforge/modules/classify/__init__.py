"""Classifier module \u2014 zero/few-shot text classification."""

from aiforge.modules.classify.module import Classifier
from aiforge.modules.classify.schemas import ClassifyExample, ClassifyResponse, LabelScore

__all__ = ["Classifier", "ClassifyExample", "ClassifyResponse", "LabelScore"]
