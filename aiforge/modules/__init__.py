"""
AIForge AI modules.

Each sub-package exposes a single main class that inherits from
:class:`aiforge.core.types.BaseModule`:

- :mod:`aiforge.modules.summarize`  — :class:`~aiforge.modules.summarize.module.Summarizer`
- :mod:`aiforge.modules.qa`         — :class:`~aiforge.modules.qa.module.DocQA`
- :mod:`aiforge.modules.extract`    — :class:`~aiforge.modules.extract.module.Extractor`
- :mod:`aiforge.modules.classify`   — :class:`~aiforge.modules.classify.module.Classifier`
- :mod:`aiforge.modules.search`     — :class:`~aiforge.modules.search.module.SemanticSearch`
- :mod:`aiforge.modules.generate`   — :class:`~aiforge.modules.generate.module.Generator`
- :mod:`aiforge.modules.chat`       — :class:`~aiforge.modules.chat.module.ChatAgent`
- :mod:`aiforge.modules.moderate`   — :class:`~aiforge.modules.moderate.module.Moderator`
- :mod:`aiforge.modules.translate`  — :class:`~aiforge.modules.translate.module.Translator`
- :mod:`aiforge.modules.vision`     — :class:`~aiforge.modules.vision.module.Vision`
"""

__all__ = [
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
]
