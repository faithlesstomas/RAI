"""Deterministic query resolution for assistant memory retrieval."""

from __future__ import annotations

import re

from .ports import MemoryQuery

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "code_examples": (
        "kod",
        "kodu",
        "kodzie",
        "język",
        "języku",
        "językiem",
        "języka",
        "przykłady",
        "przykładach",
        "przykładów",
        "programowania",
        "code",
        "coding",
        "language",
        "languages",
        "example",
        "examples",
    ),
}


class MemoryQueryResolver:
    """Deterministically extracts semantic query topics and keywords from user text."""

    @staticmethod
    def resolve(text: str, profile_scope: str = "default") -> MemoryQuery:
        """Analyze turn text and return a structured MemoryQuery."""
        lowered = text.lower()
        words = tuple(re.findall(r"\b\w+\b", lowered))

        matched_topic: str | None = None
        matched_keywords: list[str] = []

        for topic, keywords in _TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in words or kw in lowered:
                    matched_topic = topic
                    if kw not in matched_keywords:
                        matched_keywords.append(kw)

        return MemoryQuery(
            topic=matched_topic,
            keywords=tuple(matched_keywords),
            profile_scope=profile_scope,
        )
