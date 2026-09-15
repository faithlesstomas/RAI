"""Deterministic query resolution for assistant memory retrieval."""

from __future__ import annotations

import re
import unicodedata

from .ports import MemoryQuery

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "user.identity.name": (
        "imię",
        "imienia",
        "nazywam",
        "name",
    ),
    "user.identity.age": (
        "wiek",
        "lat",
        "old",
        "age",
    ),
    "user.location.home": (
        "mieszkam",
        "mieszkasz",
        "gdzie",
        "live",
    ),
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

_QUERY_STOP_WORDS = {
    "a",
    "about",
    "czy",
    "do",
    "i",
    "is",
    "jak",
    "jaki",
    "jaka",
    "jakie",
    "jest",
    "mam",
    "me",
    "moj",
    "moja",
    "moje",
    "my",
    "o",
    "pamiętasz",
    "pamietasz",
    "the",
    "to",
    "what",
    "you",
    "że",
}

_DOMAIN_MARKERS: dict[str, tuple[str, ...]] = {
    "personal": (
        "imię",
        "imienia",
        "nazywam",
        "mieszkam",
        "wiek",
        "lat",
        "lubię",
        "wolę",
        "preferuję",
        "moj",
        "moja",
        "moje",
        "o mnie",
        "my name",
        "i live",
        "about me",
    ),
    "project": (
        "projekt",
        "projekcie",
        "projektem",
        "project",
        "repository",
        "repo",
    ),
    "software": (
        "kod",
        "kodu",
        "program",
        "parser",
        "język",
        "python",
        "guile",
        "code",
        "coding",
        "language",
    ),
    "system": ("system", "linux", "komputer", "runtime", "daemon", "service"),
    "conversation": ("rozmowa", "ustaliliśmy", "conversation", "agreed"),
    "activity": ("wczoraj", "dzisiaj", "aktywność", "robiłem", "activity"),
}


def _normalized_words(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return tuple(
        word
        for word in re.findall(r"\b[a-z0-9]+\b", normalized.casefold())
        if len(word) > 1 and word not in _QUERY_STOP_WORDS
    )


def _normalized_text(text: str) -> str:
    """Normalize text without discarding domain-bearing stop words."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"\b[a-z0-9]+\b", normalized.casefold()))


class MemoryQueryResolver:
    """Deterministically extracts semantic query topics and keywords from user text."""

    @staticmethod
    def resolve(text: str, profile_scope: str = "default") -> MemoryQuery:
        """Analyze turn text and return a structured MemoryQuery."""
        lowered = text.casefold()
        words = tuple(re.findall(r"\b\w+\b", lowered))

        matched_topic: str | None = None
        matched_keywords: list[str] = []

        for topic, keywords in _TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in words or kw in lowered:
                    matched_topic = topic
                    if kw not in matched_keywords:
                        matched_keywords.append(kw)

        if not matched_keywords:
            matched_keywords.extend(_normalized_words(text)[:8])

        normalized = _normalized_text(text)
        padded_normalized = f" {normalized} "
        domains = ["general"]
        for domain, markers in _DOMAIN_MARKERS.items():
            if any(
                f" {_normalized_text(marker)} " in padded_normalized
                for marker in markers
            ):
                domains.append(domain)

        return MemoryQuery(
            topic=matched_topic,
            keywords=tuple(matched_keywords),
            profile_scope=profile_scope,
            raw_text=text,
            domain_scopes=tuple(dict.fromkeys(domains)),
        )
