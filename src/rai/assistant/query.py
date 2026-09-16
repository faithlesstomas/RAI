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
        "projektu",
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
    "system": (
        "system",
        "linux",
        "komputer",
        "komputerze",
        "runtime",
        "daemon",
        "service",
    ),
    "conversation": ("rozmowa", "ustaliliśmy", "conversation", "agreed"),
    "activity": ("wczoraj", "dzisiaj", "aktywność", "robiłem", "activity"),
}

_NON_RESTRICTIVE_QUERY_SCOPES = frozenset({"conversation", "activity"})
GLOBAL_DOMAIN_SCOPE = "global"
UNKNOWN_DOMAIN_SCOPE = "unknown"
LEGACY_GENERAL_SCOPE = "general"
_MEMORY_INTENT_MARKERS = (
    "czy pamiętasz",
    "co pamiętasz",
    "co wiesz o mnie",
    "ustaliliśmy",
    "wcześniej",
    "ostatnio",
    "do you remember",
    "what do you remember",
    "what do you know about me",
    "we agreed",
    "previously",
)
_QUESTION_PREFIXES = (
    "czy ",
    "co ",
    "gdzie ",
    "jak ",
    "jaki ",
    "jaka ",
    "jakie ",
    "what ",
    "where ",
    "which ",
    "how ",
)
_NAMED_PROJECT_PATTERN = re.compile(
    r"\b(?:projektu|projekcie|projektem|projekt|project)\s+"
    r"([A-ZĄĆĘŁŃÓŚŹŻ][\w-]*)"
)


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


def domain_scope_matches(
    evidence_scope: str, query_scopes: tuple[str, ...]
) -> bool:
    """Match hierarchical evidence domains without treating broad recall as isolation.

    ``global`` is explicit cross-domain evidence. ``unknown`` and the legacy
    ``general`` value fail closed for restrictive queries. Conversation and activity
    are query facets rather than isolation boundaries; a facet-only query may search
    every assistant-purpose domain. Once a restrictive scope is present, global
    evidence, matching ancestors and matching descendants remain eligible.
    """
    restrictive = tuple(
        scope
        for scope in query_scopes
        if scope != GLOBAL_DOMAIN_SCOPE
        and scope not in _NON_RESTRICTIVE_QUERY_SCOPES
    )
    if not restrictive:
        return True
    if evidence_scope == GLOBAL_DOMAIN_SCOPE:
        return True
    if evidence_scope in {UNKNOWN_DOMAIN_SCOPE, LEGACY_GENERAL_SCOPE}:
        return False
    if (
        evidence_scope in _NON_RESTRICTIVE_QUERY_SCOPES
        and evidence_scope in query_scopes
    ):
        return True
    return any(
        evidence_scope == scope
        or evidence_scope.startswith(f"{scope}:")
        or scope.startswith(f"{evidence_scope}:")
        for scope in restrictive
    )


def restrictive_domain_scopes(query_scopes: tuple[str, ...]) -> tuple[str, ...]:
    """Return only scopes that impose a retrieval isolation boundary."""
    return tuple(
        scope
        for scope in query_scopes
        if scope != GLOBAL_DOMAIN_SCOPE
        and scope not in _NON_RESTRICTIVE_QUERY_SCOPES
    )


class MemoryQueryResolver:
    """Deterministically extracts semantic query topics and keywords from user text."""

    @staticmethod
    def resolve(text: str, profile_scope: str = "default") -> MemoryQuery:
        """Analyze turn text and return a structured MemoryQuery."""
        lowered = text.casefold()
        words = tuple(re.findall(r"\b\w+\b", lowered))

        matched_topic: str | None = None
        matched_keywords: list[str] = []
        matched_topics: set[str] = set()

        for topic, keywords in _TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in words or kw in lowered:
                    matched_topic = topic
                    matched_topics.add(topic)
                    if kw not in matched_keywords:
                        matched_keywords.append(kw)

        if not matched_keywords:
            matched_keywords.extend(_normalized_words(text)[:8])

        normalized = _normalized_text(text)
        padded_normalized = f" {normalized} "
        domains = [GLOBAL_DOMAIN_SCOPE]
        named_project = _NAMED_PROJECT_PATTERN.search(text)
        named_scopes = {
            "project": (
                f"project:{_normalized_text(named_project.group(1))}"
                if named_project is not None
                else None
            ),
            "system": "system:rai" if re.search(r"\bRAI\b", text) else None,
        }
        for domain, markers in _DOMAIN_MARKERS.items():
            if any(
                f" {_normalized_text(marker)} " in padded_normalized
                for marker in markers
            ):
                domains.append(named_scopes.get(domain) or domain)

        restrictive_domains = tuple(
            domain
            for domain in domains
            if domain != GLOBAL_DOMAIN_SCOPE
            and domain not in _NON_RESTRICTIVE_QUERY_SCOPES
        )
        question_like = "?" in text or any(
            normalized.startswith(marker) for marker in _QUESTION_PREFIXES
        )
        evidence_required = any(
            marker in lowered for marker in _MEMORY_INTENT_MARKERS
        ) or (question_like and bool(restrictive_domains))

        return MemoryQuery(
            topic=matched_topic,
            topic_is_complete=len(matched_topics) == 1,
            keywords=tuple(matched_keywords),
            profile_scope=profile_scope,
            raw_text=text,
            domain_scopes=tuple(dict.fromkeys(domains)),
            evidence_required=evidence_required,
        )
