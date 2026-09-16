"""Deterministic answer judgment shared by retrieval and routing benchmarks."""

from __future__ import annotations

import unicodedata

_ABSTENTION_MARKERS = (
    "nie wiem",
    "nie mam wystarczających",
    "brak wystarczających",
    "nie mogę odpowiedzieć",
    "i don't know",
    "insufficient evidence",
    "cannot answer",
)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def judge_answer(
    answer_text: str,
    *,
    expected_phrases: tuple[str, ...],
    forbidden_phrases: tuple[str, ...],
    expected_abstention: bool,
) -> tuple[bool, bool]:
    """Return correctness and abstention without asking the answer model to grade."""
    normalized = _normalized_text(answer_text)
    abstained = any(marker in normalized for marker in _ABSTENTION_MARKERS)
    required = all(
        _normalized_text(phrase) in normalized for phrase in expected_phrases
    )
    forbidden = any(
        _normalized_text(phrase) in normalized for phrase in forbidden_phrases
    )
    if expected_abstention:
        return abstained and not forbidden, abstained
    return required and not forbidden and not abstained, abstained
