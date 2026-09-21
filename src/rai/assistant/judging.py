"""Deterministic answer judgment shared by retrieval and routing benchmarks."""

from __future__ import annotations

import unicodedata
import re

_ABSTENTION_MARKERS = (
    "nie wiem",
    "nie mam wystarczających",
    "brak wystarczających",
    "nie mogę odpowiedzieć",
    "nie mam informacji",
    "nie ma informacji",
    "nie posiadam informacji",
    "nie znam",
    "brak informacji",
    "brak jest danych",
    "nie mam dostępu",
    "nie mogę wskazać",
    "nie mogę podać",
    "brak danych",
    "nie podałeś",
    "nie podałaś",
    "nie podano",
    "nie mam w pamięci",
    "i don't know",
    "insufficient evidence",
    "cannot answer",
)
JUDGE_VERSION = "deterministic-phrase-and-abstention-v3"
_MIN_INFLECTION_PREFIX = 6


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_tokens(value: str) -> tuple[str, ...]:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )
    return tuple(re.findall(r"[a-z0-9]+", ascii_value))


def _required_phrase_matches(phrase: str, answer: str) -> bool:
    """Allow conservative inflection variants without fuzzy semantic grading."""
    expected = _normalized_tokens(phrase)
    actual = _normalized_tokens(answer)
    if not expected:
        return True
    for expected_token in expected:
        if not any(
            actual_token == expected_token
            or (
                min(len(actual_token), len(expected_token)) >= _MIN_INFLECTION_PREFIX
                and _common_prefix_length(actual_token, expected_token)
                >= _MIN_INFLECTION_PREFIX
            )
            for actual_token in actual
        ):
            return False
    return True


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        length += 1
    return length


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
        _required_phrase_matches(phrase, answer_text) for phrase in expected_phrases
    )
    forbidden = any(
        _normalized_text(phrase) in normalized for phrase in forbidden_phrases
    )
    if expected_abstention:
        return abstained and not forbidden, abstained
    return required and not forbidden and not abstained, abstained
