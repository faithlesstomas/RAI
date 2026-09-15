"""Deterministic, auditable extraction and grounding for durable user memory."""

from __future__ import annotations

import re
import unicodedata

from rai.kernel.records import ProducerIdentity, _new_id, _utc_now

from .records import MemoryProposal

_NAME_VALUE = r"([A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻąćęłńóśźż'’-]{1,63})"
_NAME_PATTERNS = (
    re.compile(rf"\b(?i:mam na imię|nazywam się)\s+{_NAME_VALUE}", re.UNICODE),
    re.compile(
        rf"(?:^|[.!?]\s+)(?i:(?:cześć[\s,!-]*)?jestem)\s+{_NAME_VALUE}(?=[\s,!.?]|$)",
        re.UNICODE,
    ),
    re.compile(rf"\bmy name is\s+{_NAME_VALUE}", re.IGNORECASE | re.UNICODE),
)
_LOCATION_PATTERNS = (
    re.compile(
        r"\b(?i:mieszkam|żyję)\s+(?i:w)\s+([A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻąćęłńóśźż'’ -]{1,80}?)(?=[,.!?]|$)",
        re.UNICODE,
    ),
    re.compile(
        r"\bi live in\s+([A-Z][\w'’ -]{1,80}?)(?=[,.!?]|$)",
        re.IGNORECASE | re.UNICODE,
    ),
)
_AGE_PATTERNS = (
    re.compile(r"\bmam\s+(\d{1,3})\s+lat(?:a)?\b", re.IGNORECASE),
    re.compile(r"\bi am\s+(\d{1,3})\s+years? old\b", re.IGNORECASE),
)
_EXPLICIT_FACT_PATTERNS = (
    re.compile(r"\bzapamiętaj(?: proszę)?[, :]*(?:że\s+)?(.+)", re.IGNORECASE),
    re.compile(r"\bremember(?: please)?[, :]*(?:that\s+)?(.+)", re.IGNORECASE),
)
_NAME_QUESTIONS = (
    "jak mam na imię",
    "jak się nazywam",
    "pamiętasz moje imię",
    "what is my name",
    "what's my name",
    "do you remember my name",
)
_LOCATION_QUESTIONS = ("gdzie mieszkam", "where do i live")
_AGE_QUESTIONS = ("ile mam lat", "how old am i")
_MAX_PLAUSIBLE_AGE_EXCLUSIVE = 130
_STOP_WORDS = {
    "a",
    "aby",
    "ale",
    "and",
    "że",
    "the",
    "to",
    "jest",
    "mam",
    "mój",
    "moja",
    "moje",
    "my",
    "remember",
    "zapamiętaj",
}


def _proposal(
    *,
    producer: ProducerIdentity,
    turn_id: str,
    kind: str,
    topic: str,
    content: dict[str, object],
) -> MemoryProposal:
    return MemoryProposal(
        record_id=_new_id(),
        timestamp=_utc_now(),
        producer=producer,
        source_turn_id=turn_id,
        kind=kind,
        topic=topic,
        content=content,
    )


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" \t\n.,!?;:")
    return None


def _fact_topic(fact: str) -> str:
    normalized = unicodedata.normalize("NFKD", fact).encode("ascii", "ignore").decode()
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalized.lower())
        if len(word) > 1 and word not in _STOP_WORDS
    ]
    suffix = ".".join(words[:4]) or "note"
    return f"user.fact.{suffix[:80]}"


def extract_memory_proposals(
    user_text: str,
    turn_id: str,
    producer: ProducerIdentity,
) -> tuple[MemoryProposal, ...]:
    """Extract a small, transparent set of user facts without trusting model prose."""
    proposals: list[MemoryProposal] = []

    name = _first_match(_NAME_PATTERNS, user_text)
    if name:
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic="user.identity.name",
                content={"subject": "user", "attribute": "name", "value": name},
            )
        )

    location = _first_match(_LOCATION_PATTERNS, user_text)
    if location:
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic="user.location.home",
                content={
                    "subject": "user",
                    "attribute": "home_location",
                    "value": location,
                },
            )
        )

    age = _first_match(_AGE_PATTERNS, user_text)
    if age and 0 < int(age) < _MAX_PLAUSIBLE_AGE_EXCLUSIVE:
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic="user.identity.age",
                content={"subject": "user", "attribute": "age", "value": int(age)},
            )
        )

    lowered = user_text.casefold()
    language = None
    if "guile" in lowered:
        language = "Guile"
    elif "python" in lowered or "pythona" in lowered:
        language = "Python"
    if language and any(
        marker in lowered
        for marker in ("preferuj", "preferuję", "zapamiętaj", "używaj", "prefer")
    ):
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="preference",
                topic="code_examples",
                content={
                    "topic": "code_examples",
                    "preference": language,
                    "raw_statement": user_text,
                },
            )
        )

    explicit_fact = _first_match(_EXPLICIT_FACT_PATTERNS, user_text)
    claimed_topics = {proposal.topic for proposal in proposals}
    if explicit_fact and not claimed_topics:
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic=_fact_topic(explicit_fact),
                content={"subject": "user", "fact": explicit_fact},
            )
        )

    return tuple(proposals)


def memory_value(
    durable_memories: tuple[object, ...] | list[object], topic: str
) -> object | None:
    """Return the current value for an exact durable-memory topic."""
    for memory in durable_memories:
        if not isinstance(memory, dict) or memory.get("topic") != topic:
            continue
        content = memory.get("content", {})
        if isinstance(content, dict):
            if "value" in content:
                return content["value"]
            if "preference" in content:
                return content["preference"]
    return None


def grounded_memory_response(  # noqa: PLR0911
    user_text: str,
    durable_memories: tuple[object, ...] | list[object],
    proposals: tuple[MemoryProposal, ...],
    model_text: str,
) -> tuple[str, bool]:
    """Ground critical personal-fact answers and deterministic acknowledgements."""
    proposed = {proposal.topic: proposal for proposal in proposals}
    if name_proposal := proposed.get("user.identity.name"):
        name = name_proposal.content.get("value")
        return (
            f"Miło Cię poznać, {name}. Jestem RAI, lokalnym asystentem. Zapamiętałem Twoje imię.",
            True,
        )
    if location_proposal := proposed.get("user.location.home"):
        location = location_proposal.content.get("value")
        return f"Zapamiętałem, że mieszkasz w {location}.", True
    if age_proposal := proposed.get("user.identity.age"):
        age = age_proposal.content.get("value")
        return f"Zapamiętałem, że masz {age} lat.", True
    if preference_proposal := proposed.get("code_examples"):
        language = preference_proposal.content.get("preference")
        return (
            f"Zapamiętałem: w przykładach kodu będę preferować język {language}.",
            True,
        )
    if proposals:
        return "Zapamiętałem tę informację.", True

    lowered = user_text.casefold()
    if any(question in lowered for question in _NAME_QUESTIONS):
        name = memory_value(durable_memories, "user.identity.name")
        return (
            f"Masz na imię {name}."
            if name
            else "Nie mam jeszcze zapisanego Twojego imienia.",
            True,
        )
    if any(question in lowered for question in _LOCATION_QUESTIONS):
        location = memory_value(durable_memories, "user.location.home")
        return (
            f"Według zapisanej informacji mieszkasz w {location}."
            if location
            else "Nie mam jeszcze zapisanej informacji o tym, gdzie mieszkasz.",
            True,
        )
    if any(question in lowered for question in _AGE_QUESTIONS):
        age = memory_value(durable_memories, "user.identity.age")
        return (
            f"Według zapisanej informacji masz {age} lat."
            if age is not None
            else "Nie mam jeszcze zapisanej informacji o Twoim wieku.",
            True,
        )

    asks_for_language = "jakim języku" in lowered or "w jakim języku" in lowered
    if asks_for_language:
        language = memory_value(durable_memories, "code_examples")
        return (
            "Zgodnie z Twoją zapisaną preferencją, powinienem pokazywać "
            f"przykłady kodu w języku {language}."
            if language
            else "Nie mam zapisanej preferencji dotyczącej języka w przykładach kodu.",
            True,
        )

    return model_text, False
