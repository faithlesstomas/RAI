"""Deterministic, auditable extraction and grounding for durable user memory."""

from __future__ import annotations

import json
import re
import unicodedata

from rai.kernel.records import ProducerIdentity, _new_id, _utc_now

from .records import MemoryOperationKind, MemoryProposal

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
_FORGET_PATTERNS = (
    re.compile(
        r"\b(?:zapomnij|usuń z pamięci|nie pamiętaj)(?: proszę)?[, :]*(?:że|o)?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bforget(?: please)?[, :]*(?:that|about)?\s*(.+)", re.IGNORECASE),
)
_FORGET_ALL_PATTERNS = (
    re.compile(
        r"\b(?:zapomnij|wyczyść|usuń)\s+(?:całą|cała|wszystko z)\s+pamię(?:ć|ci)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:forget|clear|delete)\s+(?:all|everything)(?: from)? memory\b",
        re.IGNORECASE,
    ),
)
_ATTRIBUTE_PATTERNS = (
    re.compile(
        r"\b(?P<label>mój|moja|moje)\s+(?P<attribute>[\wąćęłńóśźż -]{2,64}?)\s+"
        r"(?:to|jest|są)\s+(?P<value>[^.!?\n]{1,160})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?P<attribute>[a-z][a-z0-9 _-]{1,64}?)\s+"
        r"(?:is|are)\s+(?P<value>[^.!?\n]{1,160})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
)
_PREFERENCE_PATTERNS = (
    re.compile(
        r"\b(?P<verb>lubię|uwielbiam|wolę|preferuję)\s+(?P<value>[^.!?\n]{1,160})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<verb>i like|i love|i prefer)\s+(?P<value>[^.!?\n]{1,160})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
)
_PLAN_PATTERNS = (
    re.compile(
        r"\b(?P<verb>planuję|zamierzam)\s+(?P<value>[^.!?\n]{2,200})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<verb>i plan to|i intend to)\s+(?P<value>[^.!?\n]{2,200})(?=[.!?]|$)",
        re.IGNORECASE,
    ),
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
_GROUNDING_STOP_WORDS = _STOP_WORDS | {
    "czy",
    "co",
    "do",
    "i",
    "jak",
    "jaki",
    "jaka",
    "jakie",
    "jakim",
    "me",
    "na",
    "o",
    "sie",
    "się",
    "w",
    "you",
}


def _proposal(  # noqa: PLR0913
    *,
    producer: ProducerIdentity,
    turn_id: str,
    kind: str,
    topic: str,
    content: dict[str, object],
    operation: MemoryOperationKind = MemoryOperationKind.REMEMBER,
    source_span: str | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    statement_type: str = "assertion",
    modality: str = "direct",
    negated: bool = False,
    confidence: float = 1.0,
    target_topic: str | None = None,
) -> MemoryProposal:
    domain_scope = (
        "software"
        if topic == "code_examples"
        else "project"
        if topic.startswith("project.")
        else "personal"
        if topic.startswith("user.")
        else "global"
    )
    return MemoryProposal(
        record_id=_new_id(),
        timestamp=_utc_now(),
        producer=producer,
        source_turn_id=turn_id,
        operation=operation,
        kind=kind,
        topic=topic,
        target_topic=target_topic,
        content=content,
        source_span=source_span,
        span_start=span_start,
        span_end=span_end,
        statement_type=statement_type,
        modality=modality,
        negated=negated,
        confidence=confidence,
        domain_scope=domain_scope,
    )


def _fact_topic(fact: str) -> str:
    normalized = unicodedata.normalize("NFKD", fact).encode("ascii", "ignore").decode()
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalized.lower())
        if len(word) > 1 and word not in _STOP_WORDS
    ]
    suffix = ".".join(words[:4]) or "note"
    return f"user.fact.{suffix[:80]}"


def _slug(value: str, default: str = "note") -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    words = [word for word in re.findall(r"[a-z0-9]+", normalized.casefold()) if word]
    return ".".join(words[:6])[:80] or default


def _modality(text: str, start: int, end: int) -> tuple[str, bool, float]:
    """Classify obvious quotation, hearsay, uncertainty and negation signals."""
    lowered = text.casefold()
    span = text[start:end]
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    quoted = (
        before.endswith(('"', "„", "“", "'")) and after.startswith(('"', "”", "'"))
    ) or span.startswith(('"', "„", "“"))
    if quoted:
        return "quoted", False, 0.2
    if any(
        marker in lowered
        for marker in ("ktoś powiedział", "podobno", "someone said", "apparently")
    ):
        return "hearsay", False, 0.35
    if any(
        marker in lowered
        for marker in ("chyba", "wydaje mi się", "może", "i think", "maybe", "probably")
    ):
        return "hedged", False, 0.55
    negated = bool(
        re.search(r"\b(?:nie|nigdy|not|never|don'?t|do not)\b", span, re.IGNORECASE)
    )
    return "direct", negated, 0.9 if negated else 1.0


def _span_kwargs(text: str, match: re.Match[str]) -> dict[str, object]:
    return _text_span_kwargs(text, match.start(), match.end())


def _text_span_kwargs(text: str, start: int, end: int) -> dict[str, object]:
    modality, negated, confidence = _modality(text, start, end)
    return {
        "source_span": text[start:end],
        "span_start": start,
        "span_end": end,
        "modality": modality,
        "negated": negated,
        "confidence": confidence,
    }


def _match_details(
    patterns: tuple[re.Pattern[str], ...], text: str
) -> tuple[str, re.Match[str]] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip(" \t\n.,!?;:"), match
    return None


def extract_memory_proposals(  # noqa: PLR0912
    user_text: str,
    turn_id: str,
    producer: ProducerIdentity,
) -> tuple[MemoryProposal, ...]:
    """Extract bounded evidence-preserving proposals from ordinary conversation."""
    proposals: list[MemoryProposal] = []

    if any(pattern.search(user_text) for pattern in _FORGET_ALL_PATTERNS):
        return (
            _proposal(
                producer=producer,
                turn_id=turn_id,
                operation=MemoryOperationKind.FORGET,
                kind="control",
                topic="*",
                target_topic="*",
                content={"query": "*"},
                source_span=user_text,
                span_start=0,
                span_end=len(user_text),
                statement_type="request",
            ),
        )

    forget = _match_details(_FORGET_PATTERNS, user_text)
    if forget:
        query_text, match = forget
        return (
            _proposal(
                producer=producer,
                turn_id=turn_id,
                operation=MemoryOperationKind.FORGET,
                kind="control",
                topic="memory.forget",
                content={"query": query_text},
                source_span=match.group(0),
                span_start=match.start(),
                span_end=match.end(),
                statement_type="request",
            ),
        )

    name_match = _match_details(_NAME_PATTERNS, user_text)
    if name_match:
        name, match = name_match
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic="user.identity.name",
                content={"subject": "user", "attribute": "name", "value": name},
                **_span_kwargs(user_text, match),
            )
        )

    location_match = _match_details(_LOCATION_PATTERNS, user_text)
    if location_match:
        location, match = location_match
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
                **_span_kwargs(user_text, match),
            )
        )

    age_match = _match_details(_AGE_PATTERNS, user_text)
    if age_match and 0 < int(age_match[0]) < _MAX_PLAUSIBLE_AGE_EXCLUSIVE:
        age, match = age_match
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic="user.identity.age",
                content={"subject": "user", "attribute": "age", "value": int(age)},
                **_span_kwargs(user_text, match),
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
                statement_type="preference",
                **_text_span_kwargs(user_text, 0, len(user_text)),
            )
        )

    explicit_match = _match_details(_EXPLICIT_FACT_PATTERNS, user_text)
    explicit_fact = explicit_match[0] if explicit_match else None
    claimed_topics = {proposal.topic for proposal in proposals}
    if explicit_fact and not claimed_topics:
        match = explicit_match[1]
        proposals.append(
            _proposal(
                producer=producer,
                turn_id=turn_id,
                kind="fact",
                topic=_fact_topic(explicit_fact),
                content={"subject": "user", "fact": explicit_fact},
                statement_type="request",
                **_span_kwargs(user_text, match),
            )
        )

    claimed_topics = {proposal.topic for proposal in proposals}
    if not claimed_topics:
        for pattern in _ATTRIBUTE_PATTERNS:
            if match := pattern.search(user_text):
                attribute = match.group("attribute").strip()
                value = match.group("value").strip(" \t,;:")
                proposals.append(
                    _proposal(
                        producer=producer,
                        turn_id=turn_id,
                        kind="fact",
                        topic=f"user.attribute.{_slug(attribute)}",
                        content={
                            "subject": "user",
                            "attribute": attribute,
                            "value": value,
                        },
                        **_span_kwargs(user_text, match),
                    )
                )
                break

    if not proposals:
        for pattern in _PREFERENCE_PATTERNS:
            if match := pattern.search(user_text):
                value = match.group("value").strip(" \t,;:")
                proposals.append(
                    _proposal(
                        producer=producer,
                        turn_id=turn_id,
                        kind="preference",
                        topic=f"user.preference.{_slug(value)}",
                        content={"subject": "user", "preference": value},
                        statement_type="preference",
                        **_span_kwargs(user_text, match),
                    )
                )
                break

    if not proposals:
        for pattern in _PLAN_PATTERNS:
            if match := pattern.search(user_text):
                value = match.group("value").strip(" \t,;:")
                proposals.append(
                    _proposal(
                        producer=producer,
                        turn_id=turn_id,
                        kind="plan",
                        topic=f"user.plan.{_slug(value)}",
                        content={"subject": "user", "plan": value},
                        statement_type="plan",
                        **_span_kwargs(user_text, match),
                    )
                )
                break

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


def _grounding_terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return {
        word
        for word in re.findall(r"[a-z0-9]+", normalized.casefold())
        if len(word) > 1 and word not in _GROUNDING_STOP_WORDS
    }


def _relevant_generic_memory(
    user_text: str, durable_memories: tuple[object, ...] | list[object]
) -> dict[object, object] | None:
    """Select a memory only when its semantic payload overlaps the question."""
    query_terms = _grounding_terms(user_text)
    ranked: list[tuple[int, dict[object, object]]] = []
    for item in durable_memories:
        if not isinstance(item, dict):
            continue
        content = item.get("content", {})
        semantic_text = f"{item.get('topic', '')} {json.dumps(content, ensure_ascii=False)}"
        overlap = query_terms & _grounding_terms(semantic_text)
        if overlap:
            ranked.append((len(overlap), item))
    return max(ranked, key=lambda candidate: candidate[0])[1] if ranked else None


def grounded_memory_response(  # noqa: PLR0911, PLR0912
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
        if all(
            proposal.operation == MemoryOperationKind.FORGET.value
            for proposal in proposals
        ):
            return model_text, False
        if any(proposal.statement_type == "request" for proposal in proposals):
            return "Zapamiętałem tę informację.", True
        return f"{model_text.rstrip()} Zapamiętałem tę informację na przyszłość.", True

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

    question_markers = (
        "czy pamiętasz",
        "co wiesz",
        "jaki",
        "jaka",
        "jakie",
        "what",
        "which",
        "do you remember",
    )
    if durable_memories and (
        "?" in user_text or any(marker in lowered for marker in question_markers)
    ):
        memory = _relevant_generic_memory(user_text, durable_memories)
        content = memory.get("content", {}) if memory else {}
        if isinstance(content, dict):
            if "attribute" in content and "value" in content:
                return (
                    "Według zapisanej informacji "
                    f"{content['attribute']} to {content['value']}.",
                    True,
                )
            if "predicate" in content and "value" in content:
                return (
                    "Według zapisanej informacji "
                    f"{content['predicate']}: {content['value']}.",
                    True,
                )
            if preference := content.get("preference"):
                return f"Mam zapisaną preferencję: {preference}.", True
            if plan := content.get("plan"):
                return f"Mam zapisany plan: {plan}.", True
            if fact := content.get("fact"):
                return f"Mam zapisaną informację: {fact}.", True

    return model_text, False
