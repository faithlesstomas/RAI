"""Deterministic choice resolution over active selection frames.

Resolves user selections against a typed ActiveChoiceFrame without calling
a language model. Operates as a pure function testable in isolation.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from returns.result import Failure, Result, Success
from pydantic import ValidationError

from rai.assistant.records import ConversationTurn
from rai.kernel.records import DataClass, ProducerIdentity, _new_id, _utc_now
from .records import ActionProposal, ActiveChoiceFrame, ChoiceEntry, ChoiceState

ORDINAL_MAP: dict[str, int] = {
    "pierwszy": 1,
    "pierwsza": 1,
    "pierwsze": 1,
    "pierwszą": 1,
    "first": 1,
    "drugi": 2,
    "druga": 2,
    "drugie": 2,
    "drugą": 2,
    "second": 2,
    "trzeci": 3,
    "trzecia": 3,
    "trzecie": 3,
    "trzecią": 3,
    "third": 3,
    "czwarty": 4,
    "czwarta": 4,
    "czwarte": 4,
    "czwartą": 4,
    "fourth": 4,
    "piąty": 5,
    "piąta": 5,
    "piąte": 5,
    "piątą": 5,
    "fifth": 5,
}

SELECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Direct number: "5", "5.", "#5", "nr 5", "pozycja 5", "opcja 5"
    re.compile(
        r"^(?:nr|no|pozycja|opcja|numer|punkt|wybór|#)?\s*([1-9][0-9]?)\.?$",
        re.IGNORECASE,
    ),
    # Verb + number: "otwórz 5", "wybierz 2", "uruchom 1", "read 3"
    re.compile(
        r"^(?P<verb>otwórz|uruchom|wybierz|przeczytaj|pokaż|open|launch|read|select|choose|show)\s+"
        r"(?:nr|no|pozycja|opcja|numer|punkt|opcję|pozycję|#)?\s*([1-9][0-9]?)\.?$",
        re.IGNORECASE,
    ),
)


def extract_selection_index(  # noqa: PLR0911
    text: str, total_entries: int
) -> tuple[int, str | None] | None:
    """Extract a 1-based index and optional explicit verb from user text."""
    normalized = text.strip().casefold()
    if not normalized:
        return None

    # Check ordinal expressions: e.g. "pierwszy", "wybierz drugi", "ostatni"
    tokens = normalized.split()
    if len(tokens) == 1:
        if tokens[0] in {"ostatni", "ostatnia", "ostatnie", "ostatnią", "last"}:
            return total_entries, None
        if tokens[0] in ORDINAL_MAP:
            return ORDINAL_MAP[tokens[0]], None
    elif len(tokens) == 2 and tokens[0] in {  # noqa: PLR2004
        "wybierz",
        "otwórz",
        "przeczytaj",
        "uruchom",
        "select",
        "open",
        "read",
        "launch",
    }:
        verb = tokens[0]
        if tokens[1] in {"ostatni", "ostatnia", "ostatnie", "ostatnią", "last"}:
            return total_entries, verb
        if tokens[1] in ORDINAL_MAP:
            return ORDINAL_MAP[tokens[1]], verb

    for pattern in SELECTION_PATTERNS:
        match = pattern.match(normalized)
        if match:
            verb = match.groupdict().get("verb")
            idx = int(match.groups()[-1])
            return idx, verb

    return None


def resolve_choice_selection(  # noqa: PLR0913
    text: str,
    frame: ActiveChoiceFrame | None,
    now: datetime,
    *,
    actor: ProducerIdentity | None = None,
    source_turn_id: str | None = None,
    data_class: DataClass = DataClass.LOCAL,
) -> Result[ActionProposal, str]:
    """Pure deterministic resolver matching a user choice against an active frame.

    Does not perform I/O or invoke language models.
    """
    if frame is None:
        return Failure("NO_ACTIVE_FRAME")
    state_value = frame.state.value if isinstance(frame.state, ChoiceState) else str(frame.state)
    if state_value != ChoiceState.ACTIVE.value:
        return Failure(f"FRAME_NOT_ACTIVE_{state_value}")
    if now >= frame.expires_at:
        return Failure("FRAME_EXPIRED")

    extracted = extract_selection_index(text, len(frame.entries))
    if extracted is None:
        return Failure("NO_SELECTION_MATCH")

    index, verb = extracted
    if index < 1 or index > len(frame.entries):
        return Failure("SELECTION_OUT_OF_RANGE")

    selected_entry = frame.entries[index - 1]

    # Resolve capability from frame and optional user verb
    capability = frame.intended_capability
    if verb in {"przeczytaj", "read"} and capability in {
        "browser.open_result",
        "browser.read_page",
    }:
        capability = "browser.read_page"
    elif verb in {"otwórz", "open"} and capability in {
        "browser.open_result",
        "browser.read_page",
    }:
        capability = "browser.open_result"
    elif verb in {"uruchom", "launch"} and capability == "application.launch":
        capability = "application.launch"

    proposal = ActionProposal(
        record_id=_new_id(),
        timestamp=now,
        producer=actor or frame.producer,
        source_turn_id=source_turn_id or frame.source_turn_id,
        task_id=frame.task_id,
        capability=capability,
        resource_handle=selected_entry.handle,
        data_class=data_class,
    )
    return Success(proposal)


def find_active_choice_frame(  # noqa: PLR0912
    recent_turns: tuple[ConversationTurn, ...],
    profile: str,
    now: datetime,
) -> ActiveChoiceFrame | None:
    """Find the most recent non-expired, non-consumed ActiveChoiceFrame."""
    consumed_ids: set[str] = set()

    # Pass 1: gather consumed frame IDs from turns
    for turn in reversed(recent_turns):
        if turn.role == "assistant":
            consumed_id = turn.metadata.get("consumed_choice_frame_id")
            if not isinstance(consumed_id, str) and isinstance(turn.metadata.get("action"), dict):
                consumed_id = turn.metadata["action"].get("consumed_choice_frame_id")
            if isinstance(consumed_id, str):
                consumed_ids.add(consumed_id)

    # Pass 2: find latest valid active frame
    for turn in reversed(recent_turns):
        if turn.role != "assistant":
            continue
        turn_profile = turn.metadata.get("profile_scope") or "default"
        if turn_profile != profile:
            continue

        raw_frame = turn.metadata.get("active_choice_frame")
        if not isinstance(raw_frame, dict) and isinstance(turn.metadata.get("action"), dict):
            raw_frame = turn.metadata["action"].get("active_choice_frame")

        if isinstance(raw_frame, dict):
            try:
                frame = ActiveChoiceFrame.model_validate(raw_frame)
            except (ValidationError, ValueError, TypeError, KeyError):
                continue

            if frame.record_id in consumed_ids:
                return None
            if frame.state != ChoiceState.ACTIVE:
                return None
            if now >= frame.expires_at:
                return None
            return frame

    return None


def find_latest_choice_frame(
    recent_turns: tuple[ConversationTurn, ...],
    profile: str,
) -> ActiveChoiceFrame | None:
    """Find the most recent ActiveChoiceFrame regardless of active/consumed lifecycle state."""
    for turn in reversed(recent_turns):
        if turn.role != "assistant":
            continue
        turn_profile = turn.metadata.get("profile_scope") or "default"
        if turn_profile != profile:
            continue

        raw_frame = turn.metadata.get("active_choice_frame")
        if not isinstance(raw_frame, dict) and isinstance(turn.metadata.get("action"), dict):
            raw_frame = turn.metadata["action"].get("active_choice_frame")

        if isinstance(raw_frame, dict):
            try:
                return ActiveChoiceFrame.model_validate(raw_frame)
            except (ValidationError, ValueError, TypeError, KeyError):
                continue

    return None

