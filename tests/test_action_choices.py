# ruff: noqa: PLR2004, PLR0915
"""Unit tests for deterministic choice resolution over active selection frames."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from returns.result import Failure, Success

from rai.actions.choices import (
    extract_selection_index,
    find_active_choice_frame,
    resolve_choice_selection,
)
from rai.actions.records import ActiveChoiceFrame, ChoiceEntry, ChoiceState
from rai.assistant.records import ConversationTurn
from rai.kernel.records import DataClass, ProducerIdentity, _new_id, _utc_now


def _make_frame(
    *,
    count: int = 5,
    state: ChoiceState = ChoiceState.ACTIVE,
    expires_delta: timedelta = timedelta(minutes=10),
    intended_capability: str = "browser.open_result",
) -> ActiveChoiceFrame:
    now = _utc_now()
    producer = ProducerIdentity(producer_id="test", kind="service", version="1.0.0")
    entries = tuple(
        ChoiceEntry(
            index=i,
            handle=f"handle-{i}",
            label=f"Item {i}",
            metadata={"url": f"https://example.com/{i}"},
        )
        for i in range(1, count + 1)
    )
    return ActiveChoiceFrame(
        record_id=_new_id(),
        timestamp=now,
        producer=producer,
        session_id="test-session",
        profile_scope="default",
        task_id="task-123",
        source_turn_id="turn-1",
        intended_capability=intended_capability,
        entries=entries,
        state=state,
        expires_at=now + expires_delta,
        data_class=DataClass.PUBLIC,
    )


def test_extract_selection_index() -> None:
    assert extract_selection_index("5", 5) == (5, None)
    assert extract_selection_index("5.", 5) == (5, None)
    assert extract_selection_index("#5", 5) == (5, None)
    assert extract_selection_index("pozycja 3", 5) == (3, None)
    assert extract_selection_index("opcja 2", 5) == (2, None)
    assert extract_selection_index("nr 1", 5) == (1, None)
    assert extract_selection_index("otwórz 4", 5) == (4, "otwórz")
    assert extract_selection_index("przeczytaj 2", 5) == (2, "przeczytaj")
    assert extract_selection_index("read 3", 5) == (3, "read")
    assert extract_selection_index("pierwszy", 5) == (1, None)
    assert extract_selection_index("ostatni", 5) == (5, None)
    assert extract_selection_index("wybierz drugi", 5) == (2, "wybierz")
    assert extract_selection_index("Co to znaczy?", 5) is None
    assert extract_selection_index("wyszukaj więcej", 5) is None


def test_resolve_choice_selection_happy_path() -> None:
    now = _utc_now()
    frame = _make_frame(count=5, intended_capability="browser.open_result")

    # Plain index
    res = resolve_choice_selection("5", frame, now)
    assert isinstance(res, Success)
    proposal = res.unwrap()
    assert proposal.capability == "browser.open_result"
    assert proposal.resource_handle == "handle-5"

    # Explicit read verb overrides to browser.read_page
    res_read = resolve_choice_selection("przeczytaj 2", frame, now)
    assert isinstance(res_read, Success)
    assert res_read.unwrap().capability == "browser.read_page"
    assert res_read.unwrap().resource_handle == "handle-2"


def test_resolve_choice_selection_out_of_range() -> None:
    now = _utc_now()
    frame = _make_frame(count=5)

    res_zero = resolve_choice_selection("0", frame, now)
    assert isinstance(res_zero, Failure)

    res_six = resolve_choice_selection("6", frame, now)
    assert isinstance(res_six, Failure)
    assert res_six.failure() == "SELECTION_OUT_OF_RANGE"


def test_resolve_choice_selection_expired() -> None:
    now = _utc_now()
    past = now - timedelta(minutes=15)
    producer = ProducerIdentity(producer_id="test", kind="service", version="1.0.0")
    entries = (
        ChoiceEntry(index=1, handle="handle-1", label="Item 1"),
    )
    frame = ActiveChoiceFrame(
        record_id=_new_id(),
        timestamp=past,
        producer=producer,
        session_id="test-session",
        profile_scope="default",
        task_id="task-123",
        source_turn_id="turn-1",
        intended_capability="browser.open_result",
        entries=entries,
        state=ChoiceState.ACTIVE,
        expires_at=past + timedelta(minutes=5),  # expired 10 minutes ago
        data_class=DataClass.PUBLIC,
    )
    res = resolve_choice_selection("1", frame, now)
    assert isinstance(res, Failure)
    assert res.failure() == "FRAME_EXPIRED"


def test_resolve_choice_selection_inactive_state() -> None:
    now = _utc_now()
    frame = _make_frame(state=ChoiceState.CONSUMED)
    res = resolve_choice_selection("1", frame, now)
    assert isinstance(res, Failure)
    assert res.failure() == "FRAME_NOT_ACTIVE_CONSUMED"


def test_find_active_choice_frame_persists_across_intermediate_turns() -> None:
    now = _utc_now()
    frame = _make_frame(count=5)
    producer = ProducerIdentity(producer_id="test", kind="service", version="1.0.0")

    turn1_assistant = ConversationTurn(
        record_id="turn-1",
        timestamp=now - timedelta(minutes=2),
        producer=producer,
        session_id="test-session",
        role="assistant",
        text="Wyniki wyszukiwania:\n1. ...\n2. ...",
        metadata={
            "profile_scope": "default",
            "active_choice_frame": frame.model_dump(mode="json"),
        },
    )
    turn2_user = ConversationTurn(
        record_id="turn-2",
        timestamp=now - timedelta(minutes=1),
        producer=producer,
        session_id="test-session",
        role="user",
        text="Co to znaczy? Poproszę więcej informacji.",
    )
    turn2_assistant = ConversationTurn(
        record_id="turn-3",
        timestamp=now - timedelta(seconds=30),
        producer=producer,
        session_id="test-session",
        role="assistant",
        text="Odniosłeś się do kontekstu...",
        metadata={"profile_scope": "default"},
    )

    turns = (turn1_assistant, turn2_user, turn2_assistant)
    found = find_active_choice_frame(turns, "default", now)
    assert found is not None
    assert found.record_id == frame.record_id
    assert len(found.entries) == 5

    # If marked consumed in turn 4
    turn3_assistant = ConversationTurn(
        record_id="turn-4",
        timestamp=now,
        producer=producer,
        session_id="test-session",
        role="assistant",
        text="Otworzyłem stronę.",
        metadata={
            "profile_scope": "default",
            "consumed_choice_frame_id": frame.record_id,
        },
    )
    turns_consumed = (*turns, turn3_assistant)
    assert find_active_choice_frame(turns_consumed, "default", now) is None
