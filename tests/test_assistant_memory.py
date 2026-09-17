"""Tests for deterministic user-fact extraction and grounded recall."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from returns.result import Result, Success

from rai.assistant.backends.local import LocalAssistantBackend
from rai.assistant.records import ConversationTurn
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.inference.protocols import GenerationStats, InferenceResult
from rai.kernel.records import ProducerIdentity

PRODUCER = ProducerIdentity(producer_id="memory-test", kind="test", version="1.0.0")


class _ChatEngine:
    async def load(self) -> Result[None, Exception]:
        return Success(None)

    async def unload(self) -> Result[None, Exception]:
        return Success(None)

    async def generate(
        self,
        prompt: str,
        **_kwargs: Any,  # noqa: ANN401
    ) -> Result[InferenceResult, Exception]:
        return Success(
            InferenceResult(
                text="Model response.",
                stats=GenerationStats(len(prompt.split()), 2, 0.01, 200.0),
            )
        )


def _turn(turn_id: str, session_id: str, text: str) -> ConversationTurn:
    return ConversationTurn(
        record_id=turn_id,
        producer=PRODUCER,
        session_id=session_id,
        role="user",
        text=text,
    )


@pytest.mark.asyncio
async def test_name_is_recalled_after_restart_and_corrected(tmp_path: Path) -> None:
    database = tmp_path / "assistant.sqlite3"
    service_a = AssistantService(
        store=SQLiteMemoryGraphStore(database),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service_a.start()

    admission = await service_a.accept_turn(
        _turn("name-1", "session-a", "Jestem Tomek, a Ty?")
    )
    assert isinstance(admission, Success)
    assert "Zapamiętałem Twoje imię" in admission.unwrap().text
    assert len(admission.unwrap().admitted_memory_ids) == 1
    await service_a.stop()

    service_b = AssistantService(
        store=SQLiteMemoryGraphStore(database),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service_b.start()
    recall = await service_b.accept_turn(
        _turn("name-2", "session-b", "Jak mam na imię?")
    )
    assert isinstance(recall, Success)
    assert recall.unwrap().text == "Masz na imię Tomek."

    correction = await service_b.accept_turn(
        _turn("name-3", "session-b", "Mam na imię Tomasz.")
    )
    assert isinstance(correction, Success)
    assert len(correction.unwrap().admitted_memory_ids) == 1

    corrected_recall = await service_b.accept_turn(
        _turn("name-4", "session-c", "Jak się nazywam?")
    )
    assert isinstance(corrected_recall, Success)
    assert corrected_recall.unwrap().text == "Masz na imię Tomasz."
    await service_b.stop()


@pytest.mark.asyncio
async def test_recent_unpersisted_fact_can_ground_the_next_answer(
    tmp_path: Path,
) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "recent-grounding.sqlite3"),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service.start()
    old_statement = ConversationTurn(
        record_id="recent-old-name-statement",
        producer=PRODUCER,
        session_id="recent-session",
        role="user",
        text="Mam na imię Tomek.",
        status="COMPLETED",
        domain_scope="personal",
        metadata={"profile_scope": "default"},
    )
    assert isinstance(await service.store.accept_turn(old_statement), Success)
    recent_statement = ConversationTurn(
        record_id="recent-name-statement",
        producer=PRODUCER,
        session_id="recent-session",
        role="user",
        text="Mam na imię Tomasz.",
        status="COMPLETED",
        domain_scope="personal",
        metadata={"profile_scope": "default"},
        reply_to_turn_id=old_statement.record_id,
    )
    assert isinstance(await service.store.accept_turn(recent_statement), Success)

    recalled = await service.accept_turn(
        _turn("recent-name-question", "recent-session", "Jak mam na imię?")
    )

    assert isinstance(recalled, Success)
    assert recalled.unwrap().text == "Masz na imię Tomasz."
    memories = await service.store.retrieve_relevant_memories(limit=10)
    assert isinstance(memories, Success)
    assert memories.unwrap() == ()
    await service.stop()


@pytest.mark.asyncio
async def test_recent_location_correction_grounds_intervening_question_word(
    tmp_path: Path,
) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "recent-location-correction.sqlite3"),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service.start()
    old_statement = ConversationTurn(
        record_id="recent-old-location",
        producer=PRODUCER,
        session_id="recent-location-session",
        role="user",
        text="Mieszkam w Gdańsku.",
        status="COMPLETED",
        domain_scope="personal",
        metadata={"profile_scope": "default"},
    )
    assert isinstance(await service.store.accept_turn(old_statement), Success)
    correction = ConversationTurn(
        record_id="recent-new-location",
        producer=PRODUCER,
        session_id="recent-location-session",
        role="user",
        text="Przeprowadziłem się i teraz mieszkam w Warszawie.",
        status="COMPLETED",
        domain_scope="personal",
        metadata={"profile_scope": "default"},
        reply_to_turn_id=old_statement.record_id,
    )
    assert isinstance(await service.store.accept_turn(correction), Success)

    recalled = await service.accept_turn(
        _turn(
            "recent-location-question",
            "recent-location-session",
            "Gdzie teraz mieszkam?",
        )
    )

    assert isinstance(recalled, Success)
    assert recalled.unwrap().text == (
        "Według zapisanej informacji mieszkasz w Warszawie."
    )
    assert "Gdańsk" not in recalled.unwrap().text
    await service.stop()


@pytest.mark.asyncio
async def test_uncertain_recent_statement_does_not_become_grounding(
    tmp_path: Path,
) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "uncertain-grounding.sqlite3"),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service.start()
    uncertain = ConversationTurn(
        record_id="uncertain-name-statement",
        producer=PRODUCER,
        session_id="uncertain-session",
        role="user",
        text="Chyba mam na imię Tomasz.",
        status="COMPLETED",
        domain_scope="personal",
        metadata={"profile_scope": "default"},
    )
    assert isinstance(await service.store.accept_turn(uncertain), Success)

    recalled = await service.accept_turn(
        _turn("uncertain-name-question", "uncertain-session", "Jak mam na imię?")
    )

    assert isinstance(recalled, Success)
    assert recalled.unwrap().text == "Nie mam jeszcze zapisanego Twojego imienia."
    await service.stop()


@pytest.mark.asyncio
async def test_location_age_and_explicit_fact_are_admitted(tmp_path: Path) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3"),
        backend=LocalAssistantBackend(_ChatEngine(), model_name="fake-chat"),
    )
    await service.start()

    inputs = (
        ("location", "Mieszkam w Warszawie."),
        ("age", "Mam 42 lata."),
        ("fact", "Zapamiętaj, że mój ulubiony kolor to zielony."),
    )
    for turn_id, text in inputs:
        response = await service.accept_turn(_turn(turn_id, "session", text))
        assert isinstance(response, Success)
        assert len(response.unwrap().admitted_memory_ids) == 1

    memories = await service.store.retrieve_relevant_memories(limit=10)
    assert isinstance(memories, Success)
    topics = {memory.topic for memory, _reason in memories.unwrap()}
    assert "user.location.home" in topics
    assert "user.identity.age" in topics
    assert any(topic.startswith("user.fact.") for topic in topics)
    await service.stop()
