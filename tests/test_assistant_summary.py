"""Grounding and privacy tests for deterministic memory summaries."""

from __future__ import annotations

from dataclasses import replace

import pytest
from returns.result import Result, Success

from rai.assistant.ports import MemoryQuery
from rai.assistant.records import ConversationTurn, MemoryRecord
from rai.assistant.summary import (
    GroundedClaimSummaryProvider,
    validate_grounded_summary,
)
from rai.kernel.records import ActionFailure, DataClass, ProducerIdentity

PRODUCER = ProducerIdentity(producer_id="summary-test", kind="test", version="1.0.0")


class _SummaryStore:
    def __init__(self, memory: MemoryRecord, turn: ConversationTurn | None) -> None:
        self.memory = memory
        self.turn = turn

    async def retrieve_relevant_memories(
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]:
        del profile_scope, query, data_classes, limit
        return Success(((self.memory, "test claim"),))

    async def get_turn(
        self, turn_id: str
    ) -> Result[ConversationTurn | None, ActionFailure]:
        del turn_id
        return Success(self.turn)


@pytest.mark.asyncio
async def test_summary_does_not_invent_confidence_or_downgrade_privacy() -> None:
    turn = ConversationTurn(
        record_id="summary-source",
        producer=PRODUCER,
        session_id="summary-session",
        role="user",
        text="Lokalny system używa niestandardowego harmonogramu.",
        data_class=DataClass.PRIVATE,
        domain_scope="system",
    )
    memory = MemoryRecord(
        record_id="summary-memory",
        producer=PRODUCER,
        kind="fact",
        topic="system.schedule.custom",
        content={"fact": "niestandardowy harmonogram"},
        source_turn_id=turn.record_id,
        data_class=DataClass.PRIVATE,
        domain_scope="system",
    )
    store = _SummaryStore(memory, turn)
    provider = GroundedClaimSummaryProvider(store)  # type: ignore[arg-type]

    result = await provider.retrieve(
        MemoryQuery(
            keywords=("harmonogram",),
            domain_scopes=("general", "system"),
        ),
        (DataClass.PRIVATE,),
        limit=1,
    )

    assert isinstance(result, Success)
    summary = result.unwrap()[0]
    assert summary.data_class == DataClass.PRIVATE
    assert summary.content["confidence"] is None
    assert summary.content["modalities"] == ("unknown",)
    assert summary.content["source_claims"][0]["source_span"] is None
    assert validate_grounded_summary(summary) is None

    tampered = replace(
        summary,
        content={**summary.content, "source_coverage": 0.5},
    )
    failure = validate_grounded_summary(tampered)
    assert failure is not None
    assert failure.code == "INCOMPLETE_SUMMARY_COVERAGE"


@pytest.mark.asyncio
async def test_summary_disappears_when_its_source_is_missing() -> None:
    memory = MemoryRecord(
        record_id="orphan-memory",
        producer=PRODUCER,
        kind="fact",
        topic="system.orphan",
        content={"fact": "orphan"},
        source_turn_id="deleted-source",
        domain_scope="system",
    )
    provider = GroundedClaimSummaryProvider(  # type: ignore[arg-type]
        _SummaryStore(memory, None)
    )

    result = await provider.retrieve(
        MemoryQuery(keywords=("orphan",), domain_scopes=("general", "system")),
        (DataClass.LOCAL,),
        limit=1,
    )

    assert isinstance(result, Success)
    assert result.unwrap() == ()
