"""Stage-specific diagnostics for evidence-preserving assistant memory."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field
from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass, _utc_now

from .ports import MemoryGraphStore
from .records import AssistantResponse


class MemoryStageDiagnostic(BaseModel):
    """Inspectable outcome for one memory-pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Literal[
        "EXTRACTION", "ADMISSION", "STORAGE", "UPDATE", "RETRIEVAL", "ANSWER_USE"
    ]
    status: Literal["PASS", "FAIL", "NOT_APPLICABLE", "UNVERIFIED"]
    message: str
    evidence_ids: tuple[str, ...] = ()


class MemoryDiagnosticReport(BaseModel):
    """Reproducible integrity and answer-use report for one memory profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime = Field(default_factory=_utc_now)
    profile_scope: str
    operation_count: int = Field(ge=0)
    active_memory_ids: tuple[str, ...]
    replayed_memory_ids: tuple[str, ...]
    stages: tuple[MemoryStageDiagnostic, ...]

    @computed_field
    @property
    def healthy(self) -> bool:
        return all(stage.status != "FAIL" for stage in self.stages)


async def diagnose_memory(  # noqa: PLR0912, PLR0915
    store: MemoryGraphStore,
    profile_scope: str = "default",
    response: AssistantResponse | None = None,
    expected_answer_substring: str | None = None,
) -> Result[MemoryDiagnosticReport, ActionFailure]:
    """Diagnose each observable stage without treating answer text as sole evidence."""
    operations_result = await store.list_memory_operations(
        profile_scope=profile_scope, limit=1000
    )
    if isinstance(operations_result, Failure):
        return Failure(operations_result.failure())
    operations = operations_result.unwrap()

    active_result = await store.retrieve_relevant_memories(
        profile_scope=profile_scope,
        data_classes=(DataClass.PUBLIC, DataClass.LOCAL, DataClass.PRIVATE),
        limit=1000,
    )
    if isinstance(active_result, Failure):
        return Failure(active_result.failure())
    active_records = tuple(memory for memory, _reason in active_result.unwrap())
    active_ids = tuple(memory.record_id for memory in active_records)

    replay_result = await store.replay_memory_projection(profile_scope=profile_scope)
    if isinstance(replay_result, Failure):
        return Failure(replay_result.failure())
    replayed_ids = replay_result.unwrap()

    stages: list[MemoryStageDiagnostic] = []
    conversational_operations = tuple(
        operation
        for operation in operations
        if operation.trigger == "conversation_turn"
    )
    missing_evidence = tuple(
        operation.record_id
        for operation in conversational_operations
        if (
            not operation.evidence
            or operation.proposal_id is None
            or operation.source_span is None
            or operation.span_start is None
            or operation.span_end is None
        )
    )
    stages.append(
        MemoryStageDiagnostic(
            stage="EXTRACTION",
            status=(
                "NOT_APPLICABLE"
                if not conversational_operations
                else "FAIL"
                if missing_evidence
                else "PASS"
            ),
            message=(
                "no memory proposal has been observed"
                if not conversational_operations
                else "operation traces are missing source evidence"
                if missing_evidence
                else "all observed proposals retain their trigger evidence"
            ),
            evidence_ids=missing_evidence
            or tuple(operation.record_id for operation in conversational_operations),
        )
    )

    inconsistent_admission = tuple(
        operation.record_id
        for operation in operations
        if (operation.status == "REJECTED") != (str(operation.policy_outcome) == "DENY")
    )
    stages.append(
        MemoryStageDiagnostic(
            stage="ADMISSION",
            status="FAIL" if inconsistent_admission else "PASS",
            message=(
                "operation status conflicts with its policy outcome"
                if inconsistent_admission
                else "admission outcomes are explicit and internally consistent"
            ),
            evidence_ids=inconsistent_admission
            or tuple(operation.record_id for operation in operations),
        )
    )

    missing_active_records: list[str] = []
    for memory_id in active_ids:
        stored = await store.get_memory(memory_id)
        if isinstance(stored, Failure) or stored.unwrap() is None:
            missing_active_records.append(memory_id)
    stages.append(
        MemoryStageDiagnostic(
            stage="STORAGE",
            status="FAIL" if missing_active_records else "PASS",
            message=(
                "active projection references missing durable records"
                if missing_active_records
                else "every active projection entry has a durable record"
            ),
            evidence_ids=tuple(missing_active_records) or active_ids,
        )
    )

    projection_matches = set(active_ids) == set(replayed_ids)
    stages.append(
        MemoryStageDiagnostic(
            stage="UPDATE",
            status="PASS" if projection_matches else "FAIL",
            message=(
                "operation replay matches the authoritative active projection"
                if projection_matches
                else "operation replay diverges from the authoritative active projection"
            ),
            evidence_ids=tuple(dict.fromkeys((*active_ids, *replayed_ids))),
        )
    )

    missing_sources: list[str] = []
    for memory in active_records:
        source = await store.get_turn(memory.source_turn_id)
        if isinstance(source, Failure) or source.unwrap() is None:
            missing_sources.append(memory.record_id)
    stages.append(
        MemoryStageDiagnostic(
            stage="RETRIEVAL",
            status="FAIL" if missing_sources else "PASS",
            message=(
                "active memories lack an eligible source turn"
                if missing_sources
                else "active memories retain retrievable source evidence"
            ),
            evidence_ids=tuple(missing_sources) or active_ids,
        )
    )

    answer_status: Literal["PASS", "FAIL", "NOT_APPLICABLE", "UNVERIFIED"]
    answer_message: str
    answer_evidence: tuple[str, ...] = ()
    if response is None:
        answer_status = "UNVERIFIED"
        answer_message = "answer use requires a concrete response and expectation"
    else:
        context_result = await store.get_context_package(response.manifest_id)
        context = (
            context_result.unwrap() if isinstance(context_result, Success) else None
        )
        answer_evidence = (
            tuple(context.manifest.durable_memory_ids) if context is not None else ()
        )
        if context is None:
            answer_status = "FAIL"
            answer_message = "response has no persisted context package"
        elif expected_answer_substring is None:
            answer_status = "UNVERIFIED"
            answer_message = (
                "context is inspectable but no answer expectation was supplied"
            )
        elif expected_answer_substring.casefold() in response.text.casefold():
            answer_status = "PASS"
            answer_message = "response satisfies the declared answer expectation"
        else:
            answer_status = "FAIL"
            answer_message = "response does not satisfy the declared answer expectation"
    stages.append(
        MemoryStageDiagnostic(
            stage="ANSWER_USE",
            status=answer_status,
            message=answer_message,
            evidence_ids=answer_evidence,
        )
    )

    return Success(
        MemoryDiagnosticReport(
            profile_scope=profile_scope,
            operation_count=len(operations),
            active_memory_ids=active_ids,
            replayed_memory_ids=replayed_ids,
            stages=tuple(stages),
        )
    )
