"""Verified, source-covered write-back for derived assistant findings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)

from .ports import MemoryGraphStore
from .records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    MemoryOperation,
    MemoryOperationKind,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    make_assistant_failure,
)

_DATA_CLASS_RANK = {
    DataClass.PUBLIC: 0,
    DataClass.LOCAL: 1,
    DataClass.PRIVATE: 2,
}


def _data_class(value: DataClass | str) -> DataClass:
    return value if isinstance(value, DataClass) else DataClass(value)


class VerifiedDerivedClaim(BaseModel):
    """Explicit verifier output; ordinary model prose cannot instantiate write-back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verification_id: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    content: dict[str, Any]
    source_memory_ids: tuple[str, ...] = Field(min_length=1)
    profile_scope: str = Field(default="default", min_length=1)
    purpose: str = Field(default="assistant", min_length=1)
    domain_scope: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


async def write_verified_derived_claim(  # noqa: PLR0911
    store: MemoryGraphStore,
    finding: VerifiedDerivedClaim,
) -> Result[MemoryRecord, ActionFailure]:
    """Persist one derived claim only after validating every declared source."""
    if len(set(finding.source_memory_ids)) != len(finding.source_memory_ids):
        return Failure(
            make_assistant_failure(
                code="INVALID_DERIVED_SOURCES",
                message="derived source memory IDs must be unique",
                request_id=finding.verification_id,
            )
        )

    sources: list[MemoryRecord] = []
    for memory_id in finding.source_memory_ids:
        source_result = await store.get_memory(memory_id)
        if isinstance(source_result, Failure):
            return Failure(source_result.failure())
        source_and_status = source_result.unwrap()
        if source_and_status is None or source_and_status[1] != "ACTIVE":
            return Failure(
                make_assistant_failure(
                    code="INELIGIBLE_DERIVED_SOURCE",
                    message=f"source memory {memory_id} is missing or inactive",
                    request_id=finding.verification_id,
                )
            )
        sources.append(source_and_status[0])

    source_scopes = {source.profile_scope for source in sources}
    source_purposes = {source.purpose for source in sources}
    source_domains = {source.domain_scope for source in sources}
    if source_scopes != {finding.profile_scope}:
        return Failure(
            make_assistant_failure(
                code="DERIVED_SCOPE_MISMATCH",
                message="all sources must belong to the requested profile scope",
                request_id=finding.verification_id,
            )
        )
    if source_purposes != {finding.purpose}:
        return Failure(
            make_assistant_failure(
                code="DERIVED_PURPOSE_MISMATCH",
                message="all sources must have the requested purpose",
                request_id=finding.verification_id,
            )
        )
    if finding.domain_scope is None and len(source_domains) != 1:
        return Failure(
            make_assistant_failure(
                code="DERIVED_DOMAIN_AMBIGUOUS",
                message="cross-domain write-back requires an explicit domain policy",
                request_id=finding.verification_id,
            )
        )
    domain_scope = finding.domain_scope or next(iter(source_domains))
    if any(source.domain_scope != domain_scope for source in sources):
        return Failure(
            make_assistant_failure(
                code="DERIVED_DOMAIN_MISMATCH",
                message="derived claims cannot broaden their source domain",
                request_id=finding.verification_id,
            )
        )

    producer = ProducerIdentity(
        producer_id=finding.verifier_id,
        kind="verifier",
        version=finding.verifier_version,
    )
    provenance = tuple(
        ProvenanceReference(
            source_id=source.record_id,
            source_type="memory_record",
            source_version=source.schema_version,
            relation="DERIVED_FROM",
            producer=source.producer,
        )
        for source in sources
    )
    strictest_class = max(
        (_data_class(source.data_class) for source in sources),
        key=_DATA_CLASS_RANK.__getitem__,
    )
    source_confidences: list[float] = []
    for source in sources:
        raw_confidence = source.content.get("confidence")
        if not isinstance(raw_confidence, (int, float)) or isinstance(
            raw_confidence, bool
        ):
            return Failure(
                make_assistant_failure(
                    code="DERIVED_SOURCE_CONFIDENCE_UNKNOWN",
                    message=(
                        f"source memory {source.record_id} has no numeric confidence; "
                        "derived confidence cannot be bounded safely"
                    ),
                    request_id=finding.verification_id,
                )
            )
        source_confidences.append(min(1.0, max(0.0, float(raw_confidence))))
    bounded_confidence = min(finding.confidence, *source_confidences)
    memory = MemoryRecord(
        record_id=_new_id(),
        producer=producer,
        kind="derived_claim",
        topic=finding.topic,
        content={
            **finding.content,
            "confidence": bounded_confidence,
            "verification_id": finding.verification_id,
            "verifier_id": finding.verifier_id,
            "verifier_version": finding.verifier_version,
            "policy_version": finding.policy_version,
            "source_memory_ids": finding.source_memory_ids,
            "source_coverage": 1.0,
        },
        source_turn_id=sources[0].source_turn_id,
        source_type=sources[0].source_type,
        data_class=strictest_class,
        profile_scope=finding.profile_scope,
        domain_scope=domain_scope,
        purpose=finding.purpose,
        epistemic_status="inferred",
        provenance=provenance,
    )
    relations = tuple(
        MemoryRelation(
            source_id=source.record_id,
            target_id=memory.record_id,
            kind=MemoryRelationKind.SUPPORTS,
            confidence=bounded_confidence,
            epistemic_status="inferred",
            provenance=provenance,
            policy_outcome="ALLOW",
            eligible=True,
            metadata={
                "verification_id": finding.verification_id,
                "policy_version": finding.policy_version,
            },
        )
        for source in sources
    )
    operation = MemoryOperation(
        producer=producer,
        operation=MemoryOperationKind.REFLECT,
        trigger="api",
        trigger_id=finding.verification_id,
        profile_scope=finding.profile_scope,
        result_memory_ids=(memory.record_id,),
        active_memory_ids_after=(memory.record_id,),
        preconditions=(
            "all_sources_active",
            "source_coverage_complete",
            "all_source_confidences_known",
            "scope_not_broadened",
            f"verification_policy:{finding.policy_version}",
        ),
        policy_outcome="ALLOW",
        status="APPLIED",
        stage="STORAGE",
        reason="verified derived claim with complete source coverage",
        confidence=bounded_confidence,
        evidence=provenance,
    )
    session_id = f"verification-{finding.verification_id}"
    verification_turn = ConversationTurn(
        record_id=f"verification-turn-{finding.verification_id}",
        producer=producer,
        session_id=session_id,
        role="assistant",
        text="Verified derived-memory write-back.",
        data_class=strictest_class,
        domain_scope=domain_scope,
        purpose=finding.purpose,
        status="ACCEPTED",
        provenance=provenance,
        metadata={"profile_scope": finding.profile_scope, "internal": True},
    )
    accepted = await store.accept_turn(verification_turn)
    if isinstance(accepted, Failure):
        return Failure(accepted.failure())
    manifest = AssistantContextManifest(
        producer=producer,
        session_id=session_id,
        turn_id=verification_turn.record_id,
        durable_memory_ids=finding.source_memory_ids,
        routing_decision="compact_memory",
        policy_version=finding.policy_version,
    )
    response = AssistantResponse(
        producer=producer,
        session_id=session_id,
        turn_id=f"verification-response-turn-{finding.verification_id}",
        user_turn_id=verification_turn.record_id,
        request_id=f"verification-request-{finding.verification_id}",
        manifest_id=manifest.record_id,
        text="Verified derived claim persisted.",
        admitted_memory_ids=(memory.record_id,),
        provenance=provenance,
    )
    committed = await store.commit_terminal(
        response=response,
        manifest=manifest,
        assistant_turn=None,
        memories=(memory,),
        relations=relations,
        operations=(operation,),
    )
    if isinstance(committed, Failure):
        return Failure(committed.failure())
    return Success(memory)
