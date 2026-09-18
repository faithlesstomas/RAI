"""Immutable, versioned records for the Rich Assistant runtime."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    InferenceBudget,
    KernelRecord,
    PolicyOutcome,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)

MAX_TURN_TEXT_BYTES = 16 * 1024
MAX_SESSION_ID_LENGTH = 128
MAX_PROPOSAL_CONTENT_CHARS = 1024
MAX_PROPOSALS_PER_TURN = 5

AssistantSessionId = TypeAliasType(
    "AssistantSessionId",
    Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_SESSION_ID_LENGTH,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        ),
    ],
)


class MemoryRelationKind(str, Enum):
    """Typed relationship between graph nodes in the assistant memory store."""

    REPLIES_TO = "REPLIES_TO"
    DERIVED_FROM = "DERIVED_FROM"
    ABOUT = "ABOUT"
    SUPERSEDES = "SUPERSEDES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UPDATES = "UPDATES"


class MemoryOperationKind(str, Enum):
    """User-visible and internal operations over durable assistant memory."""

    REMEMBER = "REMEMBER"
    FORGET = "FORGET"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    REFLECT = "REFLECT"
    RECONSTRUCT = "RECONSTRUCT"


class MemoryRelation(BaseModel):
    """Directed typed edge between memory graph nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    relation_id: str = Field(default_factory=_new_id, min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    kind: MemoryRelationKind
    created_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    epistemic_status: Literal[
        "asserted", "observed", "inferred", "uncertain", "contested"
    ] = "asserted"
    provenance: tuple[ProvenanceReference, ...] = ()
    policy_outcome: PolicyOutcome = PolicyOutcome.ALLOW
    eligible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(KernelRecord):
    """Immutable user or assistant turn within an assistant session."""

    record_type: Literal["conversation_turn"] = "conversation_turn"
    session_id: AssistantSessionId
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=MAX_TURN_TEXT_BYTES)
    reply_to_turn_id: str | None = None
    data_class: DataClass = DataClass.LOCAL
    domain_scope: str = Field(default="unknown", min_length=1, max_length=128)
    purpose: str = Field(default="assistant", min_length=1, max_length=128)
    status: Literal["ACCEPTED", "COMPLETED", "FAILED", "CANCELLED"] = "ACCEPTED"
    provenance: tuple[ProvenanceReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data_class")
    @classmethod
    def validate_data_class(cls, value: DataClass | str) -> DataClass | str:
        raw = value.value if isinstance(value, DataClass) else str(value)
        if raw in (DataClass.SECRET.value, DataClass.BLOCKED.value):
            raise ValueError(f"conversation turn cannot have {raw} data class")
        return value


class MemoryProposal(KernelRecord):
    """Untrusted candidate memory proposed by a model backend."""

    record_type: Literal["memory_proposal"] = "memory_proposal"
    source_turn_id: str = Field(min_length=1)
    operation: MemoryOperationKind = MemoryOperationKind.REMEMBER
    kind: str = Field(default="preference", min_length=1)
    topic: str = Field(min_length=1)
    target_topic: str | None = None
    content: dict[str, Any]
    privacy_class: DataClass = DataClass.LOCAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_span: str | None = Field(default=None, min_length=1, max_length=1024)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=1)
    statement_type: Literal[
        "assertion", "preference", "plan", "correction", "request"
    ] = "assertion"
    modality: Literal["direct", "hedged", "quoted", "hearsay"] = "direct"
    negated: bool = False
    scope: str = Field(default="personal", min_length=1, max_length=128)
    domain_scope: str = Field(default="unknown", min_length=1, max_length=128)
    purpose: str = Field(default="assistant", min_length=1, max_length=128)
    source_type: Literal[
        "conversation_turn", "rich_history_episode", "system_observation"
    ] = "conversation_turn"
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    supersedes_memory_id: str | None = None
    relations: tuple[MemoryRelation, ...] = ()

    @field_validator("privacy_class")
    @classmethod
    def validate_privacy_class(cls, value: DataClass | str) -> DataClass | str:
        raw = value.value if isinstance(value, DataClass) else str(value)
        if raw in (DataClass.SECRET.value, DataClass.BLOCKED.value):
            raise ValueError(f"memory proposal cannot have {raw} privacy class")
        return value

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(value, ensure_ascii=False)
        if len(serialized) > MAX_PROPOSAL_CONTENT_CHARS:
            raise ValueError(
                f"memory proposal content exceeds max size of {MAX_PROPOSAL_CONTENT_CHARS} characters"
            )
        return value

    @model_validator(mode="after")
    def validate_source_offsets(self) -> MemoryProposal:
        """Require source offsets to be present together and bound the source span."""
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("source span offsets must be provided together")
        if self.span_start is not None and self.span_end is not None:
            if self.span_end <= self.span_start:
                raise ValueError("span_end must be greater than span_start")
            if self.source_span is None:
                raise ValueError("source_span is required when offsets are present")
            if self.span_end - self.span_start != len(self.source_span):
                raise ValueError("source span offsets do not match source_span length")
        return self

    @model_validator(mode="after")
    def validate_validity_interval(self) -> MemoryProposal:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("valid_until must not precede valid_from")
        return self


class MemoryRecord(KernelRecord):
    """Immutable, policy-admitted durable memory payload."""

    record_type: Literal["memory_record"] = "memory_record"
    kind: str = Field(default="preference", min_length=1)
    topic: str = Field(min_length=1)
    content: dict[str, Any]
    source_turn_id: str = Field(min_length=1)
    source_type: Literal[
        "conversation_turn", "rich_history_episode", "system_observation"
    ] = "conversation_turn"
    data_class: DataClass = DataClass.LOCAL
    profile_scope: str = Field(default="default", min_length=1)
    domain_scope: str = Field(default="unknown", min_length=1, max_length=128)
    purpose: str = Field(default="assistant", min_length=1, max_length=128)
    epistemic_status: Literal[
        "asserted", "observed", "inferred", "uncertain", "contested"
    ] = "asserted"
    valid_from: datetime = Field(default_factory=_utc_now)
    valid_until: datetime | None = None
    recorded_at: datetime = Field(default_factory=_utc_now)
    expired_at: datetime | None = None
    provenance: tuple[ProvenanceReference, ...] = ()

    @field_validator("data_class")
    @classmethod
    def validate_data_class(cls, value: DataClass | str) -> DataClass | str:
        raw = value.value if isinstance(value, DataClass) else str(value)
        if raw in (DataClass.SECRET.value, DataClass.BLOCKED.value):
            raise ValueError(f"memory record cannot have {raw} data class")
        return value

    @model_validator(mode="after")
    def validate_time_intervals(self) -> MemoryRecord:
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until must not precede valid_from")
        if self.expired_at is not None and self.expired_at < self.recorded_at:
            raise ValueError("expired_at must not precede recorded_at")
        return self


class MemoryOperation(KernelRecord):
    """Immutable audit record for one attempted durable-memory operation."""

    record_type: Literal["memory_operation"] = "memory_operation"
    operation: MemoryOperationKind
    trigger: Literal[
        "conversation_turn", "user_command", "api", "source_deletion", "replay"
    ]
    trigger_id: str = Field(min_length=1)
    profile_scope: str = Field(default="default", min_length=1)
    proposal_id: str | None = None
    source_span: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    modality: Literal["direct", "hedged", "quoted", "hearsay"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    target_memory_ids: tuple[str, ...] = ()
    result_memory_ids: tuple[str, ...] = ()
    active_memory_ids_before: tuple[str, ...] = ()
    active_memory_ids_after: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    policy_outcome: PolicyOutcome = PolicyOutcome.ALLOW
    status: Literal["APPLIED", "REJECTED", "NOOP"] = "APPLIED"
    stage: Literal[
        "EXTRACTION",
        "ADMISSION",
        "RETRIEVAL",
        "STORAGE",
        "UPDATE",
        "DELETION",
        "RECONSTRUCTION",
    ] = "STORAGE"
    reason: str | None = None
    evidence: tuple[ProvenanceReference, ...] = ()

    @model_validator(mode="after")
    def validate_source_span(self) -> MemoryOperation:
        offsets = (self.span_start, self.span_end)
        if any(value is not None for value in offsets):
            if any(value is None for value in offsets):
                raise ValueError("span_start and span_end must be provided together")
            if self.source_span is None:
                raise ValueError("source_span is required when offsets are present")
            if self.span_end - self.span_start != len(self.source_span):  # type: ignore[operator]
                raise ValueError("source span offsets do not match source_span length")
        return self


class AssistantContextManifestItem(BaseModel):
    """Item recorded in the assistant context manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    layer: str = Field(min_length=1)
    data_class: DataClass = DataClass.LOCAL
    domain_scope: str = Field(default="unknown", min_length=1)
    purpose: str = Field(default="assistant", min_length=1)
    ranking_reason: str | None = None
    fields: tuple[str, ...] = ()
    redactions: tuple[str, ...] = ()


class AssistantContextManifest(KernelRecord):
    """Inspectable provenance manifest for an assembled assistant context."""

    record_type: Literal["assistant_context_manifest"] = "assistant_context_manifest"
    session_id: AssistantSessionId
    turn_id: str = Field(min_length=1)
    recent_turn_ids: tuple[str, ...] = ()
    episodic_turn_ids: tuple[str, ...] = ()
    external_evidence_ids: tuple[str, ...] = ()
    durable_memory_ids: tuple[str, ...] = ()
    ranking_reasons: dict[str, str] = Field(default_factory=dict)
    exclusions: tuple[str, ...] = ()
    redactions: tuple[str, ...] = ()
    routing_decision: Literal[
        "recent_conversation",
        "compact_memory",
        "raw_evidence_fallback",
        "no_evidence",
    ] = "no_evidence"
    route_candidates: tuple[str, ...] = ()
    rejected_routes: tuple[str, ...] = ()
    sufficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency_factors: dict[str, float] = Field(default_factory=dict)
    sufficiency_reasons: tuple[str, ...] = ()
    fallback_used: bool = False
    evidence_required: bool = False
    retrieval_channel_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    graph_paths: tuple[dict[str, Any], ...] = ()
    evidence_character_budget: int = Field(default=0, ge=0)
    retriever_version: str = Field(default="1.0.0", min_length=1)
    policy_version: str = Field(default="1.0.0", min_length=1)
    backend_name: str = Field(default="unknown", min_length=1)
    model_name: str = Field(default="unknown", min_length=1)
    model_artifact_version: str | None = None
    prompt_template_version: str = Field(default="unknown", min_length=1)
    actual_tokens: int | None = None
    actual_characters: int = Field(default=0, ge=0)
    items: tuple[AssistantContextManifestItem, ...] = ()


class AssistantContextPackage(KernelRecord):
    """Reconstructed, bounded context package for an assistant inference."""

    record_type: Literal["assistant_context_package"] = "assistant_context_package"
    session_id: AssistantSessionId
    turn_id: str = Field(min_length=1)
    manifest: AssistantContextManifest
    content: dict[str, Any]
    provenance: tuple[ProvenanceReference, ...] = ()


class InferenceRequest(KernelRecord):
    """Envelope for one bounded model backend invocation."""

    record_type: Literal["assistant_inference_request"] = "assistant_inference_request"
    session_id: AssistantSessionId
    turn_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    context: AssistantContextPackage
    budget: InferenceBudget
    strategy: str = Field(default="DIRECT", min_length=1)
    model_name: str | None = None
    system_instruction: str | None = None


class AssistantCandidate(BaseModel):
    """Candidate output generated by an AssistantModelBackend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    proposals: tuple[MemoryProposal, ...] = ()
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("proposals")
    @classmethod
    def validate_proposals_count(
        cls, value: tuple[MemoryProposal, ...]
    ) -> tuple[MemoryProposal, ...]:
        if len(value) > MAX_PROPOSALS_PER_TURN:
            raise ValueError(
                f"candidate contains {len(value)} proposals, exceeding max of {MAX_PROPOSALS_PER_TURN}"
            )
        return value


class AssistantResponse(KernelRecord):
    """Exactly one terminal outcome for an accepted turn."""

    record_type: Literal["assistant_response"] = "assistant_response"
    session_id: AssistantSessionId
    turn_id: str = Field(min_length=1)
    user_turn_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    text: str
    status: Literal["COMPLETED", "FAILED", "CANCELLED"] = "COMPLETED"
    error_message: str | None = None
    admitted_memory_ids: tuple[str, ...] = ()
    memory_operation_ids: tuple[str, ...] = ()
    provenance: tuple[ProvenanceReference, ...] = ()


AnyAssistantRecord = Annotated[
    ConversationTurn
    | MemoryProposal
    | MemoryRecord
    | MemoryOperation
    | AssistantContextManifest
    | AssistantContextPackage
    | InferenceRequest
    | AssistantResponse,
    Field(discriminator="record_type"),
]

ASSISTANT_RECORD_TYPES: dict[str, type[KernelRecord]] = {
    model.model_fields["record_type"].default: model
    for model in (
        ConversationTurn,
        MemoryProposal,
        MemoryRecord,
        MemoryOperation,
        AssistantContextManifest,
        AssistantContextPackage,
        InferenceRequest,
        AssistantResponse,
    )
}


def parse_assistant_record(data: dict[str, Any]) -> KernelRecord:
    """Validate an untrusted record using its explicit record_type discriminator."""
    record_type = data.get("record_type")
    model = ASSISTANT_RECORD_TYPES.get(record_type)
    if model is None:
        raise ValueError(f"unsupported assistant record_type: {record_type!r}")
    return model.model_validate(data)


def make_assistant_failure(
    code: str,
    message: str,
    request_id: str = "assistant",
    retryable: bool = False,
    producer: ProducerIdentity | None = None,
) -> ActionFailure:
    """Construct a valid ActionFailure for assistant runtime errors."""
    return ActionFailure(
        request_id=request_id,
        capability="assistant",
        code=code,
        message=message,
        retryable=retryable,
        producer=producer
        or ProducerIdentity(
            producer_id="assistant-service", kind="service", version="1.0.0"
        ),
    )
