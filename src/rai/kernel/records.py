"""Immutable, versioned records shared by every RAI runtime boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeAlias
from uuid import uuid4

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CURRENT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_MAJOR = 1
SCHEMA_VERSION_PATTERN = r"^1\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:[-+][0-9A-Za-z.-]+)?$"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class FrozenDict(dict[str, Any]):
    """JSON-compatible mapping which rejects mutation after validation."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        raise TypeError("kernel record mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_json_value(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, dict) and not isinstance(value, FrozenDict):
        return FrozenDict(
            {key: freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_json_value(item) for item in value)
    return value


class DataClass(str, Enum):
    """Policy classification attached to persisted and outbound data."""

    PUBLIC = "PUBLIC"
    LOCAL = "LOCAL"
    PRIVATE = "PRIVATE"
    SECRET = "SECRET"
    BLOCKED = "BLOCKED"


class RiskClass(str, Enum):
    """Risk associated with invoking a capability."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyOutcome(str, Enum):
    """Machine-readable policy outcomes."""

    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"
    ESCALATE = "ESCALATE"


class ProducerIdentity(BaseModel):
    """Identity of the component which produced a record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    producer_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    version: str = Field(min_length=1)
    device_id: str | None = None


class ProvenanceReference(BaseModel):
    """Typed edge from a derived record to one of its sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    producer: ProducerIdentity


class KernelRecord(BaseModel):
    """Metadata common to every durable or externally exchanged record."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    record_type: str
    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, pattern=SCHEMA_VERSION_PATTERN)
    record_id: str = Field(default_factory=_new_id, min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    producer: ProducerIdentity
    correlation_id: str | None = None

    _record_types: ClassVar[dict[str, type[KernelRecord]]] = {}

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """Reject malformed versions and majors unsupported by this kernel."""
        try:
            parsed = Version(value)
        except InvalidVersion as exc:
            raise ValueError("schema_version must be a valid semantic version") from exc
        if parsed.major != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(
                f"unsupported schema major {parsed.major}; supported major is "
                f"{SUPPORTED_SCHEMA_MAJOR}"
            )
        return value

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require an absolute timestamp so devices do not exchange local time."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def freeze_nested_values(self) -> KernelRecord:
        """Apply the immutable-record guarantee recursively to JSON containers."""
        for field_name in self.__class__.model_fields.keys():
            current = getattr(self, field_name)
            frozen = freeze_json_value(current)
            if frozen is not current:
                object.__setattr__(self, field_name, frozen)
        return self


class DeviceDescriptor(KernelRecord):
    record_type: Literal["device_descriptor"] = "device_descriptor"
    display_name: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class MediaReference(KernelRecord):
    record_type: Literal["media_reference"] = "media_reference"
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    data_class: DataClass


class Observation(KernelRecord):
    record_type: Literal["observation"] = "observation"
    kind: str = Field(min_length=1)
    payload: dict[str, Any]
    data_class: DataClass
    device_id: str | None = None
    media: tuple[MediaReference, ...] = ()
    provenance: tuple[ProvenanceReference, ...] = ()


class Episode(KernelRecord):
    record_type: Literal["episode"] = "episode"
    started_at: datetime
    ended_at: datetime
    observation_ids: tuple[str, ...] = Field(min_length=1)
    provenance: tuple[ProvenanceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Episode:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class Claim(KernelRecord):
    record_type: Literal["claim"] = "claim"
    statement: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    epistemic_status: str = Field(min_length=1)
    data_class: DataClass
    provenance: tuple[ProvenanceReference, ...] = Field(min_length=1)


class Task(KernelRecord):
    record_type: Literal["task"] = "task"
    objective: str = Field(min_length=1)
    status: Literal["PENDING", "RUNNING", "CANCELLED", "SUCCEEDED", "FAILED"] = "PENDING"
    provenance: tuple[ProvenanceReference, ...] = ()


class ContextManifestItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    data_class: DataClass
    fields: tuple[str, ...] = ()
    redactions: tuple[str, ...] = ()


class ContextManifest(KernelRecord):
    record_type: Literal["context_manifest"] = "context_manifest"
    destination: str = Field(min_length=1)
    items: tuple[ContextManifestItem, ...]
    approved: bool = False


class ContextPackage(KernelRecord):
    record_type: Literal["context_package"] = "context_package"
    task_id: str = Field(min_length=1)
    manifest: ContextManifest
    content: dict[str, Any]
    provenance: tuple[ProvenanceReference, ...] = ()


class InferenceBudget(KernelRecord):
    record_type: Literal["inference_budget"] = "inference_budget"
    max_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)
    max_agent_turns: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_images: int = Field(ge=0)
    max_audio_seconds: float = Field(ge=0)
    max_latency_seconds: float = Field(gt=0)
    max_provider_cost: float = Field(ge=0)
    max_ram_bytes: int = Field(ge=0)
    max_vram_bytes: int = Field(ge=0)
    cancellation_deadline: datetime
    allowed_providers: tuple[str, ...] = ()
    fallback_order: tuple[str, ...] = ()


class CapabilityRequest(KernelRecord):
    record_type: Literal["capability_request"] = "capability_request"
    actor: ProducerIdentity
    capability: str = Field(min_length=1)
    arguments: dict[str, Any]
    data_class: DataClass
    target_resource: str = Field(min_length=1)
    requested_side_effects: tuple[str, ...]
    isolation: str = Field(min_length=1)
    budget: InferenceBudget | None = None
    verification_plan: tuple[str, ...] = Field(min_length=1)


class ActionResult(KernelRecord):
    record_type: Literal["action_result"] = "action_result"
    request_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    output: dict[str, Any]
    verification: dict[str, Any]
    terminal: Literal[True] = True
    provenance: tuple[ProvenanceReference, ...] = ()


class ActionFailure(KernelRecord):
    record_type: Literal["action_failure"] = "action_failure"
    request_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    terminal: Literal[True] = True
    provenance: tuple[ProvenanceReference, ...] = ()


class PolicyDecision(KernelRecord):
    record_type: Literal["policy_decision"] = "policy_decision"
    request_id: str = Field(min_length=1)
    outcome: PolicyOutcome
    risk_class: RiskClass
    policy_version: str = Field(min_length=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)
    actor: ProducerIdentity
    data_class: DataClass
    target_resource: str = Field(min_length=1)
    requested_side_effects: tuple[str, ...]
    isolation: str = Field(min_length=1)
    budget_id: str | None = None
    verification_plan: tuple[str, ...]
    approval_id: str | None = None


class UsageRecord(KernelRecord):
    record_type: Literal["usage_record"] = "usage_record"
    task_id: str = Field(min_length=1)
    processor: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    images: int = Field(ge=0)
    audio_seconds: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    provider_cost: float | None = Field(default=None, ge=0)


AnyKernelRecord: TypeAlias = Annotated[
    DeviceDescriptor
    | MediaReference
    | Observation
    | Episode
    | Claim
    | Task
    | ContextPackage
    | ContextManifest
    | CapabilityRequest
    | ActionResult
    | ActionFailure
    | PolicyDecision
    | InferenceBudget
    | UsageRecord,
    Field(discriminator="record_type"),
]

RECORD_TYPES: dict[str, type[KernelRecord]] = {
    model.model_fields["record_type"].default: model
    for model in (
        DeviceDescriptor,
        MediaReference,
        Observation,
        Episode,
        Claim,
        Task,
        ContextPackage,
        ContextManifest,
        CapabilityRequest,
        ActionResult,
        ActionFailure,
        PolicyDecision,
        InferenceBudget,
        UsageRecord,
    )
}


def parse_record(data: dict[str, Any]) -> KernelRecord:
    """Validate an untrusted record using its explicit record discriminator."""
    record_type = data.get("record_type")
    model = RECORD_TYPES.get(record_type)
    if model is None:
        raise ValueError(f"unsupported record_type: {record_type!r}")
    return model.model_validate(data)
