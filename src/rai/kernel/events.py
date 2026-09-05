"""Versioned contracts for the durable local event plane."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .records import (
    ActionFailure,
    ActionResult,
    Observation,
    PolicyDecision,
    UsageRecord,
    DataClass,
)

EVENT_SCHEMA_VERSION = "1.0.0"
EventRecord = Annotated[
    Observation | ActionResult | ActionFailure | PolicyDecision | UsageRecord,
    Field(discriminator="record_type"),
]
TerminalEvent = ActionResult | ActionFailure


class EventCursor(BaseModel):
    """Opaque, versioned position immediately after a journal sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    value: str

    @classmethod
    def from_sequence(cls, sequence: int) -> "EventCursor":
        payload = json.dumps({"v": 1, "s": sequence}, separators=(",", ":")).encode()
        return cls(value=base64.urlsafe_b64encode(payload).decode().rstrip("="))

    def sequence(self) -> int:
        try:
            padded = self.value + "=" * (-len(self.value) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            if payload.keys() != {"v", "s"} or payload["v"] != 1:
                raise ValueError
            sequence = payload["s"]
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                raise ValueError
            return sequence
        except (
            ValueError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            binascii.Error,
        ) as exc:
            raise ValueError("invalid event cursor") from exc


class EventEnvelope(BaseModel):
    """Journal metadata around an unchanged, schema-validated kernel record."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    envelope_version: Literal["1.0.0"] = EVENT_SCHEMA_VERSION
    sequence: int = Field(gt=0)
    cursor: EventCursor
    accepted_at: datetime
    clock_source: str = Field(min_length=1)
    clock_uncertainty_ms: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_class: DataClass
    record: EventRecord


class EventBatch(BaseModel):
    """A bounded ordered replay page and its next durable cursor."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    events: tuple[EventEnvelope, ...]
    next_cursor: EventCursor


class JournalFailure(BaseModel):
    """Typed failure returned by all journal boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    code: Literal[
        "CONFLICT",
        "INVALID_CURSOR",
        "INVALID_EVENT",
        "OVERSIZED_EVENT",
        "JOURNAL_UNAVAILABLE",
        "OVERLOADED",
        "ACK_REGRESSION",
    ]
    message: str = Field(min_length=1)
    record_id: str | None = None
    retryable: bool = False


def utc_now() -> datetime:
    """Return the journal acceptance clock value."""
    return datetime.now(timezone.utc)
