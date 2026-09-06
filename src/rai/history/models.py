"""Versioned inputs and status records for Rich History."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rai.kernel.ports import LifecycleState
from rai.kernel.records import DataClass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PrivacyAction(str, Enum):
    DROP = "DROP"
    METADATA_ONLY = "METADATA_ONLY"
    REDACT = "REDACT"
    ALLOW = "ALLOW"


class SourceEvent(BaseModel):
    """Bounded semantic event emitted by a platform adapter.

    Page and accessibility text is data only. It is never interpreted as a
    command by this layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["gnome", "atspi", "process", "filesystem", "browser"]
    kind: str = Field(min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=utc_now)
    application_id: str | None = Field(default=None, max_length=256)
    title: str | None = Field(default=None, max_length=2048)
    origin: str | None = Field(default=None, max_length=2048)
    url: str | None = Field(default=None, max_length=4096)
    path: str | None = Field(default=None, max_length=4096)
    field_role: str | None = Field(default=None, max_length=128)
    resource_id: str | None = Field(default=None, max_length=512)
    project: str | None = Field(default=None, max_length=512)
    toolkit: str | None = Field(default=None, max_length=128)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    selected_text: str | None = Field(default=None, max_length=8192)
    payload: dict[str, Any] = Field(default_factory=dict)
    private_browsing: bool = False
    session_locked: bool = False
    depth: int = Field(default=0, ge=0, le=32)

    @field_validator("timestamp")
    @classmethod
    def absolute_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


class PrivacyDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    action: PrivacyAction
    data_class: DataClass
    reason_codes: tuple[str, ...] = Field(min_length=1)
    policy_version: str = "1.0.0"


class FilteredEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event: SourceEvent
    decision: PrivacyDecision


class ActivityFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    fact_id: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    kind: str
    application_id: str | None = None
    resource_id: str | None = None
    project: str | None = None
    sources: tuple[str, ...]
    observation_ids: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    data_class: DataClass
    payload: dict[str, Any] = Field(default_factory=dict)
    conflicts: tuple[str, ...] = ()


class CollectorHealth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    name: str
    state: LifecycleState
    enabled: bool
    permission: Literal["GRANTED", "DENIED", "UNAVAILABLE", "UNKNOWN"]
    last_event_at: datetime | None = None
    last_error: str | None = None
    restart_count: int = Field(default=0, ge=0)
