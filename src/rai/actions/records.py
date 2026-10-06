"""Versioned action proposals and resource authority records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rai.kernel.records import DataClass, KernelRecord


class ChoiceState(str, Enum):
    """Lifecycle state of an active numbered selection frame."""

    ACTIVE = "ACTIVE"
    CONSUMED = "CONSUMED"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ChoiceEntry(BaseModel):
    """One selectable item in a numbered choice frame with a bound resource handle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=1, le=100)
    handle: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveChoiceFrame(KernelRecord):
    """Durable record of an active selection list in the current conversation."""

    record_type: Literal["active_choice_frame"] = "active_choice_frame"
    session_id: str = Field(min_length=1)
    profile_scope: str = Field(default="default", min_length=1)
    task_id: str = Field(min_length=1)
    source_turn_id: str = Field(min_length=1)
    intended_capability: Literal[
        "application.launch",
        "document.open",
        "browser.open_result",
        "browser.read_page",
        "process.inspect",
        "system.volume.set",
    ]
    entries: tuple[ChoiceEntry, ...] = Field(min_length=1)
    state: ChoiceState = ChoiceState.ACTIVE
    expires_at: datetime
    data_class: DataClass = DataClass.LOCAL

    @field_validator("expires_at")
    @classmethod
    def absolute_expiry(cls, value: datetime) -> datetime:
        """Reject ambiguous wall-clock expiry."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiry requires timezone")
        return value

    @model_validator(mode="after")
    def positive_lifetime(self) -> ActiveChoiceFrame:
        """A choice frame cannot be born expired."""
        if self.expires_at <= self.timestamp:
            raise ValueError("expiry must follow issuance")
        if self.data_class in {DataClass.SECRET, DataClass.BLOCKED}:
            raise ValueError("forbidden resource classification")
        return self


class ActionProposal(KernelRecord):
    """Untrusted intent referencing a resource issued by the runtime."""

    record_type: Literal["action_proposal"] = "action_proposal"
    source_turn_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    capability: Literal[
        "application.launch", "document.open", "browser.open_result",
        "browser.read_page", "process.inspect", "system.volume.set",
    ]
    resource_handle: str = Field(min_length=1, max_length=128)
    data_class: DataClass = DataClass.LOCAL


class ResourceHandle(KernelRecord):
    """Opaque authority; target/fingerprint stay in trusted runtime storage."""

    record_type: Literal["resource_handle"] = "resource_handle"
    actor_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    kind: Literal["application", "file", "url", "process", "audio_sink"]
    target: str = Field(min_length=1, max_length=8192)
    fingerprint: str = Field(min_length=1, max_length=256)
    operations: tuple[str, ...] = Field(min_length=1, max_length=8)
    expires_at: datetime
    data_class: DataClass = DataClass.LOCAL

    @field_validator("expires_at")
    @classmethod
    def absolute_expiry(cls, value: datetime) -> datetime:
        """Reject ambiguous wall-clock expiry."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expiry requires timezone")
        return value

    @model_validator(mode="after")
    def positive_lifetime(self) -> ResourceHandle:
        """A handle cannot be born expired."""
        if self.expires_at <= self.timestamp:
            raise ValueError("expiry must follow issuance")
        if self.data_class in {DataClass.SECRET, DataClass.BLOCKED}:
            raise ValueError("forbidden resource classification")
        return self
