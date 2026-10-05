"""Versioned action proposals and resource authority records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from rai.kernel.records import DataClass, KernelRecord


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
