"""Durable and in-memory audit ledger for Rich Assistant operations."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rai.kernel.records import _new_id, _utc_now
from rai.paths import data_dir


class AssistantAuditEntry(BaseModel):
    """Structured audit log entry for one assistant turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str = Field(default_factory=_new_id, min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    model_name: str = Field(default="default", min_length=1)
    status: Literal["COMPLETED", "FAILED", "CANCELLED"] = "COMPLETED"
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens: dict[str, int] = Field(default_factory=dict)
    admitted_memories: tuple[str, ...] = ()
    superseded_memories: tuple[str, ...] = ()


@runtime_checkable
class AssistantAuditLedger(Protocol):
    """Protocol for recording assistant audit entries."""

    async def append(self, entry: AssistantAuditEntry) -> None: ...

    async def list_for_session(
        self, session_id: str
    ) -> tuple[AssistantAuditEntry, ...]: ...


class InMemoryAssistantAuditLedger:
    """In-memory audit ledger for fast unit and conformance tests."""

    def __init__(self) -> None:
        self._entries: list[AssistantAuditEntry] = []
        self._lock = asyncio.Lock()

    async def append(self, entry: AssistantAuditEntry) -> None:
        async with self._lock:
            self._entries.append(entry)

    async def list_for_session(
        self, session_id: str
    ) -> tuple[AssistantAuditEntry, ...]:
        async with self._lock:
            return tuple(e for e in self._entries if e.session_id == session_id)


class JsonlAssistantAuditLedger:
    """Append-only JSON Lines ledger for persistent audit logging."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / "assistant" / "audit.jsonl"
        self._lock = asyncio.Lock()
        self._path_initialized = False

    def _ensure_dir(self) -> None:
        if not self._path_initialized:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._path_initialized = True

    async def append(self, entry: AssistantAuditEntry) -> None:
        self._ensure_dir()
        serialized = entry.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._sync_append, serialized)

    def _sync_append(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    async def list_for_session(
        self, session_id: str
    ) -> tuple[AssistantAuditEntry, ...]:
        if not self.path.exists():
            return ()
        async with self._lock:
            return await asyncio.to_thread(self._sync_read_session, session_id)

    def _sync_read_session(self, session_id: str) -> tuple[AssistantAuditEntry, ...]:
        results: list[AssistantAuditEntry] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                if data.get("session_id") == session_id:
                    results.append(AssistantAuditEntry.model_validate(data))
        return tuple(results)
