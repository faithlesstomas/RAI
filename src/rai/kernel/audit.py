"""Append-only audit evidence for capability decisions and terminal outcomes."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Result, Success

from rai.paths import data_dir

from .records import ActionFailure, ActionResult, PolicyDecision, ProducerIdentity

AUDIT_PRODUCER = ProducerIdentity(
    producer_id="rai.audit", kind="audit-ledger", version="1.0.0"
)


class AuditEntry(BaseModel):
    """One append-only decision or terminal event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Literal["DECISION", "TERMINAL"]
    decision: PolicyDecision
    approval_id: str | None = None
    result: ActionResult | ActionFailure | None = None


class InMemoryAuditLedger:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    async def append(self, entry: AuditEntry) -> Result[AuditEntry, ActionFailure]:
        self.entries.append(entry)
        return Success(entry)

    async def append_once(self, entry: AuditEntry) -> Result[AuditEntry, ActionFailure]:
        if entry not in self.entries:
            return await self.append(entry)
        return Success(entry)


class JsonlAuditLedger:
    """Durable local JSONL ledger with serialized, flush-before-return appends."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / "audit" / "capability-v1.jsonl"
        self._lock = asyncio.Lock()

    async def append(self, entry: AuditEntry) -> Result[AuditEntry, ActionFailure]:
        return await self._append(entry, deduplicate=False)

    async def append_once(self, entry: AuditEntry) -> Result[AuditEntry, ActionFailure]:
        return await self._append(entry, deduplicate=True)

    async def _append(self, entry: AuditEntry, *, deduplicate: bool) -> Result[AuditEntry, ActionFailure]:
        try:
            async with self._lock:
                self._append_sync(entry, deduplicate=deduplicate)
            return Success(entry)
        except OSError as exc:
            request_id = entry.decision.request_id
            return Failure(
                ActionFailure(
                    producer=AUDIT_PRODUCER,
                    correlation_id=entry.decision.correlation_id,
                    request_id=request_id,
                    capability=entry.decision.target_resource,
                    code="AUDIT_UNAVAILABLE",
                    message=str(exc),
                )
            )

    def _append_sync(self, entry: AuditEntry, *, deduplicate: bool = False) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            payload = entry.model_dump(mode="json")
            if deduplicate and entry.stage == "TERMINAL":
                stream.seek(0)
                for line in stream:
                    try:
                        existing = json.loads(line)
                    except ValueError as exc:
                        raise OSError("audit ledger contains an incomplete record") from exc
                    if existing == payload:
                        # A previous append may have succeeded before its acknowledgement failed.
                        stream.flush()
                        os.fsync(stream.fileno())
                        return
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
