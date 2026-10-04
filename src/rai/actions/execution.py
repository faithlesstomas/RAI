"""Durable execution reservations: an uncertain side effect is never retried blindly."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, PolicyDecision


MAX_EXECUTION_RECORDS = 100_000
MAX_REQUEST_BYTES = 16 * 1024
MAX_TERMINAL_BYTES = 64 * 1024


class SQLiteExecutionStore:
    """Persist reservation before effects and preserve terminal evidence for retries."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self.path.chmod(0o600)
        connection = sqlite3.connect(self.path, timeout=2)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS executions "
                    "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, terminal TEXT, decision TEXT)"
                )
                yield connection
        finally:
            connection.close()

    @staticmethod
    def digest(request: CapabilityRequest) -> str:
        """Identity includes authority and arguments, but not transport timestamps."""
        payload = request.model_dump(mode="json", exclude={"timestamp", "producer"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def reserve(  # noqa: PLR0911
        self, request: CapabilityRequest,
    ) -> Result[ActionResult | ActionFailure | None, str]:
        """None grants first execution; in-flight/recovered reservations fail closed."""
        if len(request.model_dump_json().encode("utf-8")) > MAX_REQUEST_BYTES:
            return Failure("REQUEST_SIZE_LIMIT")
        digest = self.digest(request)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT digest, terminal FROM executions WHERE id = ?", (request.record_id,)
                ).fetchone()
                if row is None:
                    count = connection.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
                    if count >= MAX_EXECUTION_RECORDS:
                        return Failure("EXECUTION_CAPACITY_EXCEEDED")
                    connection.execute("INSERT INTO executions (id, digest, terminal) VALUES (?, ?, NULL)",
                                       (request.record_id, digest))
                    return Success(None)
                if row[0] != digest:
                    return Failure("REQUEST_ID_CONFLICT")
                if row[1] is None:
                    return Failure("EXECUTION_UNKNOWN")
                payload = json.loads(row[1])
                model = ActionResult if payload.get("record_type") == "action_result" else ActionFailure
                return Success(model.model_validate(payload))
        except (OSError, sqlite3.Error, ValueError):
            return Failure("EXECUTION_STORE_UNAVAILABLE")

    def finish(
        self, request: CapabilityRequest, terminal: ActionResult | ActionFailure,
        decision: PolicyDecision | None = None,
    ) -> Result[None, str]:
        """A completed reservation cannot be overwritten with another outcome."""
        if terminal.request_id != request.record_id or terminal.capability != request.capability:
            return Failure("TERMINAL_ID_MISMATCH")
        payload = terminal.model_dump_json()
        if len(payload.encode("utf-8")) > MAX_TERMINAL_BYTES:
            return Failure("TERMINAL_SIZE_LIMIT")
        try:
            with self._connect() as connection:
                changed = connection.execute(
                    "UPDATE executions SET terminal = ?, decision = ? WHERE id = ? AND digest = ? AND terminal IS NULL",
                    (payload, decision.model_dump_json() if decision else None,
                     request.record_id, self.digest(request)),
                ).rowcount
                if changed != 1:
                    return Failure("TERMINAL_CONFLICT")
            return Success(None)
        except (OSError, sqlite3.Error):
            return Failure("EXECUTION_STORE_UNAVAILABLE")

    def decision(self, request_id: str) -> PolicyDecision | None:
        """Read the original decision for an idempotent terminal replay."""
        with self._connect() as connection:
            row = connection.execute("SELECT decision FROM executions WHERE id = ?",
                                     (request_id,)).fetchone()
        return PolicyDecision.model_validate_json(row[0]) if row and row[0] else None
