"""Durable opaque resource handles, scoped independently of model output."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import _utc_now
from .records import ResourceHandle

MAX_HANDLES = 4096


class SQLiteHandleStore:
    """Bounded private store; callers cannot supply or widen stored authority."""

    def __init__(
        self, path: Path, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = path
        self.clock = clock

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        connection = sqlite3.connect(self.path, timeout=2)
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS handles "
            "(id TEXT PRIMARY KEY, expires REAL NOT NULL, payload TEXT NOT NULL)"
        )
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def issue(self, handle: ResourceHandle) -> Result[str, str]:
        """Persist a trusted, unexpired handle without replacing another handle."""
        now = self.clock()
        if handle.timestamp > now or handle.expires_at <= now:
            return Failure("INVALID_HANDLE_LIFETIME")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM handles WHERE expires <= ?", (now.timestamp(),))
                count = connection.execute("SELECT COUNT(*) FROM handles").fetchone()[0]
                if count >= MAX_HANDLES:
                    return Failure("HANDLE_CAPACITY")
                connection.execute(
                    "INSERT INTO handles VALUES (?, ?, ?)",
                    (handle.record_id, handle.expires_at.timestamp(), handle.model_dump_json()),
                )
            return Success(handle.record_id)
        except (OSError, sqlite3.Error):
            return Failure("HANDLE_STORE_UNAVAILABLE")

    def resolve(  # noqa: PLR0913
        self, handle_id: str, *, actor_id: str, task_id: str, operation: str,
    ) -> Result[ResourceHandle, str]:
        """Resolve authority only for the original actor/task and permitted action."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload FROM handles WHERE id = ?", (handle_id,)
                ).fetchone()
            if row is None:
                return Failure("HANDLE_NOT_FOUND")
            handle = ResourceHandle.model_validate_json(row[0])
        except (OSError, sqlite3.Error, ValueError):
            return Failure("HANDLE_STORE_UNAVAILABLE")
        if handle.expires_at <= self.clock():
            return Failure("HANDLE_EXPIRED")
        if (handle.actor_id != actor_id or handle.task_id != task_id
                or operation not in handle.operations):
            return Failure("HANDLE_SCOPE_DENIED")
        return Success(handle)

    def revoke(self, handle_id: str) -> Result[None, str]:
        """Erase stored target data when its authority is revoked."""
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM handles WHERE id = ?", (handle_id,))
            return Success(None)
        except (OSError, sqlite3.Error):
            return Failure("HANDLE_STORE_UNAVAILABLE")
