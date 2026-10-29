"""REPL session state persistence mixin for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import sqlite3

from returns.result import Failure, Result, Success

from ...kernel.records import ActionFailure
from ..records import make_assistant_failure


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReplStoreMixin:
    """Session REPL heap snapshots and execution state persistence."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def save_repl_session(
        self,
        session_id: str,
        snapshot_blob: bytes,
        variables: dict[str, str],
        profile_scope: str = "default",
    ) -> Result[None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_save_repl_session,
                session_id,
                snapshot_blob,
                variables,
                profile_scope,
            )

    def _sync_save_repl_session(
        self,
        session_id: str,
        snapshot_blob: bytes,
        variables: dict[str, str],
        profile_scope: str,
    ) -> Result[None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            now = _utc_now().isoformat()
            cur.execute(
                """
                INSERT INTO assistant_repl_sessions (profile_scope, session_id, updated_at, snapshot_blob, variables_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(profile_scope, session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    snapshot_blob = excluded.snapshot_blob,
                    variables_json = excluded.variables_json
                """,
                (profile_scope, session_id, now, snapshot_blob, json.dumps(variables)),
            )
            conn.commit()
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="SAVE_REPL_SESSION_FAILED",
                    message=str(exc),
                    request_id=f"{profile_scope}:{session_id}",
                )
            )
        finally:
            conn.close()

    async def get_repl_session(
        self, session_id: str, profile_scope: str = "default"
    ) -> Result[tuple[bytes, dict[str, str]] | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_repl_session, session_id, profile_scope
            )

    def _sync_get_repl_session(
        self, session_id: str, profile_scope: str
    ) -> Result[tuple[bytes, dict[str, str]] | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT snapshot_blob, variables_json FROM assistant_repl_sessions WHERE profile_scope = ? AND session_id = ?",
                (profile_scope, session_id),
            )
            row = cur.fetchone()
            if row is None:
                return Success(None)
            blob = row["snapshot_blob"]
            variables = json.loads(row["variables_json"] or "{}")
            return Success((blob, variables))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_REPL_SESSION_FAILED",
                    message=str(exc),
                    request_id=f"{profile_scope}:{session_id}",
                )
            )
        finally:
            conn.close()

    async def clear_repl_session(
        self, session_id: str, profile_scope: str | None = "default"
    ) -> Result[None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_clear_repl_session, session_id, profile_scope
            )

    def _sync_clear_repl_session(
        self, session_id: str, profile_scope: str | None
    ) -> Result[None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            if profile_scope is None:
                cur.execute(
                    "DELETE FROM assistant_repl_sessions WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cur.execute(
                    "DELETE FROM assistant_repl_sessions WHERE profile_scope = ? AND session_id = ?",
                    (profile_scope, session_id),
                )
            conn.commit()
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="CLEAR_REPL_SESSION_FAILED",
                    message=str(exc),
                    request_id=f"{profile_scope or '*'}:{session_id}",
                )
            )
        finally:
            conn.close()
