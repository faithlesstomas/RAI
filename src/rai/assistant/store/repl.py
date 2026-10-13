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
        self, session_id: str, snapshot_blob: bytes, variables: dict[str, str]
    ) -> Result[None, ActionFailure]:
        try:
            conn = self._connect()
            cur = conn.cursor()
            now = _utc_now().isoformat()
            cur.execute(
                """
                INSERT INTO assistant_repl_sessions (session_id, updated_at, snapshot_blob, variables_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    snapshot_blob = excluded.snapshot_blob,
                    variables_json = excluded.variables_json
                """,
                (session_id, now, snapshot_blob, json.dumps(variables)),
            )
            conn.commit()
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="SAVE_REPL_SESSION_FAILED",
                    message=str(exc),
                    request_id="repl_session",
                )
            )
        finally:
            conn.close()

    async def get_repl_session(
        self, session_id: str
    ) -> Result[tuple[bytes, dict[str, str]] | None, ActionFailure]:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT snapshot_blob, variables_json FROM assistant_repl_sessions WHERE session_id = ?",
                (session_id,),
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
                    request_id="repl_session",
                )
            )
        finally:
            conn.close()

    async def clear_repl_session(
        self, session_id: str
    ) -> Result[None, ActionFailure]:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM assistant_repl_sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="CLEAR_REPL_SESSION_FAILED",
                    message=str(exc),
                    request_id="repl_session",
                )
            )
        finally:
            conn.close()
