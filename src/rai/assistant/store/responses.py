"""Assistant response, context manifest, and session listing for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure

from ..ports import AssistantSessionSummary
from ..records import (
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    make_assistant_failure,
)
from .helpers import response_from_row


class ResponsesStoreMixin:
    """Retrieval and inspection of responses, manifests, and session history."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def get_response_by_request_id(
        self, request_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_response, request_id)

    async def get_response_by_turn_id(
        self, turn_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_response_by_turn, turn_id)

    def _sync_get_response(
        self, request_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("SELECT * FROM responses WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            if row is None:
                return Success(None)
            return Success(response_from_row(row))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_RESPONSE_FAILED",
                    message=str(exc),
                    request_id=request_id,
                )
            )
        finally:
            conn.close()

    def _sync_get_response_by_turn(
        self, turn_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            row = conn.execute(
                "SELECT * FROM responses WHERE user_turn_id = ? "
                "ORDER BY created_at ASC LIMIT 1",
                (turn_id,),
            ).fetchone()
            return Success(response_from_row(row) if row is not None else None)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_RESPONSE_FAILED",
                    message=str(exc),
                    request_id=turn_id,
                )
            )
        finally:
            conn.close()

    async def get_manifest(
        self, manifest_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_manifest, manifest_id)

    def _sync_get_manifest(
        self, manifest_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT manifest_json FROM manifests WHERE manifest_id = ?",
                (manifest_id,),
            )
            row = cur.fetchone()
            if row is None:
                return Success(None)
            manifest = AssistantContextManifest.model_validate_json(
                row["manifest_json"]
            )
            return Success(manifest)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_MANIFEST_FAILED",
                    message=str(exc),
                    request_id=manifest_id,
                )
            )
        finally:
            conn.close()

    async def get_context_package(
        self, manifest_id: str
    ) -> Result[AssistantContextPackage | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_context_package, manifest_id)

    def _sync_get_context_package(
        self, manifest_id: str
    ) -> Result[AssistantContextPackage | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            row = conn.execute(
                "SELECT context_json FROM manifests WHERE manifest_id = ?",
                (manifest_id,),
            ).fetchone()
            if row is None or not row["context_json"]:
                return Success(None)
            return Success(
                AssistantContextPackage.model_validate_json(row["context_json"])
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_CONTEXT_PACKAGE_FAILED",
                    message=str(exc),
                    request_id=manifest_id,
                )
            )
        finally:
            conn.close()

    async def get_latest_manifest_for_session(
        self, session_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_latest_manifest_for_session, session_id
            )

    def _sync_get_latest_manifest_for_session(
        self, session_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            row = conn.execute(
                """
                SELECT manifest_json FROM manifests
                WHERE session_id = ?
                ORDER BY created_at DESC, manifest_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return Success(None)
            return Success(
                AssistantContextManifest.model_validate_json(row["manifest_json"])
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_LATEST_MANIFEST_FAILED",
                    message=str(exc),
                    request_id=session_id,
                )
            )
        finally:
            conn.close()

    async def list_sessions(
        self, limit: int = 50
    ) -> Result[tuple[AssistantSessionSummary, ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_list_sessions, limit)

    def _sync_list_sessions(
        self, limit: int
    ) -> Result[tuple[AssistantSessionSummary, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            bounded_limit = max(1, min(limit, 200))
            rows = conn.execute(
                """
                SELECT latest.session_id, summary.started_at, summary.updated_at,
                       summary.turn_count, latest.role, latest.text
                FROM turns AS latest
                JOIN (
                    SELECT session_id, MIN(timestamp) AS started_at,
                           MAX(timestamp) AS updated_at, COUNT(*) AS turn_count
                    FROM turns
                    WHERE status = 'COMPLETED'
                    GROUP BY session_id
                ) AS summary
                  ON latest.session_id = summary.session_id
                 AND latest.timestamp = summary.updated_at
                WHERE latest.status = 'COMPLETED'
                ORDER BY summary.updated_at DESC, latest.session_id ASC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
            sessions = tuple(
                AssistantSessionSummary(
                    session_id=str(row["session_id"]),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    turn_count=int(row["turn_count"]),
                    last_role=str(row["role"]),
                    preview=str(row["text"])[:160],
                )
                for row in rows
            )
            return Success(sessions)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="LIST_SESSIONS_FAILED",
                    message=str(exc),
                    request_id="sessions",
                )
            )
        finally:
            conn.close()
