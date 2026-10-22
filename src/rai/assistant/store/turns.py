"""Conversation turn persistence and reply chain traversal for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import json
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure

from ..records import (
    ConversationTurn,
    make_assistant_failure,
)
from .deletion import DeletionStoreMixin
from .helpers import (
    to_data_class_str,
    turn_from_row,
)


class TurnsStoreMixin:
    """Persistence, lookup, and reply chain traversal for conversation turns."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def accept_turn(
        self, turn: ConversationTurn
    ) -> Result[ConversationTurn, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_accept_turn, turn)

    def _sync_accept_turn(
        self, turn: ConversationTurn
    ) -> Result[ConversationTurn, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("SELECT * FROM turns WHERE turn_id = ?", (turn.record_id,))
            row = cur.fetchone()
            if row is not None:
                stored_turn = turn_from_row(row)
                if stored_turn == turn:
                    return Success(stored_turn)
                return Failure(
                    make_assistant_failure(
                        code="ID_CONFLICT",
                        message=f"turn id {turn.record_id} exists with differing content",
                        request_id=turn.record_id,
                    )
                )

            with conn:
                conn.execute(
                    "INSERT INTO nodes (node_id, node_type, created_at) VALUES (?, 'turn', ?)",
                    (turn.record_id, turn.timestamp.isoformat()),
                )
                conn.execute(
                    """
                    INSERT INTO turns (
                        turn_id, session_id, role, text, reply_to_turn_id,
                        data_class, status, timestamp, producer_json,
                        correlation_id, domain_scope, purpose, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.record_id,
                        turn.session_id,
                        turn.role,
                        turn.text,
                        turn.reply_to_turn_id,
                        to_data_class_str(turn.data_class),
                        turn.status,
                        turn.timestamp.isoformat(),
                        turn.producer.model_dump_json(),
                        turn.correlation_id,
                        turn.domain_scope,
                        turn.purpose,
                        json.dumps(turn.metadata),
                    ),
                )
                if turn.role == "user" and turn.status == "COMPLETED":
                    conn.execute(
                        "DELETE FROM turns_fts WHERE turn_id = ?", (turn.record_id,)
                    )
                    conn.execute(
                        """
                        INSERT INTO turns_fts (turn_id, profile_scope, text)
                        SELECT ?, ?, ?
                        WHERE NOT EXISTS (
                            SELECT 1 FROM turn_retrieval_suppressions
                            WHERE turn_id = ? AND profile_scope = ?
                        )
                        """,
                        (
                            turn.record_id,
                            str(turn.metadata.get("profile_scope", "default")),
                            turn.text,
                            turn.record_id,
                            str(turn.metadata.get("profile_scope", "default")),
                        ),
                    )
            return Success(turn)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ACCEPT_TURN_FAILED",
                    message=str(exc),
                    request_id=turn.record_id,
                )
            )
        finally:
            conn.close()

    async def get_turn(
        self, turn_id: str
    ) -> Result[ConversationTurn | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_turn, turn_id)

    def _sync_get_turn(
        self, turn_id: str
    ) -> Result[ConversationTurn | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,))
            row = cur.fetchone()
            if row is None:
                return Success(None)
            return Success(turn_from_row(row))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_TURN_FAILED",
                    message=str(exc),
                    request_id=turn_id,
                )
            )
        finally:
            conn.close()

    async def get_recent_reply_chain(
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
        profile_scope: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_reply_chain,
                session_id,
                limit,
                before_turn_id,
                profile_scope,
            )

    def _sync_get_reply_chain(  # noqa: PLR0912
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
        profile_scope: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            if limit <= 0:
                return Success(())

            current_id: str | None = None
            before_timestamp: str | None = None
            if before_turn_id:
                before_row = conn.execute(
                    "SELECT session_id, reply_to_turn_id, timestamp FROM turns "
                    "WHERE turn_id = ?",
                    (before_turn_id,),
                ).fetchone()
                if before_row is not None:
                    if str(before_row["session_id"]) != session_id:
                        return Failure(
                            make_assistant_failure(
                                code="REPLY_CHAIN_SESSION_MISMATCH",
                                message="before_turn_id belongs to another session",
                                request_id=before_turn_id,
                            )
                        )
                    current_id = before_row["reply_to_turn_id"]
                    before_timestamp = str(before_row["timestamp"])

            if current_id is None:
                if before_timestamp is None:
                    latest = conn.execute(
                        "SELECT turn_id FROM turns WHERE session_id = ? "
                        "AND status = 'COMPLETED' ORDER BY timestamp DESC LIMIT 1",
                        (session_id,),
                    ).fetchone()
                else:
                    latest = conn.execute(
                        "SELECT turn_id FROM turns WHERE session_id = ? "
                        "AND status = 'COMPLETED' AND timestamp < ? "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (session_id, before_timestamp),
                    ).fetchone()
                current_id = str(latest["turn_id"]) if latest is not None else None

            rows: list[sqlite3.Row] = []
            visited: set[str] = set()
            while current_id is not None and len(rows) < limit:
                if current_id in visited:
                    break
                visited.add(current_id)
                row = conn.execute(
                    "SELECT * FROM turns WHERE turn_id = ? AND session_id = ? "
                    "AND status = 'COMPLETED'",
                    (current_id, session_id),
                ).fetchone()
                if row is None:
                    break
                suppressed = False
                if profile_scope is not None:
                    suppressed = (
                        conn.execute(
                            """
                            SELECT 1 FROM turn_retrieval_suppressions
                            WHERE turn_id = ? AND profile_scope = ?
                            """,
                            (current_id, profile_scope),
                        ).fetchone()
                        is not None
                    )
                if not suppressed:
                    rows.append(row)
                parent_id = row["reply_to_turn_id"]
                if parent_id is not None:
                    current_id = str(parent_id)
                    continue

                # Legacy rows predate explicit reply links. Preserve a bounded,
                # chronological fallback without affecting newly linked branches.
                previous = conn.execute(
                    "SELECT turn_id FROM turns WHERE session_id = ? "
                    "AND status = 'COMPLETED' AND timestamp < ? "
                    "ORDER BY timestamp DESC LIMIT 1",
                    (session_id, row["timestamp"]),
                ).fetchone()
                current_id = str(previous["turn_id"]) if previous is not None else None

            return Success(tuple(turn_from_row(row) for row in reversed(rows)))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_REPLY_CHAIN_FAILED",
                    message=str(exc),
                    request_id=session_id,
                )
            )
        finally:
            conn.close()
