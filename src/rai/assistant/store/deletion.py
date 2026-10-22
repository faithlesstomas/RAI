"""Turn deletion and rollback operations for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from ..records import (
    MemoryOperation,
    MemoryOperationKind,
    make_assistant_failure,
)
from .helpers import dependent_memory_ids


class DeletionStoreMixin:
    """Turn deletion and rollback coordination."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def delete_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_delete_turn, turn_id)

    def _sync_delete_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("SELECT node_id FROM nodes WHERE node_id = ?", (turn_id,))
            if cur.fetchone() is None:
                return Success(0)

            now_iso = _utc_now().isoformat()
            with conn:
                direct_rows = conn.execute(
                    "SELECT memory_id, profile_scope FROM memories "
                    "WHERE source_turn_id = ?",
                    (turn_id,),
                ).fetchall()
                direct_ids = tuple(str(row["memory_id"]) for row in direct_rows)
                dep_ids: tuple[str, ...] = ()
                profile_scopes = {str(row["profile_scope"]) for row in direct_rows}
                for profile_scope in profile_scopes:
                    profile_direct_ids = tuple(
                        str(row["memory_id"])
                        for row in direct_rows
                        if str(row["profile_scope"]) == profile_scope
                    )
                    dep_ids = (
                        *dep_ids,
                        *dependent_memory_ids(
                            conn, profile_direct_ids, profile_scope
                        ),
                    )
                affected_ids = tuple(dict.fromkeys((*direct_ids, *dep_ids)))
                if affected_ids:
                    placeholders = ",".join("?" for _ in affected_ids)
                    query = f"""
                        SELECT memory_id, profile_scope, status
                        FROM memory_lifecycle
                        WHERE memory_id IN ({placeholders})
                    """  # noqa: S608
                    derived_memories = conn.execute(
                        query,
                        affected_ids,
                    ).fetchall()
                else:
                    derived_memories = []
                deleted_by_profile: dict[str, list[str]] = {}
                for row in derived_memories:
                    mem_id = row["memory_id"]
                    status = row["status"]
                    if status == "ACTIVE":
                        conn.execute(
                            "UPDATE memory_lifecycle SET status = 'DELETED', updated_at = ? WHERE memory_id = ?",
                            (now_iso, mem_id),
                        )
                        deleted_by_profile.setdefault(
                            str(row["profile_scope"]), []
                        ).append(str(mem_id))

                for profile_scope, memory_ids in deleted_by_profile.items():
                    operation = MemoryOperation(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=ProducerIdentity(
                            producer_id="assistant-store",
                            kind="store",
                            version="1.0.0",
                        ),
                        operation=MemoryOperationKind.FORGET,
                        trigger="source_deletion",
                        trigger_id=turn_id,
                        profile_scope=profile_scope,
                        target_memory_ids=tuple(memory_ids),
                        active_memory_ids_before=tuple(memory_ids),
                        active_memory_ids_after=(),
                        preconditions=("source_turn_exists",),
                        policy_outcome="ALLOW",
                        status="APPLIED",
                        stage="DELETION",
                        reason="source turn deleted",
                    )
                    conn.execute(
                        """
                        INSERT INTO memory_operations (
                            operation_id, profile_scope, operation, status,
                            trigger_id, operation_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            operation.record_id,
                            operation.profile_scope,
                            str(operation.operation),
                            operation.status,
                            operation.trigger_id,
                            operation.model_dump_json(),
                            now_iso,
                        ),
                    )

                # Source deletion is content deletion, not only retrieval invalidation.
                # The operation log retains opaque IDs and the state transition, while
                # the source-derived memory payload and graph edges are securely erased.
                for row in derived_memories:
                    conn.execute(
                        "DELETE FROM memories_fts WHERE memory_id = ?",
                        (str(row["memory_id"]),),
                    )
                    conn.execute(
                        "DELETE FROM nodes WHERE node_id = ?",
                        (str(row["memory_id"]),),
                    )

                conn.execute("DELETE FROM turns_fts WHERE turn_id = ?", (turn_id,))
                conn.execute("DELETE FROM nodes WHERE node_id = ?", (turn_id,))

            return Success(1)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="DELETE_TURN_FAILED",
                    message=str(exc),
                    request_id=turn_id,
                )
            )
        finally:
            conn.close()

    async def rollback_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        """Roll back an aborted or cancelled turn and all its associated records completely."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_rollback_turn, turn_id)

    def _sync_rollback_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute("SELECT node_id FROM nodes WHERE node_id = ?", (turn_id,))
            exists = cur.fetchone() is not None

            with conn:
                resp_rows = conn.execute(
                    "SELECT assistant_turn_id FROM responses WHERE user_turn_id = ? AND assistant_turn_id IS NOT NULL",
                    (turn_id,),
                ).fetchall()
                reply_rows = conn.execute(
                    "SELECT turn_id FROM turns WHERE reply_to_turn_id = ? AND role = 'assistant'",
                    (turn_id,),
                ).fetchall()
                assistant_turn_ids = {
                    str(r["assistant_turn_id"]) for r in resp_rows
                } | {
                    str(r["turn_id"]) for r in reply_rows
                }

                created_memories = conn.execute(
                    "SELECT memory_id FROM memories WHERE source_turn_id = ?",
                    (turn_id,),
                ).fetchall()
                for mem_row in created_memories:
                    mid = str(mem_row["memory_id"])
                    lc_row = conn.execute(
                        "SELECT supersedes_id FROM memory_lifecycle WHERE memory_id = ?",
                        (mid,),
                    ).fetchone()
                    if lc_row and lc_row["supersedes_id"]:
                        prev_id = str(lc_row["supersedes_id"])
                        conn.execute(
                            "UPDATE memory_lifecycle SET status = 'ACTIVE', superseded_by = NULL WHERE memory_id = ?",
                            (prev_id,),
                        )
                        conn.execute(
                            "UPDATE memories SET expired_at = NULL WHERE memory_id = ?",
                            (prev_id,),
                        )
                    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (mid,))
                    conn.execute("DELETE FROM nodes WHERE node_id = ?", (mid,))

                conn.execute(
                    "DELETE FROM memory_operations WHERE trigger_id = ?",
                    (turn_id,),
                )

                for at_id in assistant_turn_ids:
                    conn.execute("DELETE FROM turns_fts WHERE turn_id = ?", (at_id,))
                    conn.execute("DELETE FROM nodes WHERE node_id = ?", (at_id,))

                if exists:
                    conn.execute("DELETE FROM turns_fts WHERE turn_id = ?", (turn_id,))
                    conn.execute("DELETE FROM nodes WHERE node_id = ?", (turn_id,))

            return Success(1 if exists else 0)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ROLLBACK_TURN_FAILED",
                    message=str(exc),
                    request_id=turn_id,
                )
            )
        finally:
            conn.close()
