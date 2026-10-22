"""Graph nodes, relations, and memory records persistence for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import sqlite3
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    ProvenanceReference,
)

from ..records import (
    MemoryOperation,
    MemoryOperationKind,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    make_assistant_failure,
)
from .helpers import (
    dependent_memory_ids,
    memory_from_row,
    to_relation_kind_str,
)
class GraphStoreMixin:
    """Persistence and retrieval of memories, relations, and operations."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    @staticmethod
    def _dependent_memory_ids(
        conn: sqlite3.Connection,
        memory_ids: tuple[str, ...],
        profile_scope: str | None = None,
    ) -> tuple[str, ...]:
        return dependent_memory_ids(conn, memory_ids, profile_scope)

    async def get_dependent_memory_ids(
        self, memory_ids: tuple[str, ...], profile_scope: str | None = None
    ) -> Result[tuple[str, ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_dependent_memory_ids, memory_ids, profile_scope
            )

    def _sync_get_dependent_memory_ids(
        self, memory_ids: tuple[str, ...], profile_scope: str | None = None
    ) -> Result[tuple[str, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            return Success(dependent_memory_ids(conn, memory_ids, profile_scope))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_DEPENDENT_MEMORIES_FAILED",
                    message=str(exc),
                    request_id="memory-dependencies",
                )
            )
        finally:
            conn.close()

    async def get_memory(
        self, memory_id: str
    ) -> Result[tuple[MemoryRecord, str] | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_memory, memory_id)

    def _sync_get_memory(
        self, memory_id: str
    ) -> Result[tuple[MemoryRecord, str] | None, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            row = conn.execute(
                """
                SELECT m.*, l.status AS lifecycle_status
                FROM memories AS m
                JOIN memory_lifecycle AS l ON l.memory_id = m.memory_id
                WHERE m.memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                return Success(None)
            return Success((memory_from_row(row), str(row["lifecycle_status"])))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_MEMORY_FAILED",
                    message=str(exc),
                    request_id=memory_id,
                )
            )
        finally:
            conn.close()

    async def get_relations(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: MemoryRelationKind | None = None,
    ) -> Result[tuple[MemoryRelation, ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_relations, source_id, target_id, kind
            )

    def _sync_get_relations(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: MemoryRelationKind | None = None,
    ) -> Result[tuple[MemoryRelation, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            conditions: list[str] = []
            params: list[Any] = []
            if source_id:
                conditions.append("source_id = ?")
                params.append(source_id)
            if target_id:
                conditions.append("target_id = ?")
                params.append(target_id)
            if kind:
                conditions.append("kind = ?")
                params.append(to_relation_kind_str(kind))

            sql = "SELECT * FROM relations"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at ASC"
            cur.execute(sql, params)
            rows = cur.fetchall()
            relations = []
            for row in rows:
                metadata = json.loads(row["metadata_json"] or "{}")
                provenance = tuple(
                    ProvenanceReference.model_validate(item)
                    for item in metadata.pop("provenance", ())
                )
                relations.append(
                    MemoryRelation(
                        relation_id=row["relation_id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        kind=MemoryRelationKind(row["kind"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        confidence=float(metadata.pop("confidence", 1.0)),
                        epistemic_status=metadata.pop("epistemic_status", "asserted"),
                        provenance=provenance,
                        policy_outcome=metadata.pop("policy_outcome", "ALLOW"),
                        eligible=bool(metadata.pop("eligible", True)),
                        metadata=metadata,
                    )
                )
            return Success(tuple(relations))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_RELATIONS_FAILED",
                    message=str(exc),
                    request_id="relations",
                )
            )
        finally:
            conn.close()

    async def list_memory_operations(
        self, profile_scope: str = "default", limit: int = 100
    ) -> Result[tuple[MemoryOperation, ...], ActionFailure]:
        """Return the latest operation records in chronological order."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_list_memory_operations, profile_scope, limit
            )

    def _sync_list_memory_operations(
        self, profile_scope: str, limit: int
    ) -> Result[tuple[MemoryOperation, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            bounded_limit = max(1, min(limit, 1000))
            rows = conn.execute(
                """
                SELECT operation_json FROM memory_operations
                WHERE profile_scope = ?
                ORDER BY created_at DESC, operation_id DESC
                LIMIT ?
                """,
                (profile_scope, bounded_limit),
            ).fetchall()
            operations = tuple(
                MemoryOperation.model_validate_json(row["operation_json"])
                for row in reversed(rows)
            )
            return Success(operations)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="LIST_MEMORY_OPERATIONS_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
                )
            )
        finally:
            conn.close()

    async def replay_memory_projection(
        self, profile_scope: str = "default"
    ) -> Result[tuple[str, ...], ActionFailure]:
        """Reconstruct active memory IDs exclusively from the operation log."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_replay_memory_projection, profile_scope
            )

    def _sync_replay_memory_projection(
        self, profile_scope: str
    ) -> Result[tuple[str, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            rows = conn.execute(
                """
                SELECT operation_json FROM memory_operations
                WHERE profile_scope = ?
                ORDER BY created_at ASC, operation_id ASC
                """,
                (profile_scope,),
            ).fetchall()
            active: dict[str, None] = {}
            for row in rows:
                operation = MemoryOperation.model_validate_json(row["operation_json"])
                if operation.status != "APPLIED":
                    continue
                for memory_id in operation.target_memory_ids:
                    active.pop(memory_id, None)
                if operation.operation != MemoryOperationKind.FORGET.value:
                    for memory_id in operation.result_memory_ids:
                        active[memory_id] = None
            return Success(tuple(active))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="REPLAY_MEMORY_OPERATIONS_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
                )
            )
        finally:
            conn.close()
