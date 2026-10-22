"""Atomic terminal commit of responses, memories, and relations for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    _utc_now,
)

from ..records import (
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    MemoryOperation,
    MemoryRecord,
    MemoryRelation,
    make_assistant_failure,
)
from .helpers import (
    relation_metadata,
    response_from_row,
    to_data_class_str,
    to_relation_kind_str,
)
from .lifecycle import (
    _apply_forget_operations,
    _apply_memories_and_lifecycle,
)

logger = logging.getLogger(__name__)


def _check_existing_response(
    cur: sqlite3.Cursor, response: AssistantResponse
) -> Result[AssistantResponse, ActionFailure] | None:
    cur.execute(
        """
        SELECT * FROM responses
        WHERE request_id = ? OR user_turn_id = ?
        ORDER BY CASE WHEN request_id = ? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (response.request_id, response.user_turn_id, response.request_id),
    )
    row = cur.fetchone()
    if row is not None:
        if (
            str(row["request_id"]) == response.request_id
            and str(row["user_turn_id"]) != response.user_turn_id
        ):
            return Failure(
                make_assistant_failure(
                    code="ID_CONFLICT",
                    message=f"request id {response.request_id} belongs to another turn",
                    request_id=response.request_id,
                )
            )
        return Success(response_from_row(row))
    return None


class TerminalStoreMixin:
    """Atomic terminal commit coordination."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def commit_terminal(  # noqa: PLR0913
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
        operations: tuple[MemoryOperation, ...] = (),
        context: AssistantContextPackage | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_commit_terminal,
                response,
                manifest,
                assistant_turn,
                memories,
                relations,
                operations,
                context,
            )

    def _sync_commit_terminal(  # noqa: PLR0913
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
        operations: tuple[MemoryOperation, ...] = (),
        context: AssistantContextPackage | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            existing = _check_existing_response(conn.cursor(), response)
            if existing is not None:
                return existing

            now_iso = _utc_now().isoformat()
            recorded_operations = list(operations)
            with conn:
                if assistant_turn is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO nodes (node_id, node_type, created_at) VALUES (?, 'turn', ?)",
                        (assistant_turn.record_id, assistant_turn.timestamp.isoformat()),
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO turns (
                            turn_id, session_id, role, text, reply_to_turn_id,
                            data_class, status, timestamp, producer_json,
                            correlation_id, domain_scope, purpose, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            assistant_turn.record_id,
                            assistant_turn.session_id,
                            assistant_turn.role,
                            assistant_turn.text,
                            assistant_turn.reply_to_turn_id,
                            to_data_class_str(assistant_turn.data_class),
                            assistant_turn.status,
                            assistant_turn.timestamp.isoformat(),
                            assistant_turn.producer.model_dump_json(),
                            assistant_turn.correlation_id,
                            assistant_turn.domain_scope,
                            assistant_turn.purpose,
                            json.dumps(assistant_turn.metadata),
                        ),
                    )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO manifests (
                        manifest_id, session_id, turn_id, manifest_json, created_at,
                        context_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest.record_id,
                        manifest.session_id,
                        manifest.turn_id,
                        manifest.model_dump_json(),
                        manifest.timestamp.isoformat(),
                        context.model_dump_json() if context is not None else None,
                    ),
                )

                _apply_memories_and_lifecycle(
                    conn,
                    conn.cursor(),
                    memories,
                    recorded_operations,
                    now_iso,
                    response.user_turn_id,
                )

                for relation in relations:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO relations (
                            relation_id, source_id, target_id, kind, created_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            relation.relation_id,
                            relation.source_id,
                            relation.target_id,
                            to_relation_kind_str(relation.kind),
                            relation.created_at.isoformat(),
                            json.dumps(relation_metadata(relation)),
                        ),
                    )

                _apply_forget_operations(conn, recorded_operations, now_iso)

                conn.execute(
                    "UPDATE turns SET status = ? WHERE turn_id = ?",
                    (response.status, response.user_turn_id),
                )
                if response.status == "COMPLETED":
                    conn.execute(
                        "DELETE FROM turns_fts WHERE turn_id = ?",
                        (response.user_turn_id,),
                    )
                    conn.execute(
                        """
                        INSERT INTO turns_fts (turn_id, profile_scope, text)
                        SELECT turn_id,
                               COALESCE(json_extract(metadata_json, '$.profile_scope'), 'default'),
                               text
                        FROM turns
                        WHERE turn_id = ? AND role = 'user'
                          AND NOT EXISTS (
                              SELECT 1 FROM turn_retrieval_suppressions AS suppression
                              WHERE suppression.turn_id = turns.turn_id
                                AND suppression.profile_scope = COALESCE(
                                    json_extract(turns.metadata_json, '$.profile_scope'),
                                    'default'
                                )
                          )
                        """,
                        (response.user_turn_id,),
                    )

                response_to_store = response.model_copy(
                    update={
                        "memory_operation_ids": tuple(
                            operation.record_id for operation in recorded_operations
                        )
                    }
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO responses (
                        response_id, request_id, session_id, user_turn_id,
                        assistant_turn_id, manifest_id, status, text,
                        error_message, admitted_memories_json, provenance_json,
                        created_at, memory_operations_json, producer_json, correlation_id,
                        schema_version, reasoning_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        response_to_store.record_id,
                        response_to_store.request_id,
                        response_to_store.session_id,
                        response_to_store.user_turn_id,
                        assistant_turn.record_id if assistant_turn is not None else None,
                        manifest.record_id,
                        response_to_store.status,
                        response_to_store.text,
                        response_to_store.error_message,
                        json.dumps(list(response_to_store.admitted_memory_ids)),
                        json.dumps(
                            [
                                p.model_dump(mode="json")
                                for p in response_to_store.provenance
                            ]
                        ),
                        response_to_store.timestamp.isoformat(),
                        json.dumps(list(response_to_store.memory_operation_ids)),
                        response_to_store.producer.model_dump_json(),
                        response_to_store.correlation_id,
                        response_to_store.schema_version,
                        response_to_store.reasoning_content,
                    ),
                )

            return Success(response_to_store)
        except Exception as exc:  # noqa: BLE001
            logger.exception("commit_terminal failed")
            return Failure(
                make_assistant_failure(
                    code="COMMIT_TERMINAL_FAILED",
                    message=str(exc),
                    request_id=response.request_id,
                )
            )
        finally:
            conn.close()
