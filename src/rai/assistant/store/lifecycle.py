"""Memory lifecycle, supersession, contradiction, and forget operations."""

from __future__ import annotations

import json
import sqlite3

from rai.kernel.records import (
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from ..records import (
    MemoryOperation,
    MemoryOperationKind,
    MemoryRecord,
    MemoryRelationKind,
)
from .helpers import (
    semantic_memory_content,
    to_data_class_str,
)


def _apply_memories_and_lifecycle(  # noqa: PLR0912, PLR0913
    conn: sqlite3.Connection,
    cur: sqlite3.Cursor,
    memories: tuple[MemoryRecord, ...],
    recorded_operations: list[MemoryOperation],
    now_iso: str,
    user_turn_id: str,
) -> None:
    for memory in memories:
        conn.execute(
            "INSERT OR IGNORE INTO nodes (node_id, node_type, created_at) VALUES (?, 'memory', ?)",
            (memory.record_id, memory.timestamp.isoformat()),
        )
        cur.execute(
            """
            SELECT l.memory_id, m.content_json
            FROM memory_lifecycle AS l
            JOIN memories AS m ON m.memory_id = l.memory_id
            WHERE l.profile_scope = ? AND l.kind = ? AND l.topic = ?
              AND l.status = 'ACTIVE'
            """,
            (memory.profile_scope, memory.kind, memory.topic),
        )
        active_row = cur.fetchone()
        supersedes_id: str | None = None
        if active_row is not None:
            supersedes_id = str(active_row["memory_id"])
            conn.execute(
                """
                UPDATE memory_lifecycle
                SET status = 'SUPERSEDED', superseded_by = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (memory.record_id, now_iso, supersedes_id),
            )
            conn.execute(
                "UPDATE memories SET expired_at = ? WHERE memory_id = ?",
                (now_iso, supersedes_id),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO relations (
                    relation_id, source_id, target_id, kind, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id(),
                    memory.record_id,
                    supersedes_id,
                    MemoryRelationKind.SUPERSEDES.value,
                    now_iso,
                    json.dumps(
                        {
                            "confidence": memory.content.get("confidence", 1.0),
                            "epistemic_status": memory.epistemic_status,
                            "provenance": [
                                item.model_dump(mode="json")
                                for item in memory.provenance
                            ],
                            "policy_outcome": "ALLOW",
                            "eligible": True,
                        }
                    ),
                ),
            )
            previous_content = json.loads(active_row["content_json"])
            if semantic_memory_content(
                previous_content
            ) != semantic_memory_content(memory.content):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO relations (
                        relation_id, source_id, target_id, kind,
                        created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id(),
                        memory.record_id,
                        supersedes_id,
                        MemoryRelationKind.CONTRADICTS.value,
                        now_iso,
                        json.dumps(
                            {
                                "confidence": memory.content.get("confidence", 1.0),
                                "epistemic_status": "contested",
                                "provenance": [
                                    item.model_dump(mode="json")
                                    for item in memory.provenance
                                ],
                                "policy_outcome": "ALLOW",
                                "eligible": True,
                            }
                        ),
                    ),
                )
            if memory.content.get("statement_type") == "correction":
                conn.execute(
                    """
                    INSERT OR REPLACE INTO relations (
                        relation_id, source_id, target_id, kind,
                        created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id(),
                        memory.record_id,
                        supersedes_id,
                        MemoryRelationKind.UPDATES.value,
                        now_iso,
                        json.dumps(
                            {
                                "confidence": memory.content.get("confidence", 1.0),
                                "epistemic_status": memory.epistemic_status,
                                "provenance": [
                                    item.model_dump(mode="json")
                                    for item in memory.provenance
                                ],
                                "policy_outcome": "ALLOW",
                                "eligible": True,
                            }
                        ),
                    ),
                )
        conn.execute(
            """
            INSERT OR REPLACE INTO memories (
                memory_id, kind, topic, content_json, source_turn_id, source_type,
                data_class, profile_scope, domain_scope, purpose, epistemic_status,
                valid_from, valid_until, recorded_at, expired_at,
                created_at, producer_json, correlation_id, provenance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.record_id,
                memory.kind,
                memory.topic,
                json.dumps(memory.content),
                memory.source_turn_id,
                memory.source_type,
                to_data_class_str(memory.data_class),
                memory.profile_scope,
                memory.domain_scope,
                memory.purpose,
                memory.epistemic_status,
                memory.valid_from.isoformat(),
                memory.valid_until.isoformat() if memory.valid_until else None,
                now_iso,
                None,
                memory.timestamp.isoformat(),
                memory.producer.model_dump_json(),
                memory.correlation_id,
                json.dumps([p.model_dump(mode="json") for p in memory.provenance]),
            ),
        )
        conn.execute(
            "DELETE FROM memories_fts WHERE memory_id = ?",
            (memory.record_id,),
        )
        conn.execute(
            """
            INSERT INTO memories_fts (
                memory_id, profile_scope, kind, topic, content
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory.record_id,
                memory.profile_scope,
                memory.kind,
                memory.topic,
                json.dumps(memory.content, ensure_ascii=False),
            ),
        )

        if not any(
            memory.record_id in operation.result_memory_ids
            for operation in recorded_operations
        ):
            before = (supersedes_id,) if supersedes_id else ()
            recorded_operations.append(
                MemoryOperation(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=ProducerIdentity(
                        producer_id="assistant-store",
                        kind="store",
                        version="1.0.0",
                    ),
                    operation=(
                        MemoryOperationKind.SUPERSEDE
                        if supersedes_id
                        else MemoryOperationKind.REMEMBER
                    ),
                    trigger="conversation_turn",
                    trigger_id=user_turn_id,
                    profile_scope=memory.profile_scope,
                    target_memory_ids=before,
                    result_memory_ids=(memory.record_id,),
                    active_memory_ids_before=before,
                    active_memory_ids_after=(memory.record_id,),
                    preconditions=("source_turn_exists", "policy_allowed"),
                    policy_outcome="ALLOW",
                    status="APPLIED",
                    stage="UPDATE" if supersedes_id else "STORAGE",
                    evidence=memory.provenance,
                    proposal_id=memory.content.get("proposal_id"),
                    source_span=memory.content.get("source_span"),
                    span_start=memory.content.get("span_start"),
                    span_end=memory.content.get("span_end"),
                    modality=memory.content.get("modality"),
                    confidence=memory.content.get("confidence"),
                )
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_lifecycle (
                memory_id, profile_scope, kind, topic, status,
                superseded_by, supersedes_id, updated_at
            ) VALUES (?, ?, ?, ?, 'ACTIVE', NULL, ?, ?)
            """,
            (
                memory.record_id,
                memory.profile_scope,
                memory.kind,
                memory.topic,
                supersedes_id,
                now_iso,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO relations (
                relation_id, source_id, target_id, kind, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _new_id(),
                memory.record_id,
                memory.source_turn_id,
                MemoryRelationKind.DERIVED_FROM.value,
                now_iso,
                json.dumps(
                    {
                        "confidence": memory.content.get("confidence", 1.0),
                        "epistemic_status": memory.epistemic_status,
                        "provenance": [
                            item.model_dump(mode="json")
                            for item in memory.provenance
                        ],
                        "policy_outcome": "ALLOW",
                        "eligible": True,
                    }
                ),
            ),
        )


def _apply_forget_operations(
    conn: sqlite3.Connection,
    recorded_operations: list[MemoryOperation],
    now_iso: str,
) -> None:
    for operation in recorded_operations:
        if (
            operation.operation == MemoryOperationKind.FORGET.value
            and operation.status == "APPLIED"
        ):
            source_rows = conn.execute(
                """
                SELECT DISTINCT memories.source_turn_id
                FROM memories
                JOIN memory_lifecycle
                  ON memory_lifecycle.memory_id = memories.memory_id
                WHERE memories.memory_id IN (
                    SELECT value FROM json_each(?)
                )
                  AND memory_lifecycle.profile_scope = ?
                  AND memories.source_turn_id IS NOT NULL
                """,
                (
                    json.dumps(operation.target_memory_ids),
                    operation.profile_scope,
                ),
            ).fetchall()
            for source_row in source_rows:
                source_turn_id = str(source_row["source_turn_id"])
                response_rows = conn.execute(
                    """
                    SELECT assistant_turn_id FROM responses
                    WHERE user_turn_id = ?
                      AND assistant_turn_id IS NOT NULL
                    """,
                    (source_turn_id,),
                ).fetchall()
                suppressed_turn_ids = (
                    source_turn_id,
                    *(str(row["assistant_turn_id"]) for row in response_rows),
                )
                for suppressed_turn_id in suppressed_turn_ids:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO turn_retrieval_suppressions (
                            turn_id, profile_scope, operation_id, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            suppressed_turn_id,
                            operation.profile_scope,
                            operation.record_id,
                            now_iso,
                        ),
                    )
                    conn.execute(
                        "DELETE FROM turns_fts WHERE turn_id = ? AND profile_scope = ?",
                        (suppressed_turn_id, operation.profile_scope),
                    )
            for memory_id in operation.target_memory_ids:
                conn.execute(
                    """
                    UPDATE memory_lifecycle
                    SET status = 'DELETED', updated_at = ?
                    WHERE memory_id = ? AND profile_scope = ? AND status = 'ACTIVE'
                    """,
                    (now_iso, memory_id, operation.profile_scope),
                )
                conn.execute(
                    "UPDATE memories SET expired_at = ? WHERE memory_id = ?",
                    (now_iso, memory_id),
                )
                conn.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?",
                    (memory_id,),
                )
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_operations (
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
                operation.timestamp.isoformat(),
            ),
        )
