"""Transactional SQLite reference implementation of MemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
import unicodedata

from returns.result import Failure, Result, Success

from rai.kernel.ports import LifecycleState
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)
from rai.paths import data_dir

from .ports import AssistantSessionSummary, MemoryGraphStore, MemoryQuery
from .records import (
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    MemoryOperation,
    MemoryOperationKind,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    make_assistant_failure,
)

logger = logging.getLogger(__name__)


def _to_data_class_str(dc: DataClass | str) -> str:
    return dc.value if isinstance(dc, DataClass) else str(dc)


def _to_relation_kind_str(rk: MemoryRelationKind | str) -> str:
    return rk.value if isinstance(rk, MemoryRelationKind) else str(rk)


def _searchable_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )


def _fts_query(terms: tuple[str, ...]) -> str:
    """Build a bounded FTS5 OR query from resolver-produced lexical terms."""
    tokens = tuple(
        dict.fromkeys(
            token
            for term in terms
            for token in re.findall(r"[a-z0-9]+", _searchable_text(term))
            if token
        )
    )
    return " OR ".join(f'"{token}"' for token in tokens[:16])


def _relation_metadata(relation: MemoryRelation) -> dict[str, Any]:
    """Persist relation qualifiers inside its inspectable metadata payload."""
    return {
        **relation.metadata,
        "confidence": relation.confidence,
        "epistemic_status": relation.epistemic_status,
        "provenance": [item.model_dump(mode="json") for item in relation.provenance],
        "policy_outcome": str(relation.policy_outcome),
        "eligible": relation.eligible,
    }


def _semantic_memory_content(content: dict[str, Any]) -> str:
    keys = ("subject", "predicate", "value", "preference", "plan", "fact", "attribute")
    semantic = {key: content[key] for key in keys if key in content}
    return json.dumps(semantic or content, sort_keys=True, ensure_ascii=False)


def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
    provenance = tuple(
        ProvenanceReference.model_validate(item)
        for item in json.loads(row["provenance_json"])
    )
    return MemoryRecord(
        record_id=row["memory_id"],
        timestamp=datetime.fromisoformat(row["created_at"]),
        producer=ProducerIdentity.model_validate_json(row["producer_json"]),
        kind=row["kind"],
        topic=row["topic"],
        content=json.loads(row["content_json"]),
        source_turn_id=row["source_turn_id"],
        source_type=row["source_type"],
        data_class=DataClass(row["data_class"]),
        profile_scope=row["profile_scope"],
        epistemic_status=row["epistemic_status"],
        valid_from=datetime.fromisoformat(row["valid_from"]),
        valid_until=(
            datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None
        ),
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
        expired_at=(
            datetime.fromisoformat(row["expired_at"]) if row["expired_at"] else None
        ),
        provenance=provenance,
        correlation_id=row["correlation_id"],
    )


def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    """Restore one immutable turn from its SQLite representation."""
    return ConversationTurn(
        record_id=row["turn_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        producer=ProducerIdentity.model_validate_json(row["producer_json"]),
        session_id=row["session_id"],
        role=row["role"],
        text=row["text"],
        reply_to_turn_id=row["reply_to_turn_id"],
        data_class=DataClass(row["data_class"]),
        status=row["status"],
        correlation_id=row["correlation_id"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


class SQLiteMemoryGraphStore:
    """Graph-aware SQLite store with strict referential integrity via central nodes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or data_dir() / "assistant" / "memory_graph.sqlite3"
        self._lock = asyncio.Lock()
        self._state = LifecycleState.CREATED
        self._initialized = False

    @property
    def state(self) -> LifecycleState:
        return self._state

    def _ensure_dir_and_perms(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)

    def _connect(self) -> sqlite3.Connection:
        self._ensure_dir_and_perms()
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = FULL;")
        conn.execute("PRAGMA secure_delete = ON;")
        return conn

    def _init_db(self, conn: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    reply_to_turn_id TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
                    data_class TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    producer_json TEXT NOT NULL,
                    correlation_id TEXT,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    source_turn_id TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
                    source_type TEXT NOT NULL DEFAULT 'conversation_turn',
                    data_class TEXT NOT NULL,
                    profile_scope TEXT NOT NULL,
                    epistemic_status TEXT NOT NULL DEFAULT 'asserted',
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    recorded_at TEXT NOT NULL,
                    expired_at TEXT,
                    created_at TEXT NOT NULL,
                    producer_json TEXT NOT NULL,
                    correlation_id TEXT,
                    provenance_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_lifecycle (
                    memory_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
                    profile_scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    superseded_by TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
                    supersedes_id TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory_topic
                ON memory_lifecycle(profile_scope, kind, topic)
                WHERE status = 'ACTIVE';
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    response_id TEXT PRIMARY KEY,
                    request_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    user_turn_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    assistant_turn_id TEXT REFERENCES nodes(node_id) ON DELETE SET NULL,
                    manifest_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text TEXT NOT NULL,
                    error_message TEXT,
                    admitted_memories_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifests (
                    manifest_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    context_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_operations (
                    operation_id TEXT PRIMARY KEY,
                    profile_scope TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_id TEXT NOT NULL,
                    operation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    turn_id UNINDEXED,
                    profile_scope UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            conn.execute(
                """
                INSERT INTO turns_fts (turn_id, profile_scope, text)
                SELECT t.turn_id,
                       COALESCE(json_extract(t.metadata_json, '$.profile_scope'), 'default'),
                       t.text
                FROM turns AS t
                WHERE t.role = 'user' AND t.status = 'COMPLETED'
                  AND NOT EXISTS (
                    SELECT 1 FROM turns_fts AS f WHERE f.turn_id = t.turn_id
                  )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_memory_operations_profile_time
                ON memory_operations(profile_scope, created_at, operation_id);
                """
            )
            response_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(responses)")
            }
            if "memory_operations_json" not in response_columns:
                conn.execute(
                    "ALTER TABLE responses ADD COLUMN memory_operations_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            manifest_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(manifests)")
            }
            if "context_json" not in manifest_columns:
                conn.execute("ALTER TABLE manifests ADD COLUMN context_json TEXT")
            memory_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(memories)")
            }
            if "source_type" not in memory_columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN source_type TEXT NOT NULL "
                    "DEFAULT 'conversation_turn'"
                )
            if "epistemic_status" not in memory_columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN epistemic_status TEXT NOT NULL "
                    "DEFAULT 'asserted'"
                )
            if "recorded_at" not in memory_columns:
                conn.execute("ALTER TABLE memories ADD COLUMN recorded_at TEXT")
                conn.execute(
                    "UPDATE memories SET recorded_at = created_at WHERE recorded_at IS NULL"
                )
            if "expired_at" not in memory_columns:
                conn.execute("ALTER TABLE memories ADD COLUMN expired_at TEXT")
        self._initialized = True

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        async with self._lock:
            try:
                await asyncio.to_thread(self._sync_start)
                self._state = LifecycleState.RUNNING
                return Success(self._state)
            except Exception as exc:  # noqa: BLE001
                self._state = LifecycleState.FAILED
                return Failure(
                    make_assistant_failure(
                        code="STORE_START_FAILED",
                        message=f"failed to initialize SQLite graph store: {exc}",
                    )
                )

    def _sync_start(self) -> None:
        conn = self._connect()
        try:
            self._init_db(conn)
        finally:
            conn.close()

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        async with self._lock:
            self._state = LifecycleState.STOPPED
            return Success(self._state)

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
                if row["text"] == turn.text and row["session_id"] == turn.session_id:
                    return Success(turn)
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
                        correlation_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.record_id,
                        turn.session_id,
                        turn.role,
                        turn.text,
                        turn.reply_to_turn_id,
                        _to_data_class_str(turn.data_class),
                        turn.status,
                        turn.timestamp.isoformat(),
                        turn.producer.model_dump_json(),
                        turn.correlation_id,
                        json.dumps(turn.metadata),
                    ),
                )
                if turn.role == "user" and turn.status == "COMPLETED":
                    conn.execute(
                        "DELETE FROM turns_fts WHERE turn_id = ?", (turn.record_id,)
                    )
                    conn.execute(
                        "INSERT INTO turns_fts (turn_id, profile_scope, text) VALUES (?, ?, ?)",
                        (
                            turn.record_id,
                            str(turn.metadata.get("profile_scope", "default")),
                            turn.text,
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

    def _sync_commit_terminal(  # noqa: PLR0912, PLR0913
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
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM responses WHERE request_id = ?", (response.request_id,)
            )
            row = cur.fetchone()
            if row is not None:
                return Success(response)

            now_iso = _utc_now().isoformat()
            recorded_operations = list(operations)
            with conn:
                if assistant_turn is not None:
                    conn.execute(
                        "INSERT OR IGNORE INTO nodes (node_id, node_type, created_at) VALUES (?, 'turn', ?)",
                        (
                            assistant_turn.record_id,
                            assistant_turn.timestamp.isoformat(),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO turns (
                            turn_id, session_id, role, text, reply_to_turn_id,
                            data_class, status, timestamp, producer_json,
                            correlation_id, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            assistant_turn.record_id,
                            assistant_turn.session_id,
                            assistant_turn.role,
                            assistant_turn.text,
                            assistant_turn.reply_to_turn_id,
                            _to_data_class_str(assistant_turn.data_class),
                            assistant_turn.status,
                            assistant_turn.timestamp.isoformat(),
                            assistant_turn.producer.model_dump_json(),
                            assistant_turn.correlation_id,
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
                                        "confidence": memory.content.get(
                                            "confidence", 1.0
                                        ),
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
                        if _semantic_memory_content(
                            previous_content
                        ) != _semantic_memory_content(memory.content):
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
                                            "confidence": memory.content.get(
                                                "confidence", 1.0
                                            ),
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
                                            "confidence": memory.content.get(
                                                "confidence", 1.0
                                            ),
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
                            data_class, profile_scope, epistemic_status,
                            valid_from, valid_until, recorded_at, expired_at,
                            created_at, producer_json, correlation_id, provenance_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory.record_id,
                            memory.kind,
                            memory.topic,
                            json.dumps(memory.content),
                            memory.source_turn_id,
                            memory.source_type,
                            _to_data_class_str(memory.data_class),
                            memory.profile_scope,
                            memory.epistemic_status,
                            memory.valid_from.isoformat(),
                            memory.valid_until.isoformat()
                            if memory.valid_until
                            else None,
                            now_iso,
                            None,
                            memory.timestamp.isoformat(),
                            memory.producer.model_dump_json(),
                            memory.correlation_id,
                            json.dumps(
                                [p.model_dump(mode="json") for p in memory.provenance]
                            ),
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
                                trigger_id=response.user_turn_id,
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
                            _to_relation_kind_str(relation.kind),
                            relation.created_at.isoformat(),
                            json.dumps(_relation_metadata(relation)),
                        ),
                    )

                for operation in recorded_operations:
                    if (
                        operation.operation == MemoryOperationKind.FORGET.value
                        and operation.status == "APPLIED"
                    ):
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
                        created_at, memory_operations_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        response_to_store.record_id,
                        response_to_store.request_id,
                        response_to_store.session_id,
                        response_to_store.user_turn_id,
                        assistant_turn.record_id
                        if assistant_turn is not None
                        else None,
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

    async def get_response_by_request_id(
        self, request_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_response, request_id)

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
            admitted = tuple(json.loads(row["admitted_memories_json"]))
            operation_ids = tuple(json.loads(row["memory_operations_json"] or "[]"))
            prov_raw = json.loads(row["provenance_json"])
            provenance = tuple(ProvenanceReference.model_validate(p) for p in prov_raw)
            resp = AssistantResponse(
                record_id=row["response_id"],
                timestamp=datetime.fromisoformat(row["created_at"]),
                producer=ProducerIdentity(
                    producer_id="assistant-store", kind="store", version="1.0.0"
                ),
                session_id=row["session_id"],
                turn_id=row["assistant_turn_id"] or row["user_turn_id"],
                user_turn_id=row["user_turn_id"],
                request_id=row["request_id"],
                manifest_id=row["manifest_id"],
                text=row["text"],
                status=row["status"],
                error_message=row["error_message"],
                admitted_memory_ids=admitted,
                memory_operation_ids=operation_ids,
                provenance=provenance,
            )
            return Success(resp)
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
            return Success(_turn_from_row(row))
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
            return Success((_memory_from_row(row), str(row["lifecycle_status"])))
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

    async def get_recent_reply_chain(
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_reply_chain, session_id, limit, before_turn_id
            )

    def _sync_get_reply_chain(
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            query = "SELECT * FROM turns WHERE session_id = ? AND status = 'COMPLETED'"
            params: list[Any] = [session_id]
            if before_turn_id:
                cur.execute(
                    "SELECT timestamp FROM turns WHERE turn_id = ?", (before_turn_id,)
                )
                ts_row = cur.fetchone()
                if ts_row:
                    query += " AND timestamp < ?"
                    params.append(ts_row["timestamp"])

            query += " ORDER BY timestamp ASC"
            cur.execute(query, params)
            rows = cur.fetchall()
            selected = rows[-limit:] if len(rows) > limit else rows
            turns = [_turn_from_row(row) for row in selected]
            return Success(tuple(turns))
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

    async def retrieve_relevant_memories(
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_retrieve_memories, profile_scope, query, data_classes, limit
            )

    def _sync_retrieve_memories(
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            allowed_classes = [_to_data_class_str(dc) for dc in data_classes]
            placeholders = ",".join("?" for _ in allowed_classes)
            now = datetime.now(timezone.utc)
            transaction_at = query.transaction_at if query else None
            valid_at = query.valid_at if query else None
            transaction_iso = (transaction_at or now).isoformat()
            valid_iso = (valid_at or transaction_at or now).isoformat()

            sql = (
                "SELECT m.*, l.status AS lifecycle_status "  # noqa: S608
                "FROM memories m "
                "JOIN memory_lifecycle l ON m.memory_id = l.memory_id "
                "WHERE m.profile_scope = ? "
                f"  AND m.data_class IN ({placeholders}) "
                "  AND m.recorded_at <= ? "
                "  AND (m.expired_at IS NULL OR m.expired_at > ?) "
                "  AND m.valid_from <= ? "
                "  AND (m.valid_until IS NULL OR m.valid_until > ?) "
                "ORDER BY m.created_at DESC"
            )
            params: list[Any] = [
                profile_scope,
                *allowed_classes,
                transaction_iso,
                transaction_iso,
                valid_iso,
                valid_iso,
            ]
            cur.execute(sql, params)
            rows = cur.fetchall()

            scored: list[tuple[float, MemoryRecord, str]] = []
            target_topic = query.topic if query else None
            keywords = query.keywords if query else ()

            for row in rows:
                record = _memory_from_row(row)
                content = record.content

                score = 0.0
                reason = "active durable memory"

                if target_topic and record.topic == target_topic:
                    score = 1.0
                    reason = f"topic exact match: '{record.topic}' (score: 1.0)"
                elif keywords:
                    content_str = _searchable_text(
                        f"{record.topic} {json.dumps(content, ensure_ascii=False)}"
                    )
                    matched = [
                        kw for kw in keywords if _searchable_text(kw) in content_str
                    ]
                    if matched:
                        score = 0.5 + 0.1 * len(matched)
                        reason = f"lexical match on {matched} (score: {score:.1f})"

                if target_topic is None and not keywords:
                    score = 0.1

                if score > 0.0:
                    scored.append((score, record, reason))

            scored.sort(key=lambda item: item[0], reverse=True)
            result = tuple((rec, reason) for _, rec, reason in scored[:limit])
            return Success(result)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RETRIEVE_MEMORIES_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
                )
            )
        finally:
            conn.close()

    async def retrieve_relevant_turns(
        self,
        profile_scope: str,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        exclude_turn_ids: tuple[str, ...] = (),
        limit: int = 5,
    ) -> Result[tuple[tuple[ConversationTurn, str], ...], ActionFailure]:
        """Retrieve source turns independently of the durable claim projection."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_retrieve_turns,
                profile_scope,
                query,
                data_classes,
                exclude_turn_ids,
                limit,
            )

    def _sync_retrieve_turns(
        self,
        profile_scope: str,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        exclude_turn_ids: tuple[str, ...],
        limit: int,
    ) -> Result[tuple[tuple[ConversationTurn, str], ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            allowed_classes = [_to_data_class_str(dc) for dc in data_classes]
            placeholders = ",".join("?" for _ in allowed_classes)
            fts_query = _fts_query(query.keywords)
            excluded = set(exclude_turn_ids)
            query_terms = tuple(
                dict.fromkeys(_searchable_text(term) for term in query.keywords if term)
            )
            broad_recall = any(
                marker in _searchable_text(query.raw_text)
                for marker in (
                    "co wiesz o mnie",
                    "co pamietasz",
                    "what do you know about me",
                    "what do you remember",
                )
            )
            if fts_query:
                rows = conn.execute(
                    (
                        "SELECT t.*, bm25(turns_fts) AS bm25_score "  # noqa: S608
                        "FROM turns_fts JOIN turns AS t "
                        "ON t.turn_id = turns_fts.turn_id "
                        "WHERE turns_fts MATCH ? AND turns_fts.profile_scope = ? "
                        "AND t.role = 'user' AND t.status = 'COMPLETED' "
                        f"AND t.data_class IN ({placeholders}) "
                        "ORDER BY bm25_score ASC, t.timestamp DESC LIMIT 200"
                    ),
                    (fts_query, profile_scope, *allowed_classes),
                ).fetchall()
            elif broad_recall:
                rows = conn.execute(
                    (
                        "SELECT t.*, 0.0 AS bm25_score FROM turns AS t "  # noqa: S608
                        "WHERE t.role = 'user' AND t.status = 'COMPLETED' "
                        f"AND t.data_class IN ({placeholders}) "
                        "ORDER BY t.timestamp DESC LIMIT 200"
                    ),
                    allowed_classes,
                ).fetchall()
            else:
                rows = []
            scored: list[tuple[float, ConversationTurn, str]] = []
            for recency, row in enumerate(rows):
                turn = _turn_from_row(row)
                if turn.record_id in excluded:
                    continue
                turn_scope = str(
                    getattr(turn, "metadata", {}).get("profile_scope", "default")
                )
                if turn_scope != profile_scope:
                    continue
                searchable = _searchable_text(turn.text)
                matched = tuple(term for term in query_terms if term in searchable)
                if not matched and not broad_recall:
                    continue
                lexical = len(matched) / max(1, len(query_terms))
                bm25_score = float(row["bm25_score"])
                score = lexical + max(0.0, 0.1 - recency / 10_000)
                reason = (
                    "FTS5/BM25 source-turn match on "
                    f"{list(matched)} (bm25: {bm25_score:.4f})"
                    if matched
                    else f"source-turn broad recall (score: {score:.2f})"
                )
                scored.append((score, turn, reason))
            scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
            bounded_limit = max(1, min(limit, 50))
            return Success(
                tuple((turn, reason) for _, turn, reason in scored[:bounded_limit])
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RETRIEVE_TURNS_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
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
                cur.execute(
                    "SELECT memory_id, profile_scope, status FROM memory_lifecycle "
                    "WHERE memory_id IN (SELECT memory_id FROM memories WHERE source_turn_id = ?)",
                    (turn_id,),
                )
                derived_memories = cur.fetchall()
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
                params.append(_to_relation_kind_str(kind))

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
