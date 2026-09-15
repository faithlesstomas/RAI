"""Transactional SQLite reference implementation of MemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any

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

from .ports import MemoryGraphStore, MemoryQuery
from .records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
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
                    data_class TEXT NOT NULL,
                    profile_scope TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
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
                    created_at TEXT NOT NULL
                );
                """
            )
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

    async def commit_terminal(
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
    ) -> Result[AssistantResponse, ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_commit_terminal,
                response,
                manifest,
                assistant_turn,
                memories,
                relations,
            )

    def _sync_commit_terminal(
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
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
                        manifest_id, session_id, turn_id, manifest_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        manifest.record_id,
                        manifest.session_id,
                        manifest.turn_id,
                        manifest.model_dump_json(),
                        manifest.timestamp.isoformat(),
                    ),
                )

                for memory in memories:
                    conn.execute(
                        "INSERT OR IGNORE INTO nodes (node_id, node_type, created_at) VALUES (?, 'memory', ?)",
                        (memory.record_id, memory.timestamp.isoformat()),
                    )
                    cur.execute(
                        """
                        SELECT memory_id FROM memory_lifecycle
                        WHERE profile_scope = ? AND kind = ? AND topic = ? AND status = 'ACTIVE'
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
                                json.dumps({}),
                            ),
                        )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO memories (
                            memory_id, kind, topic, content_json, source_turn_id,
                            data_class, profile_scope, valid_from, valid_until,
                            created_at, producer_json, correlation_id, provenance_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            memory.record_id,
                            memory.kind,
                            memory.topic,
                            json.dumps(memory.content),
                            memory.source_turn_id,
                            _to_data_class_str(memory.data_class),
                            memory.profile_scope,
                            memory.valid_from.isoformat(),
                            memory.valid_until.isoformat() if memory.valid_until else None,
                            memory.timestamp.isoformat(),
                            memory.producer.model_dump_json(),
                            memory.correlation_id,
                            json.dumps([p.model_dump(mode="json") for p in memory.provenance]),
                        ),
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
                            json.dumps({}),
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
                            json.dumps(relation.metadata),
                        ),
                    )

                conn.execute(
                    "UPDATE turns SET status = ? WHERE turn_id = ?",
                    (response.status, response.user_turn_id),
                )

                conn.execute(
                    """
                    INSERT OR REPLACE INTO responses (
                        response_id, request_id, session_id, user_turn_id,
                        assistant_turn_id, manifest_id, status, text,
                        error_message, admitted_memories_json, provenance_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        response.record_id,
                        response.request_id,
                        response.session_id,
                        response.user_turn_id,
                        assistant_turn.record_id if assistant_turn is not None else None,
                        manifest.record_id,
                        response.status,
                        response.text,
                        response.error_message,
                        json.dumps(list(response.admitted_memory_ids)),
                        json.dumps([p.model_dump(mode="json") for p in response.provenance]),
                        response.timestamp.isoformat(),
                    ),
                )

            return Success(response)
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
            producer = ProducerIdentity.model_validate_json(row["producer_json"])
            metadata = json.loads(row["metadata_json"] or "{}")
            turn = ConversationTurn(
                record_id=row["turn_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                producer=producer,
                session_id=row["session_id"],
                role=row["role"],
                text=row["text"],
                reply_to_turn_id=row["reply_to_turn_id"],
                data_class=DataClass(row["data_class"]),
                status=row["status"],
                correlation_id=row["correlation_id"],
                metadata=metadata,
            )
            return Success(turn)
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
            turns = []
            for row in selected:
                turn = ConversationTurn(
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
                turns.append(turn)
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
            now_iso = datetime.now(timezone.utc).isoformat()

            sql = (
                "SELECT m.*, l.status AS lifecycle_status "  # noqa: S608
                "FROM memories m "
                "JOIN memory_lifecycle l ON m.memory_id = l.memory_id "
                "WHERE l.status = 'ACTIVE' "
                "  AND m.profile_scope = ? "
                f"  AND m.data_class IN ({placeholders}) "
                "  AND m.valid_from <= ? "
                "  AND (m.valid_until IS NULL OR m.valid_until > ?) "
                "ORDER BY m.created_at DESC"
            )
            params: list[Any] = [profile_scope, *allowed_classes, now_iso, now_iso]
            cur.execute(sql, params)
            rows = cur.fetchall()

            scored: list[tuple[float, MemoryRecord, str]] = []
            target_topic = query.topic if query else None
            keywords = query.keywords if query else ()

            for row in rows:
                content = json.loads(row["content_json"])
                prov_raw = json.loads(row["provenance_json"])
                provenance = tuple(
                    ProvenanceReference.model_validate(p) for p in prov_raw
                )
                valid_until = (
                    datetime.fromisoformat(row["valid_until"])
                    if row["valid_until"]
                    else None
                )
                record = MemoryRecord(
                    record_id=row["memory_id"],
                    timestamp=datetime.fromisoformat(row["created_at"]),
                    producer=ProducerIdentity.model_validate_json(row["producer_json"]),
                    kind=row["kind"],
                    topic=row["topic"],
                    content=content,
                    source_turn_id=row["source_turn_id"],
                    data_class=DataClass(row["data_class"]),
                    profile_scope=row["profile_scope"],
                    valid_from=datetime.fromisoformat(row["valid_from"]),
                    valid_until=valid_until,
                    provenance=provenance,
                    correlation_id=row["correlation_id"],
                )

                score = 0.0
                reason = "active durable memory"

                if target_topic and record.topic == target_topic:
                    score = 1.0
                    reason = f"topic exact match: '{record.topic}' (score: 1.0)"
                elif keywords:
                    content_str = json.dumps(content).lower()
                    matched = [kw for kw in keywords if kw.lower() in content_str]
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
                    "SELECT memory_id, status FROM memory_lifecycle WHERE memory_id IN (SELECT memory_id FROM memories WHERE source_turn_id = ?)",
                    (turn_id,),
                )
                derived_memories = cur.fetchall()
                for row in derived_memories:
                    mem_id = row["memory_id"]
                    status = row["status"]
                    if status != "SUPERSEDED":
                        conn.execute(
                            "UPDATE memory_lifecycle SET status = 'DELETED', updated_at = ? WHERE memory_id = ?",
                            (now_iso, mem_id),
                        )

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
            relations = [
                MemoryRelation(
                    relation_id=row["relation_id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    kind=MemoryRelationKind(row["kind"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
                for row in rows
            ]
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
