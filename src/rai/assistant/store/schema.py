"""Database schema, connection management, and lifecycle for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.ports import LifecycleState
from rai.kernel.records import ActionFailure
from rai.paths import data_dir

from ..query import UNKNOWN_DOMAIN_SCOPE
from ..records import make_assistant_failure

logger = logging.getLogger(__name__)


class SchemaStoreMixin:
    """Connection management, schema initialization, migrations, and lifecycle."""

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
                    domain_scope TEXT NOT NULL DEFAULT 'unknown',
                    purpose TEXT NOT NULL DEFAULT 'assistant',
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
                    domain_scope TEXT NOT NULL DEFAULT 'unknown',
                    purpose TEXT NOT NULL DEFAULT 'assistant',
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
                CREATE TABLE IF NOT EXISTS turn_retrieval_suppressions (
                    turn_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
                    profile_scope TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (turn_id, profile_scope)
                );
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
                CREATE TRIGGER IF NOT EXISTS one_terminal_response_per_turn
                BEFORE INSERT ON responses
                WHEN EXISTS (
                    SELECT 1 FROM responses
                    WHERE user_turn_id = NEW.user_turn_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'terminal response already exists for turn');
                END;
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
                CREATE TABLE IF NOT EXISTS assistant_schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
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
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    profile_scope UNINDEXED,
                    kind UNINDEXED,
                    topic,
                    content,
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
                  AND NOT EXISTS (
                    SELECT 1 FROM turn_retrieval_suppressions AS suppression
                    WHERE suppression.turn_id = t.turn_id
                      AND suppression.profile_scope = COALESCE(
                        json_extract(t.metadata_json, '$.profile_scope'), 'default'
                      )
                  )
                """
            )
            conn.execute(
                """
                INSERT INTO memories_fts (
                    memory_id, profile_scope, kind, topic, content
                )
                SELECT m.memory_id, m.profile_scope, m.kind, m.topic, m.content_json
                FROM memories AS m
                WHERE NOT EXISTS (
                    SELECT 1 FROM memories_fts AS f
                    WHERE f.memory_id = m.memory_id
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_memory_operations_profile_time
                ON memory_operations(profile_scope, created_at, operation_id);
                """
            )
            self._migrate_schema(conn)
        self._initialized = True

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        def add_column_if_missing(table: str, col: str, col_type: str) -> None:
            cols = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

        add_column_if_missing("responses", "memory_operations_json", "TEXT NOT NULL DEFAULT '[]'")
        for column in ("producer_json", "correlation_id", "schema_version", "reasoning_content"):
            add_column_if_missing("responses", column, "TEXT")
        add_column_if_missing("manifests", "context_json", "TEXT")
        add_column_if_missing("turns", "domain_scope", "TEXT NOT NULL DEFAULT 'unknown'")
        add_column_if_missing("turns", "purpose", "TEXT NOT NULL DEFAULT 'assistant'")
        add_column_if_missing("memories", "source_type", "TEXT NOT NULL DEFAULT 'conversation_turn'")
        add_column_if_missing("memories", "epistemic_status", "TEXT NOT NULL DEFAULT 'asserted'")
        add_column_if_missing("memories", "domain_scope", "TEXT NOT NULL DEFAULT 'unknown'")
        add_column_if_missing("memories", "purpose", "TEXT NOT NULL DEFAULT 'assistant'")
        add_column_if_missing("memories", "recorded_at", "TEXT")
        conn.execute("UPDATE memories SET recorded_at = created_at WHERE recorded_at IS NULL")
        add_column_if_missing("memories", "expired_at", "TEXT")
        scope_migration = conn.execute(
            "SELECT value FROM assistant_schema_metadata "
            "WHERE key = 'domain_scope_semantics'"
        ).fetchone()
        if scope_migration is None:
            conn.execute(
                "UPDATE turns SET domain_scope = ? WHERE domain_scope = 'general'",
                (UNKNOWN_DOMAIN_SCOPE,),
            )
            conn.execute(
                "UPDATE memories SET domain_scope = ? WHERE domain_scope = 'general'",
                (UNKNOWN_DOMAIN_SCOPE,),
            )
            conn.execute(
                "INSERT INTO assistant_schema_metadata (key, value) VALUES (?, ?)",
                ("domain_scope_semantics", "global-unknown-v1"),
            )

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
