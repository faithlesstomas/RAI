"""Transactional SQLite implementation of the Stage 2 event journal."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from returns.result import Failure, Result, Success

from rai.paths import data_dir

from .events import (
    EventBatch,
    EventCursor,
    EventEnvelope,
    EventRecord,
    JournalFailure,
    TerminalEvent,
    utc_now,
)
from .records import ActionFailure, ActionResult, DataClass, Observation, parse_record

DEFAULT_MAX_EVENT_BYTES = 256 * 1024
DEFAULT_MAX_BATCH = 100


class SQLiteEventJournal:
    """Durable journal with atomic idempotency and consumer checkpoints.

    SQLite commits are run with WAL and FULL synchronous durability. A returned
    append or acknowledgement has crossed the database commit boundary. An
    interruption before commit is replayed as absent; after commit, retrying the
    same record returns its original sequence.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        self.path = path or data_dir() / "events" / "journal-v1.sqlite3"
        self.max_event_bytes = max_event_bytes
        self.max_batch = max_batch
        self._lock = asyncio.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    record_type TEXT NOT NULL,
                    request_id TEXT,
                    data_class TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    clock_source TEXT NOT NULL,
                    clock_uncertainty_ms INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS consumers (
                    consumer_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS retention_requests (
                    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    requested_before_sequence INTEGER NOT NULL,
                    requested_at TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(events)")
            }
            if "request_id" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN request_id TEXT")
            if "data_class" not in columns:
                connection.execute(
                    "ALTER TABLE events ADD COLUMN data_class TEXT NOT NULL DEFAULT 'LOCAL'"
                )
            self._backfill_request_ids(connection)
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS one_terminal_per_request "
                "ON events(request_id) WHERE request_id IS NOT NULL"
            )
        self.path.chmod(0o600)
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._initialize()

    async def append(
        self,
        record: EventRecord,
        *,
        data_class: DataClass | None = None,
        clock_source: str = "system-utc",
        clock_uncertainty_ms: int = 0,
    ) -> Result[EventEnvelope, JournalFailure]:
        classification = self._classification(record, data_class)
        if isinstance(classification, JournalFailure):
            return Failure(classification)
        invalid = self._validate(record, classification, clock_source, clock_uncertainty_ms)
        if invalid is not None:
            return Failure(invalid)
        record_json = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        encoded = record_json.encode()
        if len(encoded) > self.max_event_bytes:
            return Failure(JournalFailure(code="OVERSIZED_EVENT", message="event exceeds configured size limit", record_id=record.record_id))
        digest = hashlib.sha256(
            classification.value.encode() + b"\0" + encoded
        ).hexdigest()
        accepted_at = utc_now()
        try:
            async with self._lock:
                self._ensure_initialized()
                row = self._append_sync(
                    record,
                    record_json,
                    digest,
                    classification,
                    accepted_at.isoformat(),
                    clock_source,
                    clock_uncertainty_ms,
                )
            return Success(self._envelope(row))
        except _JournalConflict:
            return Failure(JournalFailure(code="CONFLICT", message="record_id was reused with different content", record_id=record.record_id))
        except sqlite3.Error as exc:
            return Failure(JournalFailure(code="JOURNAL_UNAVAILABLE", message=str(exc), record_id=record.record_id, retryable=True))

    @staticmethod
    def _validate(
        record: EventRecord,
        data_class: DataClass,
        clock_source: str,
        clock_uncertainty_ms: int,
    ) -> JournalFailure | None:
        if (
            not isinstance(clock_source, str)
            or not clock_source
            or not isinstance(clock_uncertainty_ms, int)
            or isinstance(clock_uncertainty_ms, bool)
            or clock_uncertainty_ms < 0
        ):
            return JournalFailure(code="INVALID_EVENT", message="invalid clock metadata", record_id=record.record_id)
        if not record.producer.producer_id or not record.producer.kind or not record.producer.version:
            return JournalFailure(code="INVALID_EVENT", message="producer identity is incomplete", record_id=record.record_id)
        if data_class in {DataClass.SECRET, DataClass.BLOCKED}:
            return JournalFailure(code="INVALID_EVENT", message="forbidden data class cannot be persisted", record_id=record.record_id)
        return None

    @staticmethod
    def _classification(
        record: EventRecord, supplied: DataClass | None
    ) -> DataClass | JournalFailure:
        embedded_value = getattr(record, "data_class", None)
        try:
            embedded = DataClass(embedded_value) if embedded_value is not None else None
            explicit = DataClass(supplied) if supplied is not None else None
        except ValueError:
            return JournalFailure(
                code="INVALID_EVENT",
                message="invalid data class",
                record_id=record.record_id,
            )
        if embedded is not None and explicit is not None and embedded != explicit:
            return JournalFailure(
                code="INVALID_EVENT",
                message="envelope data class does not match the enclosed record",
                record_id=record.record_id,
            )
        classification = embedded or explicit
        if classification is None:
            return JournalFailure(
                code="INVALID_EVENT",
                message="data class is required for this event type",
                record_id=record.record_id,
            )
        return classification

    def _append_sync(  # noqa: PLR0913
        self, record: EventRecord, record_json: str, digest: str,
        data_class: DataClass, accepted_at: str,
        clock_source: str, clock_uncertainty_ms: int,
    ) -> sqlite3.Row:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM events WHERE record_id = ?", (record.record_id,)).fetchone()
            if existing is not None:
                if existing["content_sha256"] != digest:
                    raise _JournalConflict(record.record_id)
                connection.commit()
                return existing
            request_id = (
                record.request_id
                if record.record_type in {"action_result", "action_failure"}
                else None
            )
            if request_id is not None:
                terminal = connection.execute(
                    "SELECT * FROM events WHERE request_id = ?", (request_id,)
                ).fetchone()
                if terminal is not None:
                    if terminal["content_sha256"] != digest:
                        raise _JournalConflict(record.record_id)
                    connection.commit()
                    return terminal
            cursor = connection.execute(
                "INSERT INTO events(record_id,record_type,request_id,data_class,content_sha256,accepted_at,clock_source,clock_uncertainty_ms,record_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (record.record_id, record.record_type, request_id, data_class.value, digest, accepted_at, clock_source, clock_uncertainty_ms, record_json),
            )
            sequence = cursor.lastrowid
            connection.commit()
            row = connection.execute("SELECT * FROM events WHERE sequence = ?", (sequence,)).fetchone()
            assert row is not None
            return row
        except _JournalConflict:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def read(self, cursor: EventCursor, limit: int) -> Result[EventBatch, JournalFailure]:
        try:
            sequence = cursor.sequence()
        except ValueError as exc:
            return Failure(JournalFailure(code="INVALID_CURSOR", message=str(exc)))
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > self.max_batch:
            return Failure(JournalFailure(code="OVERLOADED", message=f"limit must be between 1 and {self.max_batch}"))
        try:
            async with self._lock:
                self._ensure_initialized()
                with self._connect() as connection:
                    rows = connection.execute("SELECT * FROM events WHERE sequence > ? ORDER BY sequence LIMIT ?", (sequence, limit)).fetchall()
            envelopes = tuple(self._envelope(row) for row in rows)
            next_sequence = envelopes[-1].sequence if envelopes else sequence
            return Success(EventBatch(events=envelopes, next_cursor=EventCursor.from_sequence(next_sequence)))
        except (sqlite3.Error, ValueError) as exc:
            return Failure(JournalFailure(code="JOURNAL_UNAVAILABLE", message=str(exc), retryable=True))

    async def acknowledge(self, consumer_id: str, cursor: EventCursor) -> Result[EventCursor, JournalFailure]:
        try:
            sequence = cursor.sequence()
        except ValueError as exc:
            return Failure(JournalFailure(code="INVALID_CURSOR", message=str(exc)))
        if not consumer_id:
            return Failure(JournalFailure(code="INVALID_EVENT", message="consumer_id is required"))
        try:
            async with self._lock:
                self._ensure_initialized()
                with self._connect() as connection:
                    maximum = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events").fetchone()[0]
                    current = connection.execute("SELECT sequence FROM consumers WHERE consumer_id = ?", (consumer_id,)).fetchone()
                    if sequence > maximum or (current is not None and sequence < current[0]):
                        return Failure(JournalFailure(code="ACK_REGRESSION", message="acknowledgement is outside the durable forward range"))
                    connection.execute(
                        "INSERT INTO consumers(consumer_id,sequence) VALUES(?,?) ON CONFLICT(consumer_id) DO UPDATE SET sequence=excluded.sequence",
                        (consumer_id, sequence),
                    )
            return Success(cursor)
        except sqlite3.Error as exc:
            return Failure(JournalFailure(code="JOURNAL_UNAVAILABLE", message=str(exc), retryable=True))

    async def position(self, consumer_id: str) -> Result[EventCursor, JournalFailure]:
        try:
            async with self._lock:
                self._ensure_initialized()
                with self._connect() as connection:
                    row = connection.execute("SELECT sequence FROM consumers WHERE consumer_id = ?", (consumer_id,)).fetchone()
            return Success(EventCursor.from_sequence(row[0] if row else 0))
        except sqlite3.Error as exc:
            return Failure(JournalFailure(code="JOURNAL_UNAVAILABLE", message=str(exc), retryable=True))

    async def request_retention(self, cursor: EventCursor, reason: str) -> Result[EventCursor, JournalFailure]:
        """Persist a compaction request; journal evidence is never deleted here."""
        if not reason:
            return Failure(JournalFailure(code="INVALID_EVENT", message="retention reason is required"))
        try:
            sequence = cursor.sequence()
            async with self._lock:
                self._ensure_initialized()
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO retention_requests(requested_before_sequence,requested_at,reason) VALUES(?,?,?)",
                        (sequence, utc_now().isoformat(), reason),
                    )
            return Success(cursor)
        except ValueError as exc:
            return Failure(JournalFailure(code="INVALID_CURSOR", message=str(exc)))
        except sqlite3.Error as exc:
            return Failure(JournalFailure(code="JOURNAL_UNAVAILABLE", message=str(exc), retryable=True))

    async def terminal_for(
        self, request_id: str
    ) -> Result[TerminalEvent | None, JournalFailure]:
        if not request_id:
            return Failure(
                JournalFailure(code="INVALID_EVENT", message="request_id is required")
            )
        try:
            async with self._lock:
                self._ensure_initialized()
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT * FROM events WHERE request_id = ?", (request_id,)
                    ).fetchone()
            if row is None:
                return Success(None)
            record = parse_record(json.loads(row["record_json"]))
            if not isinstance(record, (ActionResult, ActionFailure)):
                raise ValueError("terminal index references a non-terminal record")
            return Success(record)
        except (sqlite3.Error, ValueError) as exc:
            return Failure(
                JournalFailure(
                    code="JOURNAL_UNAVAILABLE", message=str(exc), retryable=True
                )
            )

    @staticmethod
    def _backfill_request_ids(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT sequence,record_json FROM events "
            "WHERE request_id IS NULL AND record_type IN ('action_result','action_failure')"
        ).fetchall()
        for row in rows:
            record = parse_record(json.loads(row["record_json"]))
            if isinstance(record, (ActionResult, ActionFailure)):
                connection.execute(
                    "UPDATE events SET request_id = ? WHERE sequence = ?",
                    (record.request_id, row["sequence"]),
                )

    @staticmethod
    def _envelope(row: sqlite3.Row) -> EventEnvelope:
        record = parse_record(json.loads(row["record_json"]))
        return EventEnvelope(
            sequence=row["sequence"], cursor=EventCursor.from_sequence(row["sequence"]),
            accepted_at=row["accepted_at"], clock_source=row["clock_source"],
            clock_uncertainty_ms=row["clock_uncertainty_ms"], content_sha256=row["content_sha256"], record=record,
            data_class=row["data_class"],
        )


class _JournalConflict(sqlite3.IntegrityError):
    pass
