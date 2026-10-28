"""Observation artifact persistence mixin for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import sqlite3

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    _utc_now,
)

from ..records import (
    ObservationArtifact,
    make_assistant_failure,
)
from .helpers import (
    observation_artifact_from_row,
    to_data_class_str,
)


class StoreObservationsMixin:
    """Observation artifact persistence, lookup, and lifecycle management."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def allocate_artifact_id(
        self, session_id: str
    ) -> Result[str, ActionFailure]:
        """Allocate the next sequential human-friendly alias (e.g. 'obs_1') for the session."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_allocate_artifact_id, session_id)

    def _sync_allocate_artifact_id(
        self, session_id: str
    ) -> Result[str, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT artifact_id FROM observation_artifacts
                WHERE session_id = ? AND artifact_id LIKE 'obs_%'
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            existing_nums: set[int] = set()
            for row in rows:
                art_id = str(row[0])
                try:
                    num = int(art_id.split("_", 1)[1])
                    existing_nums.add(num)
                except (IndexError, ValueError):
                    pass
            candidate_num = 1
            while candidate_num in existing_nums:
                candidate_num += 1
            return Success(f"obs_{candidate_num}")
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ALLOCATE_ARTIFACT_ID_FAILED",
                    message=str(exc),
                    request_id=session_id,
                )
            )
        finally:
            conn.close()

    async def put_observation_artifact(
        self, artifact: ObservationArtifact
    ) -> Result[ObservationArtifact, ActionFailure]:
        """Persist a complete capability execution artifact."""
        if artifact.data_class in (DataClass.SECRET, DataClass.BLOCKED):
            return Failure(
                make_assistant_failure(
                    code="PRIVACY_VIOLATION",
                    message=f"SECRET and BLOCKED artifacts cannot be stored in ObservationStore: {artifact.data_class}",
                    request_id=artifact.request_id,
                )
            )
        async with self._lock:
            return await asyncio.to_thread(self._sync_put_artifact, artifact)

    def _sync_put_artifact(
        self, artifact: ObservationArtifact
    ) -> Result[ObservationArtifact, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            with conn:
                try:
                    self._execute_put_artifact(conn, artifact)
                except sqlite3.IntegrityError as int_exc:
                    if "artifact_id" in str(int_exc):
                        alloc_res = self._sync_allocate_artifact_id(artifact.session_id)
                        if isinstance(alloc_res, Success):
                            artifact = artifact.model_copy(update={"artifact_id": alloc_res.unwrap()})
                            self._execute_put_artifact(conn, artifact)
                        else:
                            raise
                    else:
                        raise
            return Success(artifact)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="PUT_ARTIFACT_FAILED",
                    message=str(exc),
                    request_id=artifact.request_id,
                )
            )
        finally:
            conn.close()

    def _execute_put_artifact(
        self, conn: sqlite3.Connection, artifact: ObservationArtifact
    ) -> None:
        conn.execute(
            """
            INSERT INTO observation_artifacts (
                record_id,
                artifact_id,
                session_id,
                turn_id,
                capability,
                request_id,
                data_class,
                content_json,
                schema_summary_json,
                size_chars,
                created_at,
                expires_at,
                provenance_json,
                derived_from_json,
                producer_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                content_json = excluded.content_json,
                schema_summary_json = excluded.schema_summary_json,
                size_chars = excluded.size_chars,
                expires_at = excluded.expires_at,
                provenance_json = excluded.provenance_json,
                derived_from_json = excluded.derived_from_json
            """,
            (
                artifact.record_id,
                artifact.artifact_id,
                artifact.session_id,
                artifact.turn_id,
                artifact.capability,
                artifact.request_id,
                to_data_class_str(artifact.data_class),
                json.dumps(artifact.content, ensure_ascii=False),
                artifact.schema_summary.model_dump_json(),
                artifact.size_chars,
                artifact.created_at.isoformat(),
                artifact.expires_at.isoformat() if artifact.expires_at else None,
                json.dumps([p.model_dump(mode="json") for p in artifact.provenance]),
                json.dumps(list(artifact.derived_from)),
                artifact.producer.model_dump_json(),
            ),
        )

    async def get_observation_artifact(
        self, session_id: str, artifact_id_or_stable_id: str, now: datetime | None = None
    ) -> Result[ObservationArtifact | None, ActionFailure]:
        """Lookup an artifact by session alias (e.g. 'obs_1') or stable record_id."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_artifact, session_id, artifact_id_or_stable_id, now
            )

    def _sync_get_artifact(
        self, session_id: str, identifier: str, now: datetime | None = None
    ) -> Result[ObservationArtifact | None, ActionFailure]:
        if not session_id:
            # Enforce session isolation: an artifact must always be retrieved within a session scope
            return Success(None)

        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            now_iso = (now or _utc_now()).isoformat()
            cur.execute(
                """
                SELECT * FROM observation_artifacts
                WHERE session_id = ?
                  AND (artifact_id = ? OR record_id = ?)
                  AND (expires_at IS NULL OR expires_at >= ?)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, identifier, identifier, now_iso),
            )
            row = cur.fetchone()
            if row is None:
                return Success(None)
            return Success(observation_artifact_from_row(row))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="GET_ARTIFACT_FAILED",
                    message=str(exc),
                    request_id=identifier,
                )
            )
        finally:
            conn.close()

    async def list_observation_artifacts_for_session(
        self, session_id: str, limit: int = 100, now: datetime | None = None
    ) -> Result[tuple[ObservationArtifact, ...], ActionFailure]:
        """List all observation artifacts belonging to a session in chronological order."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_list_artifacts_for_session, session_id, limit, now
            )

    def _sync_list_artifacts_for_session(
        self, session_id: str, limit: int, now: datetime | None = None
    ) -> Result[tuple[ObservationArtifact, ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            now_iso = (now or _utc_now()).isoformat()
            cur.execute(
                """
                SELECT * FROM observation_artifacts
                WHERE session_id = ?
                  AND (expires_at IS NULL OR expires_at >= ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, now_iso, limit),
            )
            rows = cur.fetchall()
            return Success(tuple(observation_artifact_from_row(row) for row in rows))
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="LIST_ARTIFACTS_FAILED",
                    message=str(exc),
                    request_id=session_id,
                )
            )
        finally:
            conn.close()

    async def delete_observation_artifacts_for_session(
        self, session_id: str
    ) -> Result[int, ActionFailure]:
        """Delete all observation artifacts associated with a session."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_delete_artifacts_for_session, session_id
            )

    def _sync_delete_artifacts_for_session(
        self, session_id: str
    ) -> Result[int, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            with conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM observation_artifacts WHERE session_id = ?",
                    (session_id,),
                )
                deleted = cur.rowcount
            return Success(deleted)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="DELETE_ARTIFACTS_FAILED",
                    message=str(exc),
                    request_id=session_id,
                )
            )
        finally:
            conn.close()

    async def prune_expired_observation_artifacts(
        self, now: datetime | None = None
    ) -> Result[int, ActionFailure]:
        """Sweep and purge expired observation artifacts."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_prune_expired_artifacts, now or _utc_now()
            )

    def _sync_prune_expired_artifacts(
        self, now: datetime
    ) -> Result[int, ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            now_iso = now.isoformat()
            with conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM observation_artifacts WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now_iso,),
                )
                deleted = cur.rowcount
            return Success(deleted)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="PRUNE_EXPIRED_ARTIFACTS_FAILED",
                    message=str(exc),
                    request_id="sweep",
                )
            )
        finally:
            conn.close()

    # Protocol conformance aliases for ObservationStore Protocol
    async def put(
        self, artifact: ObservationArtifact
    ) -> Result[ObservationArtifact, ActionFailure]:
        return await self.put_observation_artifact(artifact)

    async def get(
        self, session_id: str, artifact_id_or_stable_id: str, now: datetime | None = None
    ) -> Result[ObservationArtifact | None, ActionFailure]:
        return await self.get_observation_artifact(session_id, artifact_id_or_stable_id, now=now)

    async def list_for_session(
        self, session_id: str, limit: int = 100, now: datetime | None = None
    ) -> Result[tuple[ObservationArtifact, ...], ActionFailure]:
        return await self.list_observation_artifacts_for_session(session_id, limit=limit, now=now)

    async def delete_for_session(
        self, session_id: str
    ) -> Result[int, ActionFailure]:
        return await self.delete_observation_artifacts_for_session(session_id)

    async def prune_expired(
        self, now: datetime | None = None
    ) -> Result[int, ActionFailure]:
        return await self.prune_expired_observation_artifacts(now=now)
