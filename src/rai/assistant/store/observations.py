"""Observation artifact persistence mixin for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
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


AES_KEY_BYTES = 32
AUTO_PRUNE_INTERVAL_SECONDS = 60.0


class StoreObservationsMixin:
    """Observation artifact persistence, lookup, and lifecycle management."""

    _lock: asyncio.Lock
    _last_prune_ts: float = 0.0

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    def _get_cipher(self, conn: sqlite3.Connection | None = None) -> AESGCM:
        if getattr(self, "_observation_cipher", None) is not None:
            return self._observation_cipher
        key = getattr(self, "_encryption_key", None)
        if key is None:
            key = self._get_or_create_key(conn=conn)
            self._encryption_key = key
        cipher = AESGCM(key)
        self._observation_cipher = cipher
        return cipher

    def _get_recorded_key_source(
        self, conn: sqlite3.Connection | None = None
    ) -> str | None:
        db_path = getattr(self, "path", None)
        if db_path is None or str(db_path) == ":memory:":
            return None
        if conn is None and not Path(db_path).exists():
            return None
        should_close = False
        target_conn = conn
        if target_conn is None:
            target_conn = self._connect()
            should_close = True
        try:
            cur = target_conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='assistant_schema_metadata'"
            )
            if cur.fetchone() is None:
                return None
            row = cur.execute(
                "SELECT value FROM assistant_schema_metadata WHERE key = 'observation_encryption_source'"
            ).fetchone()
            return str(row[0]) if row else None
        except sqlite3.Error as e:
            raise RuntimeError(
                f"Database error while reading observation encryption source: {e}"
            ) from e
        finally:
            if should_close:
                target_conn.close()

    def _record_key_source(
        self, source: str, conn: sqlite3.Connection | None = None
    ) -> None:
        db_path = getattr(self, "path", None)
        if db_path is None or str(db_path) == ":memory:":
            return
        should_close = False
        target_conn = conn
        if target_conn is None:
            target_conn = self._connect()
            should_close = True
        try:
            with target_conn:
                target_conn.execute(
                    "CREATE TABLE IF NOT EXISTS assistant_schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                target_conn.execute(
                    "INSERT OR REPLACE INTO assistant_schema_metadata (key, value) VALUES ('observation_encryption_source', ?)",
                    (source,),
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to record observation encryption source '{source}': {e}"
            ) from e
        finally:
            if should_close:
                target_conn.close()

    def _has_existing_ciphertexts(
        self, conn: sqlite3.Connection | None = None
    ) -> bool:
        db_path = getattr(self, "path", None)
        if db_path is None or str(db_path) == ":memory:":
            return False
        if conn is None and not Path(db_path).exists():
            return False
        should_close = False
        target_conn = conn
        if target_conn is None:
            target_conn = self._connect()
            should_close = True
        try:
            cur = target_conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='observation_artifacts'"
            )
            if cur.fetchone() is None:
                return False
            row = cur.execute(
                "SELECT 1 FROM observation_artifacts WHERE content_json LIKE 'enc:%' LIMIT 1"
            ).fetchone()
            return row is not None
        except sqlite3.Error as e:
            raise RuntimeError(
                f"Database error while checking for existing ciphertexts in {db_path}: {e}"
            ) from e
        finally:
            if should_close:
                target_conn.close()

    def _get_or_create_key(self, conn: sqlite3.Connection | None = None) -> bytes:
        if getattr(self, "_encryption_key", None) is not None:
            return self._encryption_key
        provider = getattr(self, "key_provider", None)
        if provider is not None:
            return provider.get_or_create()

        db_path = getattr(self, "path", None)
        if db_path is None or str(db_path) == ":memory:":
            if not hasattr(self, "_fallback_memory_key"):
                self._fallback_memory_key = os.urandom(AES_KEY_BYTES)
            return self._fallback_memory_key

        key_file = Path(db_path).parent / ".memory_graph_key"
        recorded_source = self._get_recorded_key_source(conn=conn)

        if recorded_source == "secret_service":
            from rai.history.storage import SecretServiceKeyProvider  # noqa: PLC0415

            return SecretServiceKeyProvider().get_or_create()

        if recorded_source == "key_file":
            if key_file.exists():
                data = key_file.read_bytes()
                if len(data) == AES_KEY_BYTES:
                    return data
                raise RuntimeError(
                    f"Corrupted encryption key file at {key_file}: invalid length {len(data)}"
                )
            raise RuntimeError(f"Recorded encryption key file is missing at {key_file}")

        # No source recorded yet for this database: determine initial source
        if key_file.exists():
            data = key_file.read_bytes()
            if len(data) == AES_KEY_BYTES:
                self._record_key_source("key_file", conn=conn)
                return data
            raise RuntimeError(
                f"Corrupted encryption key file at {key_file}: invalid length {len(data)}"
            )

        has_ciphertexts = self._has_existing_ciphertexts(conn=conn)
        key: bytes | None = None
        try:
            from rai.history.storage import SecretServiceKeyProvider  # noqa: PLC0415

            key = SecretServiceKeyProvider().get_or_create()
        except Exception as e:
            if has_ciphertexts:
                raise RuntimeError(
                    f"Database at {db_path} contains existing encrypted artifacts without recorded metadata, "
                    f"and SecretService keyring is unavailable ({e}). Aborting to prevent data loss."
                ) from e
            logger.debug(
                "SecretServiceKeyProvider unavailable for new store (%s), establishing key file",
                e,
            )

        if key is not None:
            self._record_key_source("secret_service", conn=conn)
            return key

        new_key = os.urandom(AES_KEY_BYTES)
        key_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_file.write_bytes(new_key)
        try:
            key_file.chmod(0o600)
        except OSError:
            pass
        self._record_key_source("key_file", conn=conn)
        return new_key

    def _encrypt_content(
        self,
        content: dict[str, Any],
        record_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> str:
        plaintext = json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        nonce = os.urandom(12)
        cipher = self._get_cipher(conn=conn)
        ciphertext = cipher.encrypt(nonce, plaintext, record_id.encode("utf-8"))
        encoded = base64.b64encode(nonce + ciphertext).decode("ascii")
        return f"enc:{encoded}"

    def _decrypt_content(
        self,
        content_str: str,
        record_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        if not content_str.startswith("enc:"):
            try:
                parsed = json.loads(content_str)
            except Exception:
                parsed = {"text": content_str}
            if conn is not None and isinstance(parsed, dict):
                encrypted = self._encrypt_content(parsed, record_id, conn=conn)
                with conn:
                    conn.execute(
                        "UPDATE observation_artifacts SET content_json = ? WHERE record_id = ?",
                        (encrypted, record_id),
                    )
            return parsed
        raw_bytes = base64.b64decode(content_str[4:])
        nonce = raw_bytes[:12]
        ciphertext = raw_bytes[12:]
        cipher = self._get_cipher(conn=conn)
        decrypted = cipher.decrypt(nonce, ciphertext, record_id.encode("utf-8"))
        return json.loads(decrypted.decode("utf-8"))

    def _maybe_auto_prune(self, conn: sqlite3.Connection) -> None:
        now_ts = time.time()
        if now_ts - getattr(self, "_last_prune_ts", 0.0) > AUTO_PRUNE_INTERVAL_SECONDS:
            try:
                now_iso = _utc_now().isoformat()
                with conn:
                    conn.execute(
                        "DELETE FROM observation_artifacts WHERE expires_at IS NOT NULL AND expires_at < ?",
                        (now_iso,),
                    )
                self._last_prune_ts = now_ts
            except Exception as e:
                logger.debug("Failed to auto-prune expired observation artifacts: %s", e)



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
            self._maybe_auto_prune(conn)
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
                self._encrypt_content(artifact.content, artifact.record_id, conn=conn),
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
            self._maybe_auto_prune(conn)
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
            return Success(
                observation_artifact_from_row(
                    row, decryptor=lambda c, r: self._decrypt_content(c, r, conn=conn)
                )
            )
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
            self._maybe_auto_prune(conn)
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
            return Success(
                tuple(
                    observation_artifact_from_row(
                        row, decryptor=lambda c, r: self._decrypt_content(c, r, conn=conn)
                    )
                    for row in rows
                )
            )
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
