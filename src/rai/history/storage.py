"""Encrypted SQLite storage, retention, queries and verifiable deletion."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from rai.kernel.records import Episode, Observation
from rai.paths import data_dir

KEY_BYTES = 32


class KeyUnavailableError(RuntimeError):
    """The per-user encryption key could not be obtained safely."""


class KeyProvider(Protocol):
    def get_or_create(self) -> bytes: ...


class StaticKeyProvider:
    """Explicit provider for tests and embedded deployments."""

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise ValueError("history encryption key must contain 32 bytes")
        self._key = key

    def get_or_create(self) -> bytes:
        return self._key


class SecretServiceKeyProvider:
    """Store the per-user AES key through libsecret's Secret Service client."""

    def __init__(self, executable: str = "secret-tool") -> None:
        self.executable = executable

    def get_or_create(self) -> bytes:
        command = shutil.which(self.executable)
        if command is None:
            raise KeyUnavailableError(
                "Secret Service client is unavailable; Rich History remains disabled"
            )
        lookup = subprocess.run(
            [command, "lookup", "application", "rai", "purpose", "rich-history-v1"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if lookup.returncode == 0 and lookup.stdout.strip():
            try:
                key = base64.b64decode(lookup.stdout.strip(), validate=True)
            except ValueError as exc:
                raise KeyUnavailableError("Secret Service returned an invalid key") from exc
            if len(key) != KEY_BYTES:
                raise KeyUnavailableError("Secret Service returned an invalid key length")
            return key
        key = secrets.token_bytes(KEY_BYTES)
        encoded = base64.b64encode(key).decode()
        stored = subprocess.run(
            [command, "store", "--label=RAI Rich History", "application", "rai", "purpose", "rich-history-v1"],
            input=encoded, check=False, capture_output=True, text=True, timeout=5,
        )
        if stored.returncode != 0:
            raise KeyUnavailableError(
                "Secret Service refused the history key; no activity was persisted"
            )
        return key


@dataclass(frozen=True)
class RetentionPolicy:
    raw_ttl: timedelta = timedelta(minutes=10)
    observation_ttl: timedelta = timedelta(days=30)
    episode_ttl: timedelta = timedelta(days=90)
    memory_ttl: timedelta = timedelta(days=365)


def retention_from_config(config: dict[str, object]) -> RetentionPolicy:
    value = config.get("retention_days", {})
    if not isinstance(value, dict):
        raise ValueError("rich_history.retention_days must be a mapping")

    def days(name: str, default: float) -> timedelta:
        setting = value.get(name, default)
        if not isinstance(setting, (int, float)) or isinstance(setting, bool) or setting <= 0:
            raise ValueError(f"rich_history.retention_days.{name} must be positive")
        return timedelta(days=setting)

    return RetentionPolicy(
        raw_ttl=days("raw", 10 / (24 * 60)),
        observation_ttl=days("observations", 30),
        episode_ttl=days("episodes", 90),
        memory_ttl=days("memories", 365),
    )


class EncryptedHistoryStore:
    """Searchable metadata with authenticated encryption for complete records."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        key_provider: KeyProvider | None = None,
        retention: RetentionPolicy | None = None,
        backup_enabled: bool = False,
    ) -> None:
        self.path = path or data_dir() / "history" / "activity-v1.sqlite3"
        self.retention = retention or RetentionPolicy()
        self._key = (key_provider or SecretServiceKeyProvider()).get_or_create()
        self._cipher = AESGCM(self._key)
        self._initialize()
        if not backup_enabled:
            (self.path.parent / "CACHEDIR.TAG").write_text(
                "Signature: 8a477f597d28d172789f06886806bc55\n"
                "# RAI Rich History is excluded from backup unless explicitly opted in.\n",
                encoding="utf-8",
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
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
                CREATE TABLE IF NOT EXISTS observations (
                    record_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                    application_id TEXT, project TEXT, resource_id TEXT,
                    activity_type TEXT NOT NULL, encrypted_record BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    record_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
                    applications TEXT NOT NULL, projects TEXT NOT NULL, resources TEXT NOT NULL,
                    activity_types TEXT NOT NULL, encrypted_record BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_observations (
                    episode_id TEXT NOT NULL REFERENCES episodes(record_id) ON DELETE CASCADE,
                    observation_id TEXT NOT NULL REFERENCES observations(record_id) ON DELETE CASCADE,
                    PRIMARY KEY (episode_id, observation_id)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    encrypted_record BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_sources (
                    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                    observation_id TEXT NOT NULL REFERENCES observations(record_id) ON DELETE CASCADE,
                    PRIMARY KEY (memory_id, observation_id)
                );
                CREATE TABLE IF NOT EXISTS context_references (
                    context_id TEXT NOT NULL, observation_id TEXT NOT NULL
                        REFERENCES observations(record_id) ON DELETE CASCADE,
                    PRIMARY KEY (context_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS observation_time ON observations(timestamp);
                CREATE INDEX IF NOT EXISTS episode_time ON episodes(started_at, ended_at);
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                """
            )
        self.path.chmod(0o600)

    def setting(self, name: str, default: bool = False) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE name = ?", (name,)
            ).fetchone()
        return default if row is None else row[0] == "true"

    def set_setting(self, name: str, value: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO settings(name,value) VALUES(?,?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (name, "true" if value else "false"),
            )

    def replace(
        self, observations: tuple[Observation, ...], episodes: tuple[Episode, ...]
    ) -> None:
        """Atomically rebuild derived episodes from retained observations."""
        with self._connect() as connection:
            for observation in observations:
                payload = observation.payload
                connection.execute(
                    "INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)",
                    (
                        observation.record_id, observation.timestamp.isoformat(),
                        payload.get("application_id"), payload.get("project"),
                        payload.get("resource_id"), observation.kind,
                        self._encrypt(observation.model_dump(mode="json"), observation.record_id),
                    ),
                )
            connection.execute("DELETE FROM episodes")
            for episode in episodes:
                connection.execute(
                    "INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?)",
                    (
                        episode.record_id, episode.started_at.isoformat(), episode.ended_at.isoformat(),
                        json.dumps(episode.applications), json.dumps(episode.projects),
                        json.dumps(episode.resources), json.dumps(episode.activity_types),
                        self._encrypt(episode.model_dump(mode="json"), episode.record_id),
                    ),
                )
                connection.executemany(
                    "INSERT INTO episode_observations VALUES(?,?)",
                    ((episode.record_id, observation_id) for observation_id in episode.observation_ids),
                )

    def observations(self) -> tuple[Observation, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id,encrypted_record FROM observations ORDER BY timestamp,record_id"
            ).fetchall()
        return tuple(
            Observation.model_validate(self._decrypt(row["encrypted_record"], row["record_id"]))
            for row in rows
        )

    def query_episodes(  # noqa: PLR0913
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        application: str | None = None,
        project: str | None = None,
        resource: str | None = None,
        activity_type: str | None = None,
    ) -> tuple[Episode, ...]:
        clauses: list[str] = []
        values: list[str] = []
        if since:
            clauses.append("ended_at >= ?")
            values.append(since.isoformat())
        if until:
            clauses.append("started_at <= ?")
            values.append(until.isoformat())
        for column, value in (("applications", application), ("projects", project), ("resources", resource), ("activity_types", activity_type)):
            if value:
                clauses.append(f"{column} LIKE ?")
                values.append(f'%"{value}"%')
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT record_id,encrypted_record FROM episodes{where} ORDER BY started_at",  # noqa: S608
                values,
            ).fetchall()
        return tuple(
            Episode.model_validate(self._decrypt(row["encrypted_record"], row["record_id"]))
            for row in rows
        )

    def delete_range(self, since: datetime, until: datetime) -> dict[str, int]:
        if until < since:
            raise ValueError("until must not precede since")
        with self._connect() as connection:
            ids = tuple(
                row[0] for row in connection.execute(
                    "SELECT record_id FROM observations WHERE timestamp BETWEEN ? AND ?",
                    (since.isoformat(), until.isoformat()),
                )
            )
            affected = tuple(
                row[0] for row in connection.execute(
                    "SELECT DISTINCT episode_id FROM episode_observations WHERE observation_id IN "
                    f"({','.join('?' for _ in ids)})", ids,
                )
            ) if ids else ()
            memory_ids = tuple(
                row[0] for row in connection.execute(
                    "SELECT DISTINCT memory_id FROM memory_sources WHERE observation_id IN "
                    f"({','.join('?' for _ in ids)})", ids,
                )
            ) if ids else ()
            for table, column, targets in (
                ("memories", "memory_id", memory_ids),
                ("episodes", "record_id", affected),
                ("observations", "record_id", ids),
            ):
                if targets:
                    connection.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({','.join('?' for _ in targets)})",  # noqa: S608
                        targets,
                    )
        return {"observations": len(ids), "episodes": len(affected), "memories": len(memory_ids)}

    def enforce_retention(self, now: datetime | None = None) -> dict[str, int]:
        now = now or datetime.now(timezone.utc)
        deleted = self.delete_range(datetime.min.replace(tzinfo=timezone.utc), now - self.retention.observation_ttl)
        with self._connect() as connection:
            old_episodes = connection.execute(
                "DELETE FROM episodes WHERE ended_at < ?", ((now - self.retention.episode_ttl).isoformat(),)
            ).rowcount
            old_memories = connection.execute(
                "DELETE FROM memories WHERE created_at < ?", ((now - self.retention.memory_ttl).isoformat(),)
            ).rowcount
        deleted["episodes"] += old_episodes
        deleted["memories"] += old_memories
        return deleted

    def residual_references(self, observation_ids: tuple[str, ...]) -> int:
        if not observation_ids:
            return 0
        placeholders = ",".join("?" for _ in observation_ids)
        with self._connect() as connection:
            return sum(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} IN ({placeholders})",  # noqa: S608
                    observation_ids,
                ).fetchone()[0]
                for table, column in (
                    ("observations", "record_id"), ("episode_observations", "observation_id"),
                    ("memory_sources", "observation_id"), ("context_references", "observation_id"),
                )
            )

    def _encrypt(self, value: dict[str, object], record_id: str) -> bytes:
        nonce = os.urandom(12)
        plaintext = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return nonce + self._cipher.encrypt(nonce, plaintext, record_id.encode())

    def _decrypt(self, value: bytes, record_id: str) -> dict[str, object]:
        nonce, ciphertext = value[:12], value[12:]
        return json.loads(self._cipher.decrypt(nonce, ciphertext, record_id.encode()))
