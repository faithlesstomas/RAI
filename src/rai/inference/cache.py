"""Persistent, policy-aware cache for validated bounded inference results."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Callable, Protocol, runtime_checkable

from rai.kernel.records import DataClass
from rai.paths import cache_dir

logger = logging.getLogger(__name__)

CACHE_KEY_VERSION = "1.0.0"
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_CACHE_CAPACITY = 512


class CacheLookupStatus(str, Enum):
    """Outcome of a bounded-result cache lookup."""

    HIT = "hit"
    MISS = "miss"


class CacheMissReason(str, Enum):
    """Sanitized reason for a cache miss."""

    NOT_FOUND = "not_found"
    EXPIRED = "expired"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class BoundedResultCacheKey:
    """Opaque cache identity; normalized input is never exposed."""

    digest: str
    model_name: str
    model_artifact_version: str
    task_kind: str
    contract_version: str
    prompt_version: str
    policy_version: str
    version: str = CACHE_KEY_VERSION

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        *,
        model_name: str,
        model_artifact_version: str,
        task_kind: str,
        contract_version: str,
        prompt_version: str,
        policy_version: str,
        normalized_input: Mapping[str, object],
    ) -> BoundedResultCacheKey:
        """Build a stable key from canonical JSON without retaining its input."""
        material = {
            "cache_key_version": CACHE_KEY_VERSION,
            "contract_version": contract_version,
            "model_artifact_version": model_artifact_version,
            "model_name": model_name,
            "normalized_input": normalized_input,
            "policy_version": policy_version,
            "prompt_version": prompt_version,
            "task_kind": task_kind,
        }
        canonical = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return cls(
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            model_name=model_name,
            model_artifact_version=model_artifact_version,
            task_kind=task_kind,
            contract_version=contract_version,
            prompt_version=prompt_version,
            policy_version=policy_version,
        )


@dataclass(frozen=True)
class CacheLookupMetadata:
    """Public lookup metadata that contains no normalized request content."""

    status: CacheLookupStatus
    key_digest: str
    key_version: str
    model_name: str
    model_artifact_version: str
    task_kind: str
    contract_version: str
    prompt_version: str
    policy_version: str
    miss_reason: CacheMissReason | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True)
class CachedBoundedResult:
    """A validated payload with its input and derived classifications."""

    payload_json: str
    input_data_class: DataClass
    result_data_class: DataClass
    metadata: CacheLookupMetadata


@dataclass(frozen=True)
class CacheLookup:
    """Typed cache lookup result."""

    metadata: CacheLookupMetadata
    result: CachedBoundedResult | None = None


@runtime_checkable
class BoundedResultCache(Protocol):
    """Storage boundary used by the processor supervisor."""

    async def lookup(self, key: BoundedResultCacheKey) -> CacheLookup:
        """Return a validated-cache candidate or typed miss metadata."""
        ...

    async def store(
        self,
        key: BoundedResultCacheKey,
        *,
        payload_json: str,
        input_data_class: DataClass,
        result_data_class: DataClass,
    ) -> None:
        """Store one already schema-validated bounded result."""
        ...

    async def invalidate(self, key: BoundedResultCacheKey) -> None:
        """Remove one incompatible or corrupt entry."""
        ...

    async def close(self) -> None:
        """Release storage resources."""
        ...


def _utc_from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


class SQLiteBoundedResultCache:
    """SQLite cache with lazy initialization, TTL and LRU capacity eviction."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_CACHE_CAPACITY,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self.path = path or cache_dir() / "inference" / "bounded-results-v1.sqlite3"
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max_entries
        self._clock = clock
        self._lock = asyncio.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bounded_results (
                    key_digest TEXT PRIMARY KEY,
                    key_version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_artifact_version TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    contract_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    input_data_class TEXT NOT NULL,
                    result_data_class TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS bounded_results_lru "
                "ON bounded_results(last_accessed_at, created_at, key_digest)"
            )
        os.chmod(self.path, 0o600)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await asyncio.to_thread(self._initialize_sync)
        self._initialized = True

    async def lookup(self, key: BoundedResultCacheKey) -> CacheLookup:
        """Look up one entry, deleting it atomically when its TTL has elapsed."""
        async with self._lock:
            try:
                await self._ensure_initialized()
                return await asyncio.to_thread(self._lookup_sync, key)
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Bounded inference cache lookup failed: %s", exc)
                return self._miss(key, CacheMissReason.ERROR)

    def _lookup_sync(self, key: BoundedResultCacheKey) -> CacheLookup:
        now = self._clock()
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT key_version, model_name, model_artifact_version, task_kind, "
                "contract_version, prompt_version, policy_version, payload_json, "
                "input_data_class, result_data_class, created_at, expires_at "
                "FROM bounded_results WHERE key_digest = ?",
                (key.digest,),
            ).fetchone()
            if row is None:
                return self._miss(key, CacheMissReason.NOT_FOUND)
            (
                key_version,
                model_name,
                model_artifact_version,
                task_kind,
                contract_version,
                prompt_version,
                policy_version,
                payload_json,
                input_data_class,
                result_data_class,
                created_at,
                expires_at,
            ) = row
            stored_identity = (
                key_version,
                model_name,
                model_artifact_version,
                task_kind,
                contract_version,
                prompt_version,
                policy_version,
            )
            expected_identity = (
                key.version,
                key.model_name,
                key.model_artifact_version,
                key.task_kind,
                key.contract_version,
                key.prompt_version,
                key.policy_version,
            )
            if stored_identity != expected_identity:
                connection.execute(
                    "DELETE FROM bounded_results WHERE key_digest = ?", (key.digest,)
                )
                return self._miss(key, CacheMissReason.INVALID)
            if expires_at <= now:
                connection.execute(
                    "DELETE FROM bounded_results WHERE key_digest = ?", (key.digest,)
                )
                return self._miss(key, CacheMissReason.EXPIRED)
            try:
                parsed_input_data_class = DataClass(input_data_class)
                parsed_result_data_class = DataClass(result_data_class)
            except ValueError:
                connection.execute(
                    "DELETE FROM bounded_results WHERE key_digest = ?", (key.digest,)
                )
                return self._miss(key, CacheMissReason.INVALID)
            connection.execute(
                "UPDATE bounded_results SET last_accessed_at = ? WHERE key_digest = ?",
                (now, key.digest),
            )

        metadata = CacheLookupMetadata(
            status=CacheLookupStatus.HIT,
            key_digest=key.digest,
            key_version=key.version,
            model_name=key.model_name,
            model_artifact_version=key.model_artifact_version,
            task_kind=key.task_kind,
            contract_version=key.contract_version,
            prompt_version=key.prompt_version,
            policy_version=key.policy_version,
            created_at=_utc_from_epoch(created_at),
            expires_at=_utc_from_epoch(expires_at),
        )
        return CacheLookup(
            metadata=metadata,
            result=CachedBoundedResult(
                payload_json=payload_json,
                input_data_class=parsed_input_data_class,
                result_data_class=parsed_result_data_class,
                metadata=metadata,
            ),
        )

    @staticmethod
    def _miss(key: BoundedResultCacheKey, reason: CacheMissReason) -> CacheLookup:
        return CacheLookup(
            metadata=CacheLookupMetadata(
                status=CacheLookupStatus.MISS,
                key_digest=key.digest,
                key_version=key.version,
                model_name=key.model_name,
                model_artifact_version=key.model_artifact_version,
                task_kind=key.task_kind,
                contract_version=key.contract_version,
                prompt_version=key.prompt_version,
                policy_version=key.policy_version,
                miss_reason=reason,
            )
        )

    async def store(
        self,
        key: BoundedResultCacheKey,
        *,
        payload_json: str,
        input_data_class: DataClass,
        result_data_class: DataClass,
    ) -> None:
        """Upsert one entry and evict expired/least-recently-used rows."""
        async with self._lock:
            try:
                await self._ensure_initialized()
                await asyncio.to_thread(
                    self._store_sync,
                    key, payload_json, input_data_class, result_data_class
                )
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Bounded inference cache store failed: %s", exc)

    def _store_sync(
        self,
        key: BoundedResultCacheKey,
        payload_json: str,
        input_data_class: DataClass,
        result_data_class: DataClass,
    ) -> None:
        now = self._clock()
        expires_at = now + self.ttl_seconds
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM bounded_results WHERE expires_at <= ?", (now,)
            )
            connection.execute(
                """
                INSERT INTO bounded_results (
                    key_digest, key_version, model_name, model_artifact_version,
                    task_kind, contract_version, prompt_version, policy_version,
                    payload_json, input_data_class, result_data_class,
                    created_at, expires_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_digest) DO UPDATE SET
                    key_version = excluded.key_version,
                    model_name = excluded.model_name,
                    model_artifact_version = excluded.model_artifact_version,
                    task_kind = excluded.task_kind,
                    contract_version = excluded.contract_version,
                    prompt_version = excluded.prompt_version,
                    policy_version = excluded.policy_version,
                    payload_json = excluded.payload_json,
                    input_data_class = excluded.input_data_class,
                    result_data_class = excluded.result_data_class,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (
                    key.digest,
                    key.version,
                    key.model_name,
                    key.model_artifact_version,
                    key.task_kind,
                    key.contract_version,
                    key.prompt_version,
                    key.policy_version,
                    payload_json,
                    input_data_class.value,
                    result_data_class.value,
                    now,
                    expires_at,
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM bounded_results
                WHERE key_digest IN (
                    SELECT key_digest FROM bounded_results
                    ORDER BY last_accessed_at DESC, created_at DESC, key_digest DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_entries,),
            )

    async def invalidate(self, key: BoundedResultCacheKey) -> None:
        """Delete one entry; failures are non-fatal because cache is disposable."""
        async with self._lock:
            try:
                await self._ensure_initialized()
                await asyncio.to_thread(self._invalidate_sync, key)
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Bounded inference cache invalidation failed: %s", exc)

    def _invalidate_sync(self, key: BoundedResultCacheKey) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM bounded_results WHERE key_digest = ?", (key.digest,)
            )

    async def close(self) -> None:
        """No-op: each operation owns its short-lived SQLite connection."""
