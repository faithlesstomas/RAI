"""Rich History privacy, journal, fusion, episode and query orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.ports import EventJournal
from rai.kernel.records import ActionFailure, DataClass, Observation, ProducerIdentity

from .collectors import (
    AtspiSemanticCollector,
    BrowserSemanticCollector,
    CollectorSupervisor,
    FilesystemProjectCollector,
    GnomeSessionCollector,
    ProcessContextCollector,
    QueueEventSource,
)
from .fusion import DeterministicEpisodeBuilder, DeterministicFusion
from .models import SourceEvent
from .privacy import PrivacyFirewall
from .storage import EncryptedHistoryStore

HISTORY_PRODUCER = ProducerIdentity(
    producer_id="rai.rich-history", kind="privacy-filter", version="1.0.0"
)
MAX_RECENT_SIGNATURES = 4096
MAX_RAW_EVENTS = 10_000
MAX_FUTURE_SKEW = timedelta(minutes=5)


class RichHistoryService:
    """A local-only deterministic history pipeline with provenance."""

    def __init__(  # noqa: PLR0913
        self,
        journal: EventJournal,
        store: EncryptedHistoryStore,
        *,
        firewall: PrivacyFirewall | None = None,
        fusion: DeterministicFusion | None = None,
        episode_builder: DeterministicEpisodeBuilder | None = None,
        collection_enabled: bool | None = None,
        filesystem_roots: tuple[Path, ...] = (),
        retention_interval_seconds: float | None = None,
    ) -> None:
        self.journal = journal
        self.store = store
        self.firewall = firewall or PrivacyFirewall()
        self.fusion = fusion or DeterministicFusion()
        self.episode_builder = episode_builder or DeterministicEpisodeBuilder()
        if retention_interval_seconds is not None and (
            not isinstance(retention_interval_seconds, (int, float))
            or isinstance(retention_interval_seconds, bool)
            or retention_interval_seconds < 1
        ):
            raise ValueError("retention interval must be a number of at least one second")
        self.retention_interval_seconds = retention_interval_seconds
        self._retention_task: asyncio.Task[None] | None = None
        self.last_retention_error: str | None = None
        self.supervisor = CollectorSupervisor(self._collect)
        self.supervisor.enabled = (
            self.store.setting("collection_enabled", False)
            if collection_enabled is None else collection_enabled
        )
        self.supervisor.emergency_stopped = self.store.setting(
            "emergency_stopped", False
        )
        self._recent_events: OrderedDict[str, datetime] = OrderedDict()
        self._raw_events: deque[SourceEvent] = deque(maxlen=MAX_RAW_EVENTS)
        source = QueueEventSource()
        self._source_collectors = {
            "gnome": GnomeSessionCollector(source),
            "atspi": AtspiSemanticCollector(source),
            "process": ProcessContextCollector(source),
            "browser": BrowserSemanticCollector(source),
            "filesystem": FilesystemProjectCollector(source, filesystem_roots),
        }

    async def _collect(
        self, event: SourceEvent
    ) -> Result[Observation | None, ActionFailure]:
        if event.source == "gnome" and event.kind in {
            "session_locked", "session_unlocked"
        }:
            if event.kind == "session_locked":
                self._raw_events.clear()
            await self.supervisor.observe_session_state(
                event.kind == "session_locked"
            )
            return Success(None)
        return await self.ingest(event)

    async def ingest(  # noqa: PLR0911
        self, event: SourceEvent
    ) -> Result[Observation | None, ActionFailure]:
        if (
            not self.supervisor.enabled
            or self.supervisor.session_locked
            or self.supervisor.emergency_stopped
        ):
            return Success(None)
        if event.timestamp > datetime.now(timezone.utc) + MAX_FUTURE_SKEW:
            return Failure(
                self._failure(
                    "INVALID_TIMESTAMP",
                    "collector timestamp exceeds the allowed future skew",
                )
            )
        try:
            event = self._source_collectors[event.source].sanitize(event)
        except (OSError, TypeError, ValueError) as exc:
            return Failure(
                self._failure("INVALID_EVENT", type(exc).__name__)
            )
        if event is None:
            return Success(None)
        try:
            filtered = self.firewall.apply(event)
        except (OSError, TypeError, ValueError) as exc:
            return Failure(
                self._failure("PRIVACY_POLICY_FAILED", type(exc).__name__)
            )
        if filtered is None:
            return Success(None)
        safe = filtered.event
        self._raw_events.append(safe)
        self._prune_raw(datetime.now(timezone.utc))
        payload: dict[str, Any] = {
            "source": safe.source,
            "kind": safe.kind,
            "application_id": safe.application_id,
            "resource_id": safe.resource_id,
            "project": safe.project,
            "origin": safe.origin,
            "url": safe.url,
            "path": safe.path,
            "title": safe.title,
            "selected_text": safe.selected_text,
            "toolkit": safe.toolkit,
            "quality": safe.quality,
            "semantic": safe.payload,
            "privacy": filtered.decision.model_dump(mode="json"),
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        signature = hashlib.sha256(
            json.dumps(
                {
                    "source": safe.source,
                    "kind": safe.kind,
                    "application_id": safe.application_id,
                    "resource_id": safe.resource_id,
                    "project": safe.project,
                    "payload": payload,
                },
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        previous = self._recent_events.get(signature)
        debounce = timedelta(milliseconds=self.fusion.config.debounce_ms)
        if previous is not None and abs(safe.timestamp - previous) <= debounce:
            return Success(None)
        digest = hashlib.sha256(
            (
                safe.timestamp.isoformat()
                + "\0"
                + safe.source
                + "\0"
                + safe.kind
                + "\0"
                + json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
            ).encode()
        ).hexdigest()[:32]
        observation = Observation(
            record_id=f"history:{digest}", timestamp=safe.timestamp,
            producer=HISTORY_PRODUCER, kind=safe.kind, payload=payload,
            data_class=DataClass(filtered.decision.data_class),
        )
        appended = await self.journal.append(self._journal_projection(observation))
        if isinstance(appended, Failure):
            failure = appended.failure()
            return Failure(self._failure("JOURNAL_REJECTED", failure.code, retryable=failure.retryable))
        try:
            observations = (*self.store.observations(), observation)
            observations = tuple({item.record_id: item for item in observations}.values())
            episodes = self.episode_builder.build(self.fusion.fuse(observations))
            self.store.replace(observations, episodes)
            self._remember(signature, safe.timestamp)
            return Success(observation)
        except Exception as exc:  # storage/crypto isolation boundary
            return Failure(self._failure("HISTORY_STORE_FAILED", type(exc).__name__, retryable=True))

    def query(self, **filters: Any) -> tuple[Any, ...]:  # noqa: ANN401
        return self.store.query_episodes(**filters)

    def answer_what_was_i_working_on(
        self, *, since: datetime | None = None, until: datetime | None = None
    ) -> dict[str, Any]:
        episodes = self.store.query_episodes(since=since, until=until)
        applications = tuple(sorted({item for episode in episodes for item in episode.applications}))
        projects = tuple(sorted({item for episode in episodes for item in episode.projects}))
        resources = tuple(sorted({item for episode in episodes for item in episode.resources}))
        evidence = tuple(
            {
                "episode_id": episode.record_id,
                "observation_ids": episode.observation_ids,
                "confidence": episode.confidence,
                "started_at": episode.started_at,
                "ended_at": episode.ended_at,
            }
            for episode in episodes
        )
        return {
            "answer_type": "deterministic_activity_summary",
            "applications": applications,
            "projects": projects,
            "resources": resources,
            "evidence": evidence,
            "remote_tokens": 0,
        }

    async def set_controls(
        self, *, enabled: bool | None = None,
        session_locked: bool | None = None,
        emergency_stop: bool | None = None,
    ) -> None:
        await self.supervisor.set_controls(
            enabled=enabled, session_locked=session_locked,
            emergency_stop=emergency_stop,
        )
        self.store.set_setting("collection_enabled", self.supervisor.enabled)
        self.store.set_setting("emergency_stopped", self.supervisor.emergency_stopped)
        if session_locked or emergency_stop or enabled is False:
            self._raw_events.clear()
        if (
            self.retention_interval_seconds is not None
            and self.supervisor.enabled
            and not self.supervisor.emergency_stopped
        ):
            self._start_retention_task()
        else:
            await self._stop_retention_task()

    async def close(self) -> None:
        await self._stop_retention_task()
        await self.supervisor.stop()

    async def delete_range(self, since: datetime, until: datetime) -> dict[str, Any]:
        self._validate_interval(since, until)
        ids = self.store.observation_ids_between(since, until)
        journal_deleted = await self.journal.delete_observations(ids)
        if isinstance(journal_deleted, Failure):
            raise RuntimeError("source journal deletion could not be verified")
        try:
            result = self.store.delete_observation_ids(ids)
        except Exception as exc:  # storage/crypto isolation boundary
            raise RuntimeError("encrypted history deletion failed") from exc
        result["journal_observations"] = journal_deleted.unwrap()
        result["residual_references"] = self.store.residual_references(ids)
        result["verified"] = result["residual_references"] == 0
        remaining = self.store.observations()
        self.store.replace(
            remaining, self.episode_builder.build(self.fusion.fuse(remaining))
        )
        self.store.secure_checkpoint()
        self._raw_events = deque(
            (
                event
                for event in self._raw_events
                if not since <= event.timestamp <= until
            ),
            maxlen=MAX_RAW_EVENTS,
        )
        return result

    async def delete_preset(self, preset: str) -> dict[str, Any]:
        """Resolve user-facing deletion shortcuts to a verifiable time range."""
        now = datetime.now(timezone.utc)
        durations = {
            "last_10_minutes": timedelta(minutes=10),
            "last_hour": timedelta(hours=1),
            "last_day": timedelta(days=1),
        }
        if preset in durations:
            return await self.delete_range(now - durations[preset], now)
        if preset == "all":
            return await self.delete_range(
                datetime.min.replace(tzinfo=timezone.utc),
                datetime.max.replace(tzinfo=timezone.utc),
            )
        if preset == "current_application_session":
            episodes = self.store.query_episodes()
            if not episodes:
                return await self.delete_range(now, now)
            return await self.delete_range(episodes[-1].started_at, now)
        raise ValueError("unsupported deletion preset")

    async def enforce_retention(
        self, now: datetime | None = None
    ) -> Result[dict[str, Any], ActionFailure]:
        """Apply every TTL across the raw buffer, journal and encrypted store."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            return Failure(self._failure("INVALID_TIMESTAMP", "retention time must be absolute"))
        self._prune_raw(now)
        try:
            ids = self.store.expired_observation_ids(now)
        except Exception as exc:  # storage/crypto isolation boundary
            return Failure(
                self._failure(
                    "RETENTION_STORE_FAILED", type(exc).__name__, retryable=True
                )
            )
        journal_deleted = await self.journal.delete_observations(ids)
        if isinstance(journal_deleted, Failure):
            failure = journal_deleted.failure()
            return Failure(
                self._failure(
                    "RETENTION_JOURNAL_FAILED",
                    failure.code,
                    retryable=failure.retryable,
                )
            )
        try:
            deleted = self.store.delete_observation_ids(ids)
            remaining = self.store.observations()
            self.store.replace(
                remaining,
                self.episode_builder.build(self.fusion.fuse(remaining)),
            )
            derived = self.store.delete_expired_derived(now)
            self.store.secure_checkpoint()
            deleted["episodes"] += derived["episodes"]
            deleted["memories"] += derived["memories"]
            deleted["journal_observations"] = journal_deleted.unwrap()
            deleted["raw_events_remaining"] = len(self._raw_events)
            deleted["residual_references"] = self.store.residual_references(ids)
            deleted["verified"] = deleted["residual_references"] == 0
            return Success(deleted)
        except Exception as exc:  # storage/crypto isolation boundary
            return Failure(
                self._failure(
                    "RETENTION_STORE_FAILED", type(exc).__name__, retryable=True
                )
            )

    @property
    def raw_event_count(self) -> int:
        return len(self._raw_events)

    def _prune_raw(self, now: datetime) -> None:
        cutoff = now - self.store.retention.raw_ttl
        self._raw_events = deque(
            (event for event in self._raw_events if event.timestamp >= cutoff),
            maxlen=MAX_RAW_EVENTS,
        )

    def _start_retention_task(self) -> None:
        if self._retention_task is None or self._retention_task.done():
            self._retention_task = asyncio.create_task(self._retention_loop())

    async def _stop_retention_task(self) -> None:
        if self._retention_task is None:
            return
        self._retention_task.cancel()
        await asyncio.gather(self._retention_task, return_exceptions=True)
        self._retention_task = None

    async def _retention_loop(self) -> None:
        if self.retention_interval_seconds is None:
            raise RuntimeError("retention loop started without an interval")
        while True:
            result = await self.enforce_retention()
            self.last_retention_error = (
                result.failure().code if isinstance(result, Failure) else None
            )
            await asyncio.sleep(self.retention_interval_seconds)

    def _remember(self, signature: str, timestamp: datetime) -> None:
        previous = self._recent_events.pop(signature, None)
        self._recent_events[signature] = (
            max(previous, timestamp) if previous is not None else timestamp
        )
        while len(self._recent_events) > MAX_RECENT_SIGNATURES:
            self._recent_events.popitem(last=False)

    @staticmethod
    def _journal_projection(observation: Observation) -> Observation:
        """Keep private record contents only in the encrypted history store."""
        if DataClass(observation.data_class) != DataClass.PRIVATE:
            return observation
        privacy = observation.payload.get("privacy")
        payload: dict[str, Any] = {
            "source": observation.payload.get("source"),
            "kind": observation.kind,
            "history_record_id": observation.record_id,
        }
        if privacy is not None:
            payload["privacy"] = privacy
        return observation.model_copy(update={"payload": payload})

    @staticmethod
    def _validate_interval(since: datetime, until: datetime) -> None:
        if any(value.tzinfo is None or value.utcoffset() is None for value in (since, until)):
            raise ValueError("time range must use absolute timestamps")
        if until < since:
            raise ValueError("until must not precede since")

    @staticmethod
    def _failure(code: str, message: str, *, retryable: bool = False) -> ActionFailure:
        return ActionFailure(
            producer=HISTORY_PRODUCER, timestamp=datetime.now(timezone.utc),
            request_id="rich-history", capability="history.ingest", code=code,
            message=message, retryable=retryable,
        )
