"""Rich History privacy, journal, fusion, episode and query orchestration."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.ports import EventJournal
from rai.kernel.records import ActionFailure, DataClass, Observation, ProducerIdentity

from .collectors import CollectorSupervisor
from .fusion import DeterministicEpisodeBuilder, DeterministicFusion
from .models import SourceEvent
from .privacy import PrivacyFirewall
from .storage import EncryptedHistoryStore

HISTORY_PRODUCER = ProducerIdentity(
    producer_id="rai.rich-history", kind="privacy-filter", version="1.0.0"
)


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
    ) -> None:
        self.journal = journal
        self.store = store
        self.firewall = firewall or PrivacyFirewall()
        self.fusion = fusion or DeterministicFusion()
        self.episode_builder = episode_builder or DeterministicEpisodeBuilder()
        self.supervisor = CollectorSupervisor(self._collect)
        self.supervisor.enabled = (
            self.store.setting("collection_enabled", False)
            if collection_enabled is None else collection_enabled
        )
        self.supervisor.emergency_stopped = self.store.setting(
            "emergency_stopped", False
        )
        self._recent_events: dict[str, datetime] = {}

    async def _collect(self, event: SourceEvent) -> None:
        await self.ingest(event)

    async def ingest(
        self, event: SourceEvent
    ) -> Result[Observation | None, ActionFailure]:
        if (
            not self.supervisor.enabled
            or self.supervisor.session_locked
            or self.supervisor.emergency_stopped
        ):
            return Success(None)
        filtered = self.firewall.apply(event)
        if filtered is None:
            return Success(None)
        safe = filtered.event
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
            repr((safe.source, safe.kind, safe.application_id, safe.resource_id, safe.project, payload)).encode()
        ).hexdigest()
        previous = self._recent_events.get(signature)
        debounce = timedelta(milliseconds=self.fusion.config.debounce_ms)
        if previous is not None and safe.timestamp - previous <= debounce:
            return Success(None)
        self._recent_events[signature] = safe.timestamp
        digest = hashlib.sha256(
            (safe.timestamp.isoformat() + "\0" + safe.source + "\0" + safe.kind + "\0" + repr(payload)).encode()
        ).hexdigest()[:32]
        observation = Observation(
            record_id=f"history:{digest}", timestamp=safe.timestamp,
            producer=HISTORY_PRODUCER, kind=safe.kind, payload=payload,
            data_class=DataClass(filtered.decision.data_class),
        )
        appended = await self.journal.append(observation)
        if isinstance(appended, Failure):
            failure = appended.failure()
            return Failure(self._failure("JOURNAL_REJECTED", failure.code, retryable=failure.retryable))
        try:
            observations = (*self.store.observations(), observation)
            observations = tuple({item.record_id: item for item in observations}.values())
            episodes = self.episode_builder.build(self.fusion.fuse(observations))
            self.store.replace(observations, episodes)
            return Success(observation)
        except (OSError, ValueError) as exc:
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

    async def delete_range(self, since: datetime, until: datetime) -> dict[str, Any]:
        ids = tuple(
            item.record_id for item in self.store.observations()
            if since <= item.timestamp <= until
        )
        result = self.store.delete_range(since, until)
        journal_deleted = await self.journal.delete_observations(ids)
        if isinstance(journal_deleted, Failure):
            raise RuntimeError("source journal deletion could not be verified")
        result["journal_observations"] = journal_deleted.unwrap()
        result["residual_references"] = self.store.residual_references(ids)
        result["verified"] = result["residual_references"] == 0
        remaining = self.store.observations()
        self.store.replace(
            remaining, self.episode_builder.build(self.fusion.fuse(remaining))
        )
        return result

    @staticmethod
    def _failure(code: str, message: str, *, retryable: bool = False) -> ActionFailure:
        return ActionFailure(
            producer=HISTORY_PRODUCER, timestamp=datetime.now(timezone.utc),
            request_id="rich-history", capability="history.ingest", code=code,
            message=message, retryable=retryable,
        )
