"""Deterministic event fusion and episode construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta

from rai.kernel.records import DataClass, Episode, Observation, ProducerIdentity, ProvenanceReference

from .models import ActivityFact

FUSION_PRODUCER = ProducerIdentity(producer_id="rai.history-fusion", kind="processor", version="1.0.0")
EPISODE_PRODUCER = ProducerIdentity(producer_id="rai.episode-builder", kind="processor", version="1.0.0")


@dataclass(frozen=True)
class FusionConfig:
    debounce_ms: int = 500
    fusion_window_ms: int = 2000
    version: str = "1.0.0"


class DeterministicFusion:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(self, observations: tuple[Observation, ...]) -> tuple[ActivityFact, ...]:
        ordered = sorted(observations, key=lambda item: (item.timestamp, item.record_id))
        groups: list[list[Observation]] = []
        for observation in ordered:
            if groups and self._same_activity(groups[-1][0], observation):
                if self._signature(groups[-1][-1]) == self._signature(observation):
                    elapsed = observation.timestamp - groups[-1][-1].timestamp
                    if elapsed <= timedelta(milliseconds=self.config.debounce_ms):
                        continue
                groups[-1].append(observation)
            else:
                groups.append([observation])
        return tuple(self._fact(group) for group in groups)

    def _same_activity(self, first: Observation, second: Observation) -> bool:
        elapsed = second.timestamp - first.timestamp
        if elapsed > timedelta(milliseconds=self.config.fusion_window_ms):
            return False
        left, right = first.payload, second.payload
        shared = any(
            left.get(key) and left.get(key) == right.get(key)
            for key in ("application_id", "resource_id", "project")
        )
        return shared or (
            first.correlation_id is not None
            and first.correlation_id == second.correlation_id
        )

    @staticmethod
    def _signature(observation: Observation) -> str:
        return json.dumps(
            {"kind": observation.kind, "payload": observation.payload},
            sort_keys=True, default=str, separators=(",", ":"),
        )

    def _fact(self, group: list[Observation]) -> ActivityFact:
        ids = tuple(item.record_id for item in group)
        digest = hashlib.sha256((self.config.version + "\0" + "\0".join(ids)).encode()).hexdigest()[:24]
        payload: dict[str, object] = {}
        sources: set[str] = set()
        qualities: list[float] = []
        for item in group:
            payload.update(item.payload)
            sources.add(str(item.payload.get("source", item.producer.producer_id)))
            qualities.append(float(item.payload.get("quality", 1.0)))
        classes = [DataClass(item.data_class) for item in group]
        classification = DataClass.PRIVATE if DataClass.PRIVATE in classes else DataClass.LOCAL
        conflicts = tuple(
            f"conflicting {key}: {', '.join(sorted(map(str, values)))}"
            for key in ("application_id", "resource_id", "project")
            if len(values := set(self._values(group, key))) > 1
        )
        return ActivityFact(
            fact_id=f"fact:{digest}", started_at=group[0].timestamp, ended_at=group[-1].timestamp,
            kind=str(payload.get("kind", group[-1].kind)), application_id=self._last(payload, "application_id"),
            resource_id=self._last(payload, "resource_id"), project=self._last(payload, "project"),
            sources=tuple(sorted(sources)), observation_ids=ids,
            confidence=min(qualities) if len(set(self._values(group, "application_id"))) > 1 else sum(qualities) / len(qualities),
            data_class=classification, payload=payload, conflicts=conflicts,
        )

    @staticmethod
    def _last(payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        return str(value) if value else None

    @staticmethod
    def _values(group: list[Observation], key: str) -> list[object]:
        return [item.payload[key] for item in group if item.payload.get(key)]


@dataclass(frozen=True)
class EpisodeConfig:
    idle_gap_seconds: int = 300
    version: str = "1.0.0"


class DeterministicEpisodeBuilder:
    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self.config = config or EpisodeConfig()

    def build(self, facts: tuple[ActivityFact, ...]) -> tuple[Episode, ...]:
        ordered = sorted(facts, key=lambda item: (item.started_at, item.fact_id))
        groups: list[list[ActivityFact]] = []
        for fact in ordered:
            if not groups or self._boundary(groups[-1][-1], fact):
                groups.append([fact])
            else:
                groups[-1].append(fact)
        return tuple(self._episode(group) for group in groups)

    def _boundary(self, previous: ActivityFact, current: ActivityFact) -> bool:
        return any((
            current.started_at - previous.ended_at > timedelta(seconds=self.config.idle_gap_seconds),
            current.kind in {"session_locked", "idle"},
            bool(previous.project and current.project and previous.project != current.project),
        ))

    def _episode(self, group: list[ActivityFact]) -> Episode:
        observation_ids = tuple(dict.fromkeys(item for fact in group for item in fact.observation_ids))
        digest = hashlib.sha256((self.config.version + "\0" + "\0".join(observation_ids)).encode()).hexdigest()[:24]
        applications = tuple(sorted({fact.application_id for fact in group if fact.application_id}))
        resources = tuple(sorted({fact.resource_id for fact in group if fact.resource_id}))
        projects = tuple(sorted({fact.project for fact in group if fact.project}))
        outcomes = tuple(sorted({fact.kind for fact in group if fact.kind in {"save", "build", "test_run", "test_passed", "test_failed"}}))
        classification = DataClass.PRIVATE if any(fact.data_class == DataClass.PRIVATE for fact in group) else DataClass.LOCAL
        return Episode(
            record_id=f"episode:{digest}", producer=EPISODE_PRODUCER,
            started_at=group[0].started_at, ended_at=group[-1].ended_at,
            observation_ids=observation_ids,
            provenance=tuple(ProvenanceReference(
                source_id=item, source_type="observation", source_version="1.0.0",
                relation="derived-from", producer=FUSION_PRODUCER,
            ) for item in observation_ids),
            applications=applications, resources=resources, projects=projects,
            activity_types=tuple(dict.fromkeys(fact.kind for fact in group)),
            outcome_signals=outcomes, confidence=min(fact.confidence for fact in group),
            data_class=classification, builder_version=self.config.version,
        )
