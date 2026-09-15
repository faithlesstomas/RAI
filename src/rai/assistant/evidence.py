"""Adapters exposing approved local sources to assistant context retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol
import unicodedata

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass, Episode

from .ports import AssistantEvidence, MemoryQuery
from .records import make_assistant_failure


class RichHistoryQuery(Protocol):
    """Minimal read-only Rich History surface used by the assistant."""

    def query(self, **filters: Any) -> tuple[Any, ...]: ...  # noqa: ANN401


def _searchable(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )


class RichHistoryEvidenceProvider:
    """Retrieve bounded episode evidence without promoting it to a user claim."""

    def __init__(self, service: RichHistoryQuery) -> None:
        self.service = service

    async def retrieve(
        self,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        limit: int,
    ) -> Result[tuple[AssistantEvidence, ...], ActionFailure]:
        try:
            raw_episodes = await asyncio.to_thread(self.service.query)
            episodes = tuple(
                episode for episode in raw_episodes if isinstance(episode, Episode)
            )
            allowed = {
                value if isinstance(value, DataClass) else DataClass(value)
                for value in data_classes
            }
            terms = tuple(_searchable(term) for term in query.keywords if term)
            activity_intent = any(
                marker in _searchable(query.raw_text)
                for marker in (
                    "co robilem",
                    "nad czym pracowalem",
                    "moja aktywnosc",
                    "what was i working on",
                    "my activity",
                )
            )
            ranked: list[tuple[float, Episode, tuple[str, ...]]] = []
            for episode in episodes:
                episode_class = (
                    episode.data_class
                    if isinstance(episode.data_class, DataClass)
                    else DataClass(episode.data_class)
                )
                if episode_class not in allowed:
                    continue
                searchable = _searchable(
                    " ".join(
                        (
                            *episode.applications,
                            *episode.projects,
                            *episode.resources,
                            *episode.activity_types,
                            *episode.outcome_signals,
                        )
                    )
                )
                matched = tuple(term for term in terms if term in searchable)
                if not matched and not activity_intent:
                    continue
                score = len(matched) / max(1, len(terms)) + episode.confidence * 0.1
                ranked.append((score, episode, matched))
            ranked.sort(key=lambda item: (item[0], item[1].ended_at), reverse=True)
            evidence = tuple(
                AssistantEvidence(
                    source_id=episode.record_id,
                    source_type="rich_history_episode",
                    timestamp=episode.ended_at,
                    content={
                        "started_at": episode.started_at.isoformat(),
                        "ended_at": episode.ended_at.isoformat(),
                        "applications": episode.applications,
                        "projects": episode.projects,
                        "resources": episode.resources,
                        "activity_types": episode.activity_types,
                        "outcome_signals": episode.outcome_signals,
                        "confidence": episode.confidence,
                        "observation_ids": episode.observation_ids,
                    },
                    data_class=(
                        episode.data_class
                        if isinstance(episode.data_class, DataClass)
                        else DataClass(episode.data_class)
                    ),
                    ranking_reason=(
                        f"Rich History lexical match on {list(matched)}"
                        if matched
                        else "Rich History activity-intent recency"
                    ),
                )
                for _, episode, matched in ranked[: max(1, min(limit, 50))]
            )
            return Success(evidence)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RICH_HISTORY_RETRIEVAL_FAILED",
                    message=f"Rich History evidence retrieval failed: {type(exc).__name__}",
                    request_id=query.profile_scope,
                )
            )
