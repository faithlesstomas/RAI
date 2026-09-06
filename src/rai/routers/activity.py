"""Authenticated local review and deletion API for Rich History."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, model_validator
from returns.result import Failure

from rai.dependencies import get_rich_history_service
from rai.history.models import SourceEvent
from rai.history.service import RichHistoryService
from rai.kernel.records import Episode, Observation

router = APIRouter(prefix="/api/v1/activity", tags=["Rich History"])


class CollectionControl(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["pause", "resume", "emergency_stop", "clear_emergency_stop", "lock", "unlock"]


class DeleteRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    since: datetime | None = None
    until: datetime | None = None
    preset: Literal[
        "current_application_session",
        "last_10_minutes",
        "last_hour",
        "last_day",
        "all",
    ] | None = None

    @model_validator(mode="after")
    def exactly_one_range_form(self) -> "DeleteRange":
        has_interval = self.since is not None or self.until is not None
        if self.preset is not None and has_interval:
            raise ValueError("use either preset or since/until")
        if self.preset is None and (self.since is None or self.until is None):
            raise ValueError("provide a preset or both since and until")
        return self


@router.post("/observations", response_model=Observation | None)
async def ingest_semantic_observation(
    event: SourceEvent,
    service: RichHistoryService = Depends(get_rich_history_service),
) -> Observation | None:
    result = await service.ingest(event)
    if isinstance(result, Failure):
        raise HTTPException(status_code=503, detail=result.failure().model_dump(mode="json"))
    return result.unwrap()


@router.get("/episodes", response_model=tuple[Episode, ...])
async def query_activity(  # noqa: PLR0913
    since: datetime | None = None,
    until: datetime | None = None,
    application: str | None = None,
    project: str | None = None,
    resource: str | None = None,
    activity_type: str | None = None,
    service: RichHistoryService = Depends(get_rich_history_service),
) -> tuple[Episode, ...]:
    return service.query(
        since=since, until=until, application=application, project=project,
        resource=resource, activity_type=activity_type,
    )


@router.get("/answer")
async def answer_activity_question(
    question: Literal["what_was_i_working_on", "which_applications_were_active"] = Query(),
    since: datetime | None = None,
    until: datetime | None = None,
    service: RichHistoryService = Depends(get_rich_history_service),
) -> dict[str, Any]:
    answer = service.answer_what_was_i_working_on(since=since, until=until)
    if question == "which_applications_were_active":
        return {"answer_type": question, "applications": answer["applications"], "evidence": answer["evidence"], "remote_tokens": 0}
    return answer


@router.get("/collectors")
async def collector_status(
    service: RichHistoryService = Depends(get_rich_history_service),
) -> dict[str, Any]:
    return {
        "collectors": service.supervisor.status(),
        "retention_error": service.last_retention_error,
        "collection": {
            "enabled": service.supervisor.enabled,
            "session_locked": service.supervisor.session_locked,
            "emergency_stopped": service.supervisor.emergency_stopped,
        },
    }


@router.post("/collection")
async def control_collection(
    control: CollectionControl,
    service: RichHistoryService = Depends(get_rich_history_service),
) -> dict[str, Any]:
    values = {
        "pause": {"enabled": False}, "resume": {"enabled": True},
        "emergency_stop": {"emergency_stop": True},
        "clear_emergency_stop": {"emergency_stop": False},
        "lock": {"session_locked": True}, "unlock": {"session_locked": False},
    }
    await service.set_controls(**values[control.action])
    return {
        "enabled": service.supervisor.enabled,
        "session_locked": service.supervisor.session_locked,
        "emergency_stopped": service.supervisor.emergency_stopped,
    }


@router.delete("/episodes")
async def delete_activity(
    interval: DeleteRange,
    service: RichHistoryService = Depends(get_rich_history_service),
) -> dict[str, Any]:
    try:
        if interval.preset is not None:
            return await service.delete_preset(interval.preset)
        assert interval.since is not None and interval.until is not None
        return await service.delete_range(interval.since, interval.until)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
