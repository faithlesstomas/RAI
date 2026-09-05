"""Authenticated loopback HTTP API for the durable local event plane."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from rai.dependencies import get_event_service, get_event_journal
from rai.kernel.event_service import EventService
from rai.kernel.events import EventBatch, EventCursor, EventEnvelope, EventRecord
from rai.kernel.ports import EventJournal
from rai.kernel.records import DataClass

router = APIRouter(prefix="/api/v1/events", tags=["Events"])


class IngestMetadata(BaseModel):
    """Validated envelope metadata supplied as ingest query parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    data_class: DataClass | None
    clock_source: str
    clock_uncertainty_ms: int


async def get_ingest_metadata(
    data_class: DataClass | None = Query(None),
    clock_source: str = Query("system-utc"),
    clock_uncertainty_ms: int = Query(0, ge=0),
) -> IngestMetadata:
    return IngestMetadata(
        data_class=data_class,
        clock_source=clock_source,
        clock_uncertainty_ms=clock_uncertainty_ms,
    )


def _status(code: str) -> int:
    return {
        "CONFLICT": 409,
        "OVERSIZED_EVENT": 413,
        "OVERLOADED": 429,
        "JOURNAL_UNAVAILABLE": 503,
    }.get(code, 422)


@router.post("", response_model=EventEnvelope)
async def ingest_event(
    request: Request,
    event: EventRecord,
    metadata: IngestMetadata = Depends(get_ingest_metadata),
    service: EventService = Depends(get_event_service),
) -> EventEnvelope:
    content_length = request.headers.get("content-length")
    try:
        encoded_length = int(content_length) if content_length is not None else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid Content-Length") from exc
    if encoded_length is not None and encoded_length < 0:
        raise HTTPException(status_code=422, detail="invalid Content-Length")
    if encoded_length is not None and encoded_length > service.max_event_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "OVERSIZED_EVENT",
                "message": "event exceeds configured size limit",
                "record_id": None,
                "retryable": False,
            },
        )
    result = await service.ingest(
        json.dumps(
            event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode(),
        data_class=metadata.data_class,
        clock_source=metadata.clock_source,
        clock_uncertainty_ms=metadata.clock_uncertainty_ms,
    )
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(status_code=_status(failure.code), detail=failure.model_dump())
    return result.unwrap()


@router.get("", response_model=EventBatch)
async def replay_events(
    cursor: str,
    limit: int = Query(50, ge=1),
    service: EventService = Depends(get_event_service),
) -> EventBatch:
    result = await service.replay(cursor, limit)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(status_code=_status(failure.code), detail=failure.model_dump())
    return result.unwrap()


@router.post("/consumers/{consumer_id}/ack", response_model=EventCursor)
async def acknowledge_event(
    consumer_id: str,
    cursor: EventCursor,
    journal: EventJournal = Depends(get_event_journal),
) -> EventCursor:
    result = await journal.acknowledge(consumer_id, cursor)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(status_code=_status(failure.code), detail=failure.model_dump())
    return result.unwrap()


@router.get("/subscriptions/{consumer_id}", response_model=EventBatch)
async def poll_subscription(
    consumer_id: str,
    cursor: str,
    limit: int = Query(50, ge=1),
    journal: EventJournal = Depends(get_event_journal),
) -> EventBatch:
    position = await journal.position(consumer_id)
    if isinstance(position, Failure):
        failure = position.failure()
        raise HTTPException(status_code=_status(failure.code), detail=failure.model_dump())
    stored = position.unwrap()
    start = stored if stored.sequence() > 0 else EventCursor(value=cursor)
    result = await journal.read(start, limit)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(status_code=_status(failure.code), detail=failure.model_dump())
    return result.unwrap()
