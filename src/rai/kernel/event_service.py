"""Validated ingest, replay and bounded local subscriptions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError
from returns.result import Failure, Result

from .events import EventBatch, EventCursor, EventEnvelope, EventRecord, JournalFailure
from .ports import EventJournal
from .records import ActionFailure, ActionResult, Observation, PolicyDecision, UsageRecord, parse_record
from .records import DataClass

SUPPORTED_EVENTS = (Observation, ActionResult, ActionFailure, PolicyDecision, UsageRecord)


class EventService:
    """One validation and flow-control boundary shared by local transports."""

    def __init__(self, journal: EventJournal, *, max_event_bytes: int = 256 * 1024) -> None:
        self.journal = journal
        self.max_event_bytes = max_event_bytes

    async def ingest(
        self,
        payload: bytes,
        *,
        data_class: DataClass | None = None,
        clock_source: str = "system-utc",
        clock_uncertainty_ms: int = 0,
    ) -> Result[EventEnvelope, JournalFailure]:
        if len(payload) > self.max_event_bytes:
            return Failure(JournalFailure(code="OVERSIZED_EVENT", message="event exceeds configured size limit"))
        try:
            data: Any = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("event must be a JSON object")
            record = parse_record(data)
            if not isinstance(record, SUPPORTED_EVENTS):
                raise ValueError(f"record_type {record.record_type!r} is not an event")
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            return Failure(JournalFailure(code="INVALID_EVENT", message=str(exc)))
        return await self.journal.append(
            record,  # type: ignore[arg-type]
            data_class=data_class,
            clock_source=clock_source,
            clock_uncertainty_ms=clock_uncertainty_ms,
        )

    async def replay(self, cursor: str, limit: int) -> Result[EventBatch, JournalFailure]:
        try:
            event_cursor = EventCursor(value=cursor)
        except ValidationError as exc:
            return Failure(JournalFailure(code="INVALID_CURSOR", message=str(exc)))
        return await self.journal.read(event_cursor, limit)

    async def subscribe(
        self,
        cursor: EventCursor,
        *,
        batch_size: int = 10,
        queue_size: int = 2,
        poll_interval: float = 0.05,
    ) -> AsyncIterator[Result[EventBatch, JournalFailure]]:
        """Stream bounded batches; overload terminates only the slow subscriber."""
        queue: asyncio.Queue[Result[EventBatch, JournalFailure]] = asyncio.Queue(queue_size)
        stopped = asyncio.Event()

        async def produce() -> None:
            current = cursor
            while not stopped.is_set():
                batch_result = await self.journal.read(current, batch_size)
                if isinstance(batch_result, Failure):
                    await queue.put(batch_result)
                    return
                batch = batch_result.unwrap()
                if batch.events:
                    try:
                        queue.put_nowait(batch_result)
                    except asyncio.QueueFull:
                        overload = Failure(JournalFailure(code="OVERLOADED", message="subscriber queue is full", retryable=True))
                        while not queue.empty():
                            queue.get_nowait()
                        queue.put_nowait(overload)
                        return
                    current = batch.next_cursor
                await asyncio.sleep(poll_interval)

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                yield item
                if isinstance(item, Failure):
                    return
        finally:
            stopped.set()
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
