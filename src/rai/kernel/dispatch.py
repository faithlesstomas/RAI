"""Deterministic Stage 2 observation-to-capability subscriber."""

from __future__ import annotations

from returns.result import Failure, Result, Success

from .events import JournalFailure
from .ports import CancellationToken, EventJournal
from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    Observation,
    ProducerIdentity,
)
from .service import CapabilityService

DISPATCH_PRODUCER = ProducerIdentity(
    producer_id="rai.stage2-dispatch", kind="deterministic-subscriber", version="1.0.0"
)


class DeterministicEventDispatcher:
    """Consume synthetic observations without model reasoning or providers."""

    def __init__(
        self,
        journal: EventJournal,
        capabilities: CapabilityService,
        *,
        consumer_id: str = "rai.stage2-dispatch.v1",
    ) -> None:
        self.journal = journal
        self.capabilities = capabilities
        self.consumer_id = consumer_id

    async def process_next(  # noqa: PLR0911
        self, cancellation: CancellationToken | None = None
    ) -> Result[ActionResult | ActionFailure | None, JournalFailure]:
        token = cancellation or CancellationToken()
        position = await self.journal.position(self.consumer_id)
        if isinstance(position, Failure):
            return Failure(position.failure())
        batch_result = await self.journal.read(position.unwrap(), 1)
        if isinstance(batch_result, Failure):
            return Failure(batch_result.failure())
        batch = batch_result.unwrap()
        if not batch.events:
            return Success(None)
        envelope = batch.events[0]
        record = envelope.record
        terminal: ActionResult | ActionFailure | None = None
        if isinstance(record, Observation) and record.kind == "person_present":
            request = self._request(record)
            existing = await self.journal.terminal_for(request.record_id)
            if isinstance(existing, Failure):
                return Failure(existing.failure())
            terminal = existing.unwrap()
            if terminal is None:
                decision, invocation = await self.capabilities.invoke(request, token)
                if decision is not None:
                    decision_append = await self.journal.append(decision)
                    if isinstance(decision_append, Failure):
                        return Failure(decision_append.failure())
                terminal = (
                    invocation.unwrap()
                    if isinstance(invocation, Success)
                    else invocation.failure()
                )
                terminal_append = await self.journal.append(
                    terminal, data_class=request.data_class
                )
                if isinstance(terminal_append, Failure):
                    return Failure(terminal_append.failure())
        acknowledgement = await self.journal.acknowledge(self.consumer_id, envelope.cursor)
        if isinstance(acknowledgement, Failure):
            return Failure(acknowledgement.failure())
        return Success(terminal)

    @staticmethod
    def _request(observation: Observation) -> CapabilityRequest:
        present = observation.payload.get("present")
        return CapabilityRequest(
            record_id=f"request:{observation.record_id}",
            timestamp=observation.timestamp,
            producer=DISPATCH_PRODUCER,
            correlation_id=observation.correlation_id or observation.record_id,
            actor=DISPATCH_PRODUCER,
            capability="test.echo",
            arguments={"text": f"person_present:{str(bool(present)).lower()}"},
            data_class=DataClass.LOCAL,
            target_resource="local:test-actuator",
            requested_side_effects=(),
            isolation="in-process",
            verification_plan=("compare-output",),
        )
