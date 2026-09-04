"""Deterministic reference implementations for every Stage 1 runtime port."""

from __future__ import annotations

from collections.abc import AsyncIterator

from returns.result import Failure, Result, Success

from .ports import CancellationToken, LifecycleState
from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    Claim,
    ContextPackage,
    DataClass,
    DeviceDescriptor,
    Episode,
    InferenceBudget,
    Observation,
    PolicyDecision,
    ProducerIdentity,
    ProvenanceReference,
    Task,
    UsageRecord,
)

SYNTHETIC_PRODUCER = ProducerIdentity(
    producer_id="rai.synthetic", kind="reference", version="1.0.0"
)


def _failure(
    request_id: str, capability: str, code: str, message: str
) -> ActionFailure:
    return ActionFailure(
        record_id=f"failure:{request_id}:{code}",
        producer=SYNTHETIC_PRODUCER,
        correlation_id=request_id,
        request_id=request_id,
        capability=capability,
        code=code,
        message=message,
    )


def _cancelled(request_id: str, capability: str) -> ActionFailure:
    return _failure(request_id, capability, "CANCELLED", "operation was cancelled")


class SyntheticCapability:
    """Echo capability used by conformance and transport tests."""

    name = "test.echo"

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_cancelled(request.record_id, self.name))
        text = request.arguments.get("text")
        if not isinstance(text, str):
            return Failure(
                _failure(request.record_id, self.name, "INVALID_ARGUMENT", "text must be a string")
            )
        return Success(
            ActionResult(
                producer=SYNTHETIC_PRODUCER,
                correlation_id=request.correlation_id,
                request_id=request.record_id,
                capability=self.name,
                output={"text": text},
                verification={"matched": True},
            )
        )


class SyntheticCollector:
    def __init__(self, observations: tuple[Observation, ...] = ()) -> None:
        self._state = LifecycleState.CREATED
        self._observations = observations

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    async def events(
        self, cancellation: CancellationToken
    ) -> AsyncIterator[Result[Observation, ActionFailure]]:
        for observation in self._observations:
            if cancellation.cancelled:
                yield Failure(_cancelled(observation.record_id, "collector.events"))
                return
            yield Success(observation)


class SyntheticActuator:
    async def act(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        return await SyntheticCapability().invoke(request, cancellation)


class SyntheticLocalProcessor:
    def __init__(self) -> None:
        self._state = LifecycleState.CREATED

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def process(
        self,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]:
        del budget
        if cancellation.cancelled:
            return Failure(_cancelled(task.record_id, "processor.synthetic"))
        self._state = LifecycleState.RUNNING
        source = ProvenanceReference(
            source_id=context.record_id,
            source_type=context.record_type,
            source_version=context.schema_version,
            relation="derived-from",
            producer=context.producer,
        )
        return Success(
            Claim(
                producer=SYNTHETIC_PRODUCER,
                correlation_id=task.correlation_id,
                statement=task.objective,
                confidence=1,
                epistemic_status="synthetic",
                data_class=DataClass.LOCAL,
                provenance=(source,),
            )
        )

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)


class SyntheticDeviceAgent:
    def __init__(self, descriptor: DeviceDescriptor) -> None:
        self.descriptor = descriptor
        self.observations: list[Observation] = []

    async def describe(self) -> Result[DeviceDescriptor, ActionFailure]:
        return Success(self.descriptor)

    async def submit(
        self, observation: Observation
    ) -> Result[Observation, ActionFailure]:
        self.observations.append(observation)
        return Success(observation)


class SyntheticAgentBackend:
    def __init__(self) -> None:
        self._state = LifecycleState.CREATED

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def execute(
        self,
        task: Task,
        context: ContextPackage,
        capabilities: tuple[str, ...],
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]:
        del context, capabilities, budget
        if cancellation.cancelled:
            return Failure(_cancelled(task.record_id, "backend.synthetic"))
        return Success(
            ActionResult(
                producer=SYNTHETIC_PRODUCER,
                correlation_id=task.correlation_id,
                request_id=task.record_id,
                capability="backend.synthetic",
                output={"objective": task.objective},
                verification={"synthetic": True},
            )
        )

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)


class SyntheticObservationStore:
    def __init__(self) -> None:
        self.records: dict[str, Observation] = {}

    async def append(
        self, observation: Observation
    ) -> Result[Observation, ActionFailure]:
        self.records[observation.record_id] = observation
        return Success(observation)

    async def get(self, record_id: str) -> Result[Observation, ActionFailure]:
        record = self.records.get(record_id)
        return (
            Success(record)
            if record is not None
            else Failure(_failure(record_id, "observation.get", "NOT_FOUND", "not found"))
        )


class SyntheticEpisodeStore:
    def __init__(self) -> None:
        self.records: dict[str, Episode] = {}

    async def append(self, episode: Episode) -> Result[Episode, ActionFailure]:
        self.records[episode.record_id] = episode
        return Success(episode)

    async def get(self, record_id: str) -> Result[Episode, ActionFailure]:
        record = self.records.get(record_id)
        return (
            Success(record)
            if record is not None
            else Failure(_failure(record_id, "episode.get", "NOT_FOUND", "not found"))
        )


class SyntheticUsageLedger:
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def append(self, usage: UsageRecord) -> Result[UsageRecord, ActionFailure]:
        self.records.append(usage)
        return Success(usage)

    async def list_for_task(
        self, task_id: str
    ) -> Result[tuple[UsageRecord, ...], ActionFailure]:
        return Success(tuple(record for record in self.records if record.task_id == task_id))


class SyntheticApprovalBroker:
    def __init__(self, approved: bool = True, available: bool = True) -> None:
        self.approved = approved
        self.available = available

    async def request(
        self, decision: PolicyDecision, cancellation: CancellationToken
    ) -> Result[str, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_cancelled(decision.request_id, "approval.request"))
        if not self.available:
            return Failure(
                _failure(
                    decision.request_id,
                    "approval.request",
                    "APPROVAL_UNAVAILABLE",
                    "approval broker is unavailable",
                )
            )
        if not self.approved:
            return Failure(
                _failure(decision.request_id, "approval.request", "DENIED", "approval denied")
            )
        return Success(f"approval-{decision.request_id}")
