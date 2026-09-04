"""Provider-neutral runtime ports and lifecycle contracts."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import AsyncIterator, Protocol, runtime_checkable

from returns.result import Result

from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    Claim,
    ContextPackage,
    DeviceDescriptor,
    Episode,
    InferenceBudget,
    Observation,
    PolicyDecision,
    Task,
    UsageRecord,
)


class LifecycleState(str, Enum):
    """Common lifecycle for active kernel components."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class CancellationToken:
    """Explicit cancellation signal shared across transport boundaries."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


@runtime_checkable
class Capability(Protocol):
    """One validated operation exposed through every RAI interface."""

    @property
    def name(self) -> str: ...

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]: ...


@runtime_checkable
class Collector(Protocol):
    @property
    def state(self) -> LifecycleState: ...

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...

    def events(
        self, cancellation: CancellationToken
    ) -> AsyncIterator[Result[Observation, ActionFailure]]: ...


@runtime_checkable
class Actuator(Protocol):
    async def act(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]: ...


@runtime_checkable
class LocalProcessor(Protocol):
    @property
    def state(self) -> LifecycleState: ...

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def process(
        self,
        task: Task,
        context: ContextPackage,
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...


@runtime_checkable
class DeviceAgent(Protocol):
    async def describe(self) -> Result[DeviceDescriptor, ActionFailure]: ...

    async def submit(
        self, observation: Observation
    ) -> Result[Observation, ActionFailure]: ...


@runtime_checkable
class AgentBackend(Protocol):
    @property
    def state(self) -> LifecycleState: ...

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def execute(
        self,
        task: Task,
        context: ContextPackage,
        capabilities: tuple[str, ...],
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...


@runtime_checkable
class ObservationStore(Protocol):
    async def append(
        self, observation: Observation
    ) -> Result[Observation, ActionFailure]: ...

    async def get(self, record_id: str) -> Result[Observation, ActionFailure]: ...


@runtime_checkable
class EpisodeStore(Protocol):
    async def append(self, episode: Episode) -> Result[Episode, ActionFailure]: ...

    async def get(self, record_id: str) -> Result[Episode, ActionFailure]: ...


@runtime_checkable
class UsageLedger(Protocol):
    async def append(self, usage: UsageRecord) -> Result[UsageRecord, ActionFailure]: ...

    async def list_for_task(
        self, task_id: str
    ) -> Result[tuple[UsageRecord, ...], ActionFailure]: ...


@runtime_checkable
class ApprovalBroker(Protocol):
    async def request(
        self, decision: PolicyDecision, cancellation: CancellationToken
    ) -> Result[str, ActionFailure]: ...
