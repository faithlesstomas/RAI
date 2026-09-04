"""Conformance checks for Stage 1 protocols and synthetic implementations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from returns.result import Failure, Success

from rai.kernel.ports import (
    Actuator,
    AgentBackend,
    ApprovalBroker,
    CancellationToken,
    Capability,
    Collector,
    DeviceAgent,
    EpisodeStore,
    LifecycleState,
    LocalProcessor,
    ObservationStore,
    UsageLedger,
)
from rai.kernel.records import (
    CapabilityRequest,
    ContextManifest,
    ContextPackage,
    DataClass,
    DeviceDescriptor,
    InferenceBudget,
    Observation,
    ProducerIdentity,
    Task,
)
from rai.kernel.synthetic import (
    SYNTHETIC_PRODUCER,
    SyntheticActuator,
    SyntheticAgentBackend,
    SyntheticApprovalBroker,
    SyntheticCapability,
    SyntheticCollector,
    SyntheticDeviceAgent,
    SyntheticEpisodeStore,
    SyntheticLocalProcessor,
    SyntheticObservationStore,
    SyntheticUsageLedger,
)

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _request() -> CapabilityRequest:
    return CapabilityRequest(
        record_id="request-port-1",
        timestamp=NOW,
        producer=SYNTHETIC_PRODUCER,
        actor=SYNTHETIC_PRODUCER,
        capability="test.echo",
        arguments={"text": "hello"},
        data_class=DataClass.LOCAL,
        target_resource="memory://echo",
        requested_side_effects=(),
        isolation="in-process",
        verification_plan=("compare-output",),
    )


def test_every_synthetic_implementation_conforms_to_its_port() -> None:
    descriptor = DeviceDescriptor(
        timestamp=NOW,
        producer=SYNTHETIC_PRODUCER,
        display_name="synthetic",
        platform="linux",
    )
    implementations = (
        (SyntheticCapability(), Capability),
        (SyntheticCollector(), Collector),
        (SyntheticActuator(), Actuator),
        (SyntheticLocalProcessor(), LocalProcessor),
        (SyntheticDeviceAgent(descriptor), DeviceAgent),
        (SyntheticAgentBackend(), AgentBackend),
        (SyntheticObservationStore(), ObservationStore),
        (SyntheticEpisodeStore(), EpisodeStore),
        (SyntheticUsageLedger(), UsageLedger),
        (SyntheticApprovalBroker(), ApprovalBroker),
    )
    for implementation, port in implementations:
        assert isinstance(implementation, port)


@pytest.mark.asyncio
async def test_collector_lifecycle_and_cancellation_have_one_terminal_failure() -> None:
    observation = Observation(
        timestamp=NOW,
        producer=SYNTHETIC_PRODUCER,
        kind="synthetic",
        payload={},
        data_class=DataClass.LOCAL,
    )
    collector = SyntheticCollector((observation, observation))
    assert collector.state == LifecycleState.CREATED
    assert isinstance(await collector.start(), Success)
    cancellation = CancellationToken()
    cancellation.cancel()
    events = [event async for event in collector.events(cancellation)]
    assert len(events) == 1
    assert isinstance(events[0], Failure)
    assert events[0].failure().code == "CANCELLED"
    assert isinstance(await collector.stop(), Success)
    assert collector.state == LifecycleState.STOPPED


@pytest.mark.asyncio
async def test_processor_and_backend_honor_pre_cancelled_token() -> None:
    task = Task(timestamp=NOW, producer=SYNTHETIC_PRODUCER, objective="test")
    manifest = ContextManifest(
        timestamp=NOW, producer=SYNTHETIC_PRODUCER, destination="synthetic", items=()
    )
    context = ContextPackage(
        timestamp=NOW,
        producer=SYNTHETIC_PRODUCER,
        task_id=task.record_id,
        manifest=manifest,
        content={},
    )
    budget = InferenceBudget(
        timestamp=NOW,
        producer=SYNTHETIC_PRODUCER,
        max_input_tokens=1,
        max_output_tokens=1,
        max_agent_turns=1,
        max_tool_calls=1,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=1,
        max_provider_cost=0,
        max_ram_bytes=1,
        max_vram_bytes=0,
        cancellation_deadline=NOW + timedelta(seconds=1),
    )
    cancellation = CancellationToken()
    cancellation.cancel()
    processor = SyntheticLocalProcessor()
    assert processor.state == LifecycleState.CREATED
    assert isinstance(await processor.start(), Success)
    processor_result = await processor.process(
        task, context, budget, cancellation
    )
    backend_result = await SyntheticAgentBackend().execute(
        task, context, (), budget, cancellation
    )
    assert isinstance(processor_result, Failure)
    assert isinstance(backend_result, Failure)
    assert processor_result.failure().terminal
    assert backend_result.failure().terminal
    assert isinstance(await processor.stop(), Success)
    assert processor.state == LifecycleState.STOPPED
