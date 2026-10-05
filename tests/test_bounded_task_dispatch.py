"""Unit and integration tests for bounded local inference task dispatch and REST transport."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, List, Optional
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Result, Success

from conftest import ASGITestClient
from rai.container import ApplicationContainer
from rai.inference import (
    BoundedTaskKind,
    InferenceResult,
    LocalTextEngine,
    ProcessorSupervisor,
)
from rai.inference.capabilities import (
    BOUNDED_TASK_CAPABILITY,
    BoundedTaskCapability,
    bounded_task_descriptor,
)
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    RiskClass,
)
from rai.kernel.service import CapabilityService
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.routers.inference import BoundedTaskRequest
from rai.server import create_app

TEST_PRODUCER = ProducerIdentity(
    producer_id="test.bounded-dispatch", kind="test", version="1.0.0"
)


class ScriptedEngine(LocalTextEngine):
    """Deterministic async engine returning scripted responses."""

    model_name = "test-scripted-model"

    def __init__(
        self,
        responses: list[str] | str,
        *,
        delay: float = 0.0,
    ) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.delay = delay
        self._is_loaded = False
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []
        self.temperatures: list[float] = []

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    async def load(self) -> Result[None, Exception]:
        self._is_loaded = True
        return Success(None)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        del stop
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        self.temperatures.append(temperature)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        response = self.responses.pop(0) if self.responses else "{}"
        return Success(InferenceResult(text=response))

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[Result[str, Exception]]:
        del prompt, stop, max_tokens, temperature
        if False:
            yield Success("")

    async def unload(self) -> Result[None, Exception]:
        self._is_loaded = False
        return Success(None)


def _make_request(
    kind: str,
    objective: str,
    content: dict[str, Any] | None = None,
    data_class: DataClass = DataClass.LOCAL,
    **kwargs: Any,
) -> CapabilityRequest:
    args: dict[str, Any] = {
        "kind": kind,
        "objective": objective,
        "content": content or {},
        **kwargs,
    }
    return CapabilityRequest(
        producer=TEST_PRODUCER,
        actor=TEST_PRODUCER,
        capability=BOUNDED_TASK_CAPABILITY,
        arguments=args,
        data_class=data_class,
        target_resource=f"capability://{BOUNDED_TASK_CAPABILITY}",
        requested_side_effects=(),
        isolation="in-process",
        verification_plan=("claim-schema-validation",),
    )


@pytest.mark.asyncio
async def test_bounded_task_capability_in_container() -> None:
    container = ApplicationContainer(config={}, testing=True)
    descriptor = container.capability_registry.descriptor(BOUNDED_TASK_CAPABILITY)
    assert descriptor is not None
    assert descriptor.name == BOUNDED_TASK_CAPABILITY
    assert descriptor.risk_class == RiskClass.LOW
    assert "episode_summarization" in descriptor.input_schema["properties"]["kind"]["enum"]
    await container.close()


@pytest.mark.asyncio
async def test_all_six_bounded_tasks_dispatch_and_produce_claims() -> None:
    cases = [
        (
            BoundedTaskKind.EPISODE_SUMMARIZATION,
            '{"summary":"User edited Python files.", "key_points":["Modified code"], "confidence":0.9}',
            "User edited Python files.",
            "inferred:episode_summarization@1.0.0",
        ),
        (
            BoundedTaskKind.INTENT_CLASSIFICATION,
            '{"intent":"command", "rationale":"User commanded build", "confidence":0.85}',
            "Intent: command. Rationale: User commanded build",
            "inferred:intent_classification@1.0.0",
        ),
        (
            BoundedTaskKind.ENTITY_EXTRACTION,
            '{"entities":[{"text":"RAI", "entity_type":"project", "source_id":"src1"}], "confidence":0.88}',
            None,  # will check substring
            "inferred:entity_extraction@1.0.0",
        ),
        (
            BoundedTaskKind.SALIENCE_ESTIMATION,
            '{"score":0.75, "level":"high", "rationale":"Significant change", "confidence":0.82}',
            "Salience: high (0.750). Rationale: Significant change",
            "inferred:salience_estimation@1.0.0",
        ),
        (
            BoundedTaskKind.PRIVACY_RISK_ELEVATION,
            '{"data_class":"PRIVATE", "risk_factors":["email"], "rationale":"Contains private email", "confidence":0.95}',
            "Privacy risk: PRIVATE. Risk factors: email. Rationale: Contains private email",
            "inferred:privacy_risk_elevation@1.0.0",
        ),
        (
            BoundedTaskKind.ROUTING_HINT,
            '{"decision":"LOCAL", "rationale":"Task is safe for local handling", "confidence":0.9}',
            "Routing hint: LOCAL. Rationale: Task is safe for local handling",
            "inferred:routing_hint@1.0.0",
        ),
    ]

    for kind, raw_json, expected_statement, expected_status in cases:
        engine = ScriptedEngine(raw_json)
        supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
        capability = BoundedTaskCapability(supervisor)

        content = {"src1": {"data": "test payload"}} if kind == BoundedTaskKind.ENTITY_EXTRACTION else {}
        req = _make_request(kind.value, f"Execute {kind.value}", content=content)
        result = await capability.invoke(req, CancellationToken())

        assert isinstance(result, Success), f"Failed for {kind}: {result}"
        action_res = result.unwrap()
        assert isinstance(action_res, ActionResult)
        assert action_res.capability == BOUNDED_TASK_CAPABILITY
        if expected_statement:
            assert action_res.output["statement"] == expected_statement
        else:
            assert "RAI" in action_res.output["statement"]
        assert action_res.output["epistemic_status"] == expected_status
        assert action_res.output["confidence"] >= 0.8
        assert "claim_id" in action_res.verification
        await supervisor.stop()


@pytest.mark.asyncio
async def test_bounded_task_governed_service_invocation() -> None:
    engine = ScriptedEngine(
        '{"summary":"Session completed cleanly.", "key_points":[], "confidence":0.92}'
    )
    supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
    audit = InMemoryAuditLedger()
    container = ApplicationContainer(config={}, audit_ledger=audit, testing=True)
    container._processor_supervisor = supervisor

    req = _make_request(
        BoundedTaskKind.EPISODE_SUMMARIZATION.value,
        "Summarize recent work",
    )
    decision, res = await container.capability_service.invoke(req)

    assert decision.outcome == "ALLOW"
    assert isinstance(res, Success)
    action_res = res.unwrap()
    assert action_res.output["statement"] == "Session completed cleanly."

    # Audit records all stages
    stages = [entry.stage for entry in audit.entries]
    assert "DECISION" in stages
    assert "TERMINAL" in stages

    await container.close()


@pytest.mark.asyncio
async def test_cancellation_propagation() -> None:
    engine = ScriptedEngine(
        '{"summary":"Slow", "key_points":[], "confidence":0.9}', delay=0.5
    )
    supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
    capability = BoundedTaskCapability(supervisor)

    # 1. Pre-cancelled token
    token = CancellationToken()
    token.cancel()
    req = _make_request(BoundedTaskKind.EPISODE_SUMMARIZATION.value, "Summarize")
    res = await capability.invoke(req, token)
    assert isinstance(res, Failure)
    assert res.failure().code == "CANCELLED"

    # 2. Cancel during execution
    active_token = CancellationToken()
    task = asyncio.create_task(capability.invoke(req, active_token))
    await asyncio.sleep(0.05)
    active_token.cancel()
    res2 = await task
    assert isinstance(res2, Failure)
    assert res2.failure().code == "CANCELLED"

    await supervisor.stop()


@pytest.mark.asyncio
async def test_validation_and_bad_arguments() -> None:
    engine = ScriptedEngine('{"invalid_json": true}')
    supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
    capability = BoundedTaskCapability(supervisor)

    # Missing kind
    req1 = _make_request("", "Summarize")
    res1 = await capability.invoke(req1, CancellationToken())
    assert isinstance(res1, Failure)
    assert res1.failure().code == "INVALID_ARGUMENT"

    # Unsupported kind
    req2 = _make_request("non_existent_kind", "Summarize")
    res2 = await capability.invoke(req2, CancellationToken())
    assert isinstance(res2, Failure)
    assert res2.failure().code == "INVALID_TASK_KIND"

    # Invalid model output JSON (does not conform to EpisodeSummaryOutput)
    req3 = _make_request(BoundedTaskKind.EPISODE_SUMMARIZATION.value, "Summarize")
    res3 = await capability.invoke(req3, CancellationToken())
    assert isinstance(res3, Failure)
    assert res3.failure().code == "INVALID_MODEL_OUTPUT"

    await supervisor.stop()


@pytest.mark.asyncio
async def test_rest_api_contracts_and_task_dispatch() -> None:
    import httpx

    engine = ScriptedEngine(
        '{"decision":"LOCAL", "rationale":"Fully bounded operation", "confidence":0.93}'
    )
    supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
    container = ApplicationContainer(config={}, testing=True)
    container._processor_supervisor = supervisor

    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. GET /api/v1/inference/contracts
        contracts_resp = await client.get("/api/v1/inference/contracts")
        assert contracts_resp.status_code == 200
        contracts_data = contracts_resp.json()
        assert "contracts" in contracts_data
        kinds = [c["kind"] for c in contracts_data["contracts"]]
        assert len(kinds) == 6
        assert "routing_hint" in kinds
        assert "episode_summarization" in kinds

        # 2. POST /api/v1/inference/tasks
        task_payload = {
            "kind": "routing_hint",
            "objective": "Determine routing for task",
            "content": {"query": "local search"},
            "data_class": "LOCAL",
        }
        dispatch_resp = await client.post("/api/v1/inference/tasks", json=task_payload)
        assert dispatch_resp.status_code == 200
        envelope = dispatch_resp.json()
        assert envelope["ok"] is True
        assert envelope["result"]["record_type"] == "action_result"
        assert envelope["result"]["output"]["statement"] == "Routing hint: LOCAL. Rationale: Fully bounded operation"
        assert envelope["result"]["output"]["confidence"] == 0.93

        # 3. Bad request on unknown task kind
        bad_resp = await client.post("/api/v1/inference/tasks", json={"kind": "unknown_kind", "objective": "test"})
        assert bad_resp.status_code == 422

    await container.close()


@pytest.mark.asyncio
async def test_event_loop_responsiveness_during_inference() -> None:
    """Ensure that slow inference tasks do not block other concurrent async operations."""
    engine = ScriptedEngine(
        '{"summary":"Done.", "key_points":[], "confidence":0.9}', delay=0.2
    )
    supervisor = ProcessorSupervisor(engine=engine, idle_unload_seconds=0)
    capability = BoundedTaskCapability(supervisor)
    req = _make_request(BoundedTaskKind.EPISODE_SUMMARIZATION.value, "Summarize")

    ticks = 0

    async def tick_loop() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.03)
            ticks += 1

    inference_task = asyncio.create_task(capability.invoke(req, CancellationToken()))
    ticker_task = asyncio.create_task(tick_loop())

    await asyncio.gather(inference_task, ticker_task)
    assert ticks >= 4, f"Event loop was blocked: only ticked {ticks} times"
    await supervisor.stop()
