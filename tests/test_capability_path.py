"""End-to-end acceptance slice for the shared capability and policy path."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from returns.result import Failure, Success

from conftest import ASGITestClient
from rai.cli import cli
from rai.cli_transport import invoke_local
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger, JsonlAuditLedger
from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.defaults import DEFAULT_ACTOR, create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    CapabilityRequest,
    DataClass,
    Observation,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
)
from rai.kernel.service import CapabilityService
from rai.kernel.stores import JsonObservationStore
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request
from rai.routers.mcp import call_tool
from rai.server import create_app

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
EXPECTED_INTERFACE_AUDIT_EVENTS = 6


def _echo_request(registry: CapabilityRegistry) -> CapabilityRequest:
    descriptor = registry.descriptor("test.echo")
    assert descriptor is not None
    return normalize_request(
        descriptor,
        {"text": "shared-path"},
        request_id="request-shared-path",
        correlation_id="correlation-shared-path",
        timestamp=NOW,
    )


@pytest.mark.asyncio
async def test_synthetic_observation_round_trips_through_durable_store(
    tmp_path: Path,
) -> None:
    observation = Observation(
        record_id="observation-durable-1",
        timestamp=NOW,
        producer=ProducerIdentity(
            producer_id="synthetic-collector", kind="collector", version="1.0.0"
        ),
        kind="person_present",
        payload={"present": True},
        data_class=DataClass.LOCAL,
    )
    store = JsonObservationStore(tmp_path)
    appended = await store.append(observation)
    loaded = await store.get(observation.record_id)
    assert isinstance(appended, Success)
    assert isinstance(loaded, Success)
    assert loaded.unwrap() == observation


def test_cli_rest_and_mcp_return_identical_policy_and_envelope() -> None:
    audit = InMemoryAuditLedger()
    container = ApplicationContainer(config={}, audit_ledger=audit, testing=True)
    request = _echo_request(container.capability_registry)
    rest = ASGITestClient(create_app(container)).post(
        "/api/v1/capabilities/invoke", json=request.model_dump(mode="json")
    )
    assert rest.status_code == 200  # noqa: PLR2004

    mcp = asyncio.run(
        call_tool(
            container.capability_service,
            request.capability,
            request.arguments,
            request_record=request,
        )
    )
    assert mcp.structuredContent is not None

    cli_envelope = asyncio.run(invoke_local(request, container))

    envelopes = (
        rest.json(),
        mcp.structuredContent,
        cli_envelope.model_dump(mode="json"),
    )
    assert envelopes[0] == envelopes[1] == envelopes[2]
    assert envelopes[0]["decision"]["outcome"] == "ALLOW"
    assert envelopes[0]["result"]["record_type"] == "action_result"
    assert len(audit.entries) == EXPECTED_INTERFACE_AUDIT_EVENTS
    assert all(entry.decision == audit.entries[0].decision for entry in audit.entries)


def test_click_command_uses_capability_transport() -> None:
    container = ApplicationContainer(config={}, audit_ledger=InMemoryAuditLedger(), testing=True)
    request = _echo_request(container.capability_registry)
    with patch("rai.cli_transport.ApplicationContainer", return_value=container):
        result = CliRunner().invoke(cli, ["capability", "invoke", request.model_dump_json()])
    assert result.exit_code == 0
    assert json.loads(result.output)["result"]["output"] == {"text": "shared-path"}


@pytest.mark.asyncio
async def test_unregistered_and_backend_specific_capabilities_cannot_bypass_policy() -> None:
    registry = create_default_capability_registry()
    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger())
    request = _echo_request(registry).model_copy(
        update={"capability": "antigravity.run_shell_command"}
    )
    decision, result = await service.invoke(request)
    assert decision is None
    assert isinstance(result, Failure)
    assert result.failure().code == "CAPABILITY_NOT_FOUND"


@pytest.mark.asyncio
async def test_unavailable_approval_and_isolation_fail_closed() -> None:
    descriptor = CapabilityDescriptor(
        name="test.moderate",
        description="moderate test",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk_class=RiskClass.MODERATE,
        side_effects=("write",),
        isolation="sandbox",
        requires_isolation=True,
        verification_plan=("verified",),
    )
    registry = CapabilityRegistry()
    registry.register(RegisteredCapability(descriptor, lambda _arguments: {}))
    request = normalize_request(descriptor, {}, timestamp=NOW)

    unavailable_isolation = CapabilityService(
        registry,
        PolicyEngine(isolation_available=lambda _isolation: False),
        InMemoryAuditLedger(),
        SyntheticApprovalBroker(),
    )
    decision, result = await unavailable_isolation.invoke(request)
    assert decision is not None and decision.outcome == PolicyOutcome.DENY
    assert isinstance(result, Failure)

    unavailable_approval = CapabilityService(
        registry,
        PolicyEngine(isolation_available=lambda _isolation: True),
        InMemoryAuditLedger(),
        SyntheticApprovalBroker(available=False),
    )
    decision, result = await unavailable_approval.invoke(request)
    assert decision is not None and decision.outcome == PolicyOutcome.ASK
    assert isinstance(result, Failure)
    assert result.failure().code == "APPROVAL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_policy_and_terminal_result_are_durably_audited(tmp_path: Path) -> None:
    registry = create_default_capability_registry()
    path = tmp_path / "audit.jsonl"
    service = CapabilityService(registry, PolicyEngine(), JsonlAuditLedger(path))
    decision, result = await service.invoke(_echo_request(registry))
    assert decision is not None
    assert isinstance(result, Success)
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [entry["stage"] for entry in entries] == ["DECISION", "TERMINAL"]
    assert entries[1]["decision"]["policy_version"] == "1.0.0"
    assert entries[1]["result"]["record_type"] == "action_result"


@pytest.mark.asyncio
async def test_granted_approval_is_persisted_with_terminal_result() -> None:
    descriptor = CapabilityDescriptor(
        name="test.approved",
        description="approval audit test",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk_class=RiskClass.MODERATE,
        side_effects=("write",),
        isolation="in-process",
        verification_plan=("verified",),
    )
    registry = CapabilityRegistry()
    registry.register(RegisteredCapability(descriptor, lambda _arguments: {}))
    audit = InMemoryAuditLedger()
    service = CapabilityService(
        registry,
        PolicyEngine(),
        audit,
        SyntheticApprovalBroker(),
    )
    request = normalize_request(descriptor, {}, timestamp=NOW)

    decision, _result = await service.invoke(request)

    assert decision is not None and decision.outcome == PolicyOutcome.ASK
    assert audit.entries[-1].stage == "TERMINAL"
    assert audit.entries[-1].approval_id == f"approval-{request.record_id}"


@pytest.mark.asyncio
async def test_cancelled_invocation_has_exactly_one_terminal_result() -> None:
    registry = create_default_capability_registry()
    audit = InMemoryAuditLedger()
    service = CapabilityService(registry, PolicyEngine(), audit)
    cancellation = CancellationToken()
    cancellation.cancel()
    decision, result = await service.invoke(_echo_request(registry), cancellation)
    assert decision is not None
    assert isinstance(result, Failure)
    assert result.failure().code == "CANCELLED"
    terminal = [entry for entry in audit.entries if entry.stage == "TERMINAL"]
    assert len(terminal) == 1
    assert terminal[0].result == result.failure()


def test_policy_has_all_four_machine_readable_outcomes() -> None:
    registry = create_default_capability_registry()
    low = registry.descriptor("test.echo")
    moderate = registry.descriptor("send_desktop_notification")
    assert low is not None and moderate is not None
    policy = PolicyEngine(isolation_available=lambda _isolation: True)
    allow = policy.evaluate(_echo_request(registry), low)
    ask_request = normalize_request(
        moderate,
        {"summary": "test", "body": "test"},
        timestamp=NOW,
    )
    ask = policy.evaluate(ask_request, moderate)
    deny = policy.evaluate(
        _echo_request(registry).model_copy(update={"data_class": DataClass.SECRET}),
        low,
    )
    escalate_request = _echo_request(registry).model_copy(
        update={
            "data_class": DataClass.PRIVATE,
            "target_resource": "https://external.example/task",
        }
    )
    escalate = policy.evaluate(escalate_request, low)
    assert (allow.outcome, ask.outcome, deny.outcome, escalate.outcome) == (
        PolicyOutcome.ALLOW,
        PolicyOutcome.ASK,
        PolicyOutcome.DENY,
        PolicyOutcome.ESCALATE,
    )
