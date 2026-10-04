"""The durable action boundary preserves approval, cancellation and replay semantics."""
from pathlib import Path
import asyncio

from returns.result import Failure, Success

from rai.actions.capabilities import result
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.service import ActionCapabilityService
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import CapabilityRequest, RiskClass
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request


def service(tmp_path: Path, audit: InMemoryAuditLedger) -> ActionCapabilityService:
    registry = CapabilityRegistry()
    registry.register(RegisteredCapability(CapabilityDescriptor(
        name="application.launch", description="Test launch", input_schema={"type": "object", "properties": {}},
        risk_class=RiskClass.MODERATE, side_effects=("application-launch",),
        isolation="host-api", verification_plan=("observed",),
    ), handler=lambda _: {"test": True}))
    instance = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    instance.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    instance.handles = SQLiteHandleStore(tmp_path / "handles.db")
    return instance


async def test_replay_returns_original_decision_without_repeat_audit(tmp_path: Path) -> None:
    audit = InMemoryAuditLedger()
    runtime = service(tmp_path, audit)
    request = normalize_request(runtime.registry.descriptor("application.launch"), {})
    first = await runtime.invoke(request)
    assert isinstance(first[1], Success)
    restarted = service(tmp_path, audit)
    restarted.approvals = SyntheticApprovalBroker(approved=False)
    assert await restarted.invoke(request) == first
    assert len([e for e in audit.entries if e.stage == "TERMINAL"]) == 1


async def test_cancellation_while_waiting_for_approval_has_no_effect(tmp_path: Path) -> None:
    audit = InMemoryAuditLedger()
    runtime = service(tmp_path, audit)
    entered = asyncio.Event()

    class PendingApproval:
        async def request(self, decision, cancellation):  # noqa: ANN001, ANN202
            entered.set()
            await asyncio.Event().wait()

    runtime.approvals = PendingApproval()
    request = normalize_request(runtime.registry.descriptor("application.launch"), {})
    token = CancellationToken()
    running = asyncio.create_task(runtime.invoke(request, token))
    await entered.wait()
    token.cancel()
    _, outcome = await asyncio.wait_for(running, 1)
    assert isinstance(outcome, Failure)
    assert outcome.failure().code == "CANCELLED"
    assert len([e for e in audit.entries if e.stage == "TERMINAL"]) == 1
    assert (await runtime.invoke(request))[1] == outcome


async def test_action_capacity_rejects_excess_and_releases_after_cancellation(tmp_path: Path) -> None:
    from rai.actions.service import MAX_CONCURRENT_ACTIONS
    runtime = service(tmp_path, InMemoryAuditLedger())
    ready = asyncio.Event()
    entered = 0

    class PendingApproval:
        async def request(self, decision, cancellation):
            nonlocal entered
            entered += 1
            if entered == MAX_CONCURRENT_ACTIONS:
                ready.set()
            await asyncio.Event().wait()

    runtime.approvals = PendingApproval()
    tokens = [CancellationToken() for _ in range(MAX_CONCURRENT_ACTIONS)]
    requests = [normalize_request(runtime.registry.descriptor('application.launch'), {}) for _ in tokens]
    tasks = [asyncio.create_task(runtime.invoke(request, token)) for request, token in zip(requests, tokens)]
    await asyncio.wait_for(ready.wait(), 1)
    extra = normalize_request(runtime.registry.descriptor('application.launch'), {})
    _, rejected = await runtime.invoke(extra)
    assert rejected.failure().code == 'ACTION_CAPACITY_EXCEEDED'
    assert entered == MAX_CONCURRENT_ACTIONS
    for token in tokens:
        token.cancel()
    await asyncio.gather(*tasks)
    assert runtime._active_actions == 0
    assert (await runtime.invoke(extra))[1] == rejected
    runtime.approvals = SyntheticApprovalBroker()
    _, allowed = await runtime.invoke(normalize_request(runtime.registry.descriptor('application.launch'), {}))
    assert isinstance(allowed, Success)


async def test_timeout_after_effect_started_is_unknown_and_not_retried(tmp_path, monkeypatch):
    import rai.actions.service as action_service
    monkeypatch.setattr(action_service, 'ACTION_TIMEOUT', 0.01)
    audit = InMemoryAuditLedger()
    runtime = service(tmp_path, audit)
    calls = []

    class SlowEffect:
        name = 'application.launch'

        async def invoke(self, request, cancellation):
            calls.append(request.record_id)
            await asyncio.Event().wait()

    descriptor = runtime.registry.descriptor('application.launch')
    runtime.registry = CapabilityRegistry()
    runtime.registry.register(RegisteredCapability(descriptor, implementation=SlowEffect()))
    request = normalize_request(descriptor, {})
    _, outcome = await runtime.invoke(request)
    assert outcome.failure().code == 'UNKNOWN'
    assert (await runtime.invoke(request))[1] == outcome
    assert calls == [request.record_id]
    terminal = [entry for entry in audit.entries if entry.stage == 'TERMINAL']
    assert len(terminal) == 1
    assert terminal[0].result.code == 'UNKNOWN'


async def test_cancellation_after_effect_started_is_unknown_and_not_retried(tmp_path):
    audit = InMemoryAuditLedger()
    runtime = service(tmp_path, audit)
    entered = asyncio.Event()
    reaped = asyncio.Event()
    calls = []

    class CancellableEffect:
        name = 'application.launch'

        async def invoke(self, request, cancellation):
            calls.append(request.record_id)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                reaped.set()

    descriptor = runtime.registry.descriptor('application.launch')
    runtime.registry = CapabilityRegistry()
    runtime.registry.register(RegisteredCapability(descriptor, implementation=CancellableEffect()))
    token = CancellationToken()
    request = normalize_request(descriptor, {})
    pending = asyncio.create_task(runtime.invoke(request, token))
    await asyncio.wait_for(entered.wait(), 1)
    token.cancel()
    _, outcome = await asyncio.wait_for(pending, 1)
    assert reaped.is_set()
    assert outcome.failure().code == 'UNKNOWN'
    assert (await runtime.invoke(request))[1] == outcome
    assert calls == [request.record_id]
    assert len([entry for entry in audit.entries if entry.stage == 'TERMINAL']) == 1
