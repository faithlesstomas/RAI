"""CLI and MCP action invocations share approval, denial and terminal semantics."""
from types import SimpleNamespace

import pytest
from returns.result import Success

from rai.actions.capabilities import result
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.service import ActionCapabilityService
from rai.cli_transport import invoke_local
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.policy import PolicyEngine
from rai.kernel.records import RiskClass
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request
from rai.routers.mcp import call_tool


@pytest.mark.parametrize('approved,available', [(True, True), (False, True), (True, False)])
async def test_cli_mcp_action_contract_parity(tmp_path, approved, available):
    outputs = []
    calls = []
    for transport in ('cli', 'mcp'):
        class Backend:
            name = "application.launch"
            async def invoke(self, request, cancellation):
                calls.append(transport)
                return Success(result(request, {'status': 'SUCCEEDED'}, {'observed': True}))

        registry = CapabilityRegistry()
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name='application.launch', description='Transport contract fixture',
            input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
            risk_class=RiskClass.MODERATE, side_effects=('application-launch',),
            isolation='host-api', verification_plan=('observed',),
        ), implementation=Backend()))
        audit = InMemoryAuditLedger()
        runtime = ActionCapabilityService(registry, PolicyEngine(), audit,
                                          SyntheticApprovalBroker(approved=approved, available=available))
        runtime.handles = SQLiteHandleStore(tmp_path / transport / 'handles.db')
        runtime.executions = SQLiteExecutionStore(tmp_path / transport / 'executions.db')
        request = normalize_request(registry.descriptor('application.launch'), {}, request_id='same-request')
        if transport == 'cli':
            envelope = await invoke_local(request, SimpleNamespace(capability_service=runtime))
            payload = envelope.model_dump(mode='json')
        else:
            response = await call_tool(runtime, request.capability, {}, request_record=request)
            payload = response.structuredContent
            assert response.isError == (not payload['ok'])
        assert payload['decision']['outcome'] == 'ASK'
        assert [entry.stage for entry in audit.entries] == ['DECISION', 'TERMINAL']
        terminal = payload['result']
        outputs.append((payload['ok'], terminal.get('code'), terminal.get('output'), terminal.get('verification')))
    assert outputs[0] == outputs[1]
    assert calls == (['cli', 'mcp'] if approved and available else [])
