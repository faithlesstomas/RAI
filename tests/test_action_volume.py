"""Volume writes require the same sink and observed postcondition."""
from returns.result import Failure, Success
import pytest

from rai.actions.handles import SQLiteHandleStore
from rai.actions.volume import register_volume_capabilities
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request


class AudioBackend:
    def __init__(self):
        self.identity = 'sink-one'
        self.volume = 30
        self.calls = []
        self.ignore_write = False

    async def read(self, token):
        return Success({'sink': 'default', 'fingerprint': self.identity,
                        'volumes': [self.volume, self.volume], 'muted': False})

    async def set(self, sink, percent, token):
        self.calls.append((sink, percent))
        if not self.ignore_write:
            self.volume = percent
        return Success(None)


@pytest.mark.parametrize('scenario', ['success', 'changed_sink', 'ignored_write', 'invalid_percent'])
async def test_volume_requires_identity_and_readback(tmp_path, scenario):
    backend = AudioBackend()
    registry = CapabilityRegistry()
    register_volume_capabilities(registry, SQLiteHandleStore(tmp_path / 'handles.db'), backend)
    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    request = normalize_request(registry.descriptor('system.volume.get'), {'task_id': 'audio-task'})
    _, read = await service.invoke(request)
    handle = read.unwrap().output['handle']
    if scenario == 'changed_sink':
        backend.identity = 'replacement'
    backend.ignore_write = scenario == 'ignored_write'
    request = normalize_request(registry.descriptor('system.volume.set'),
                                {'task_id': 'audio-task', 'handle': handle,
                                 'percent': 101 if scenario == 'invalid_percent' else 40})
    _, changed = await service.invoke(request)
    if scenario == 'success':
        assert changed.unwrap().verification['observed_percent'] == (40, 40)
        assert backend.calls == [('default', 40)]
    else:
        assert isinstance(changed, Failure)
        assert changed.failure().code == {'changed_sink': 'STALE_RESOURCE',
                                         'ignored_write': 'POSTCONDITION_FAILED',
                                         'invalid_percent': 'INVALID_ARGUMENT'}[scenario]
        assert len(backend.calls) == (1 if scenario == 'ignored_write' else 0)
