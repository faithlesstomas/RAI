"""Process handles cannot silently follow a recycled PID."""
from pathlib import Path

from rai.actions.processes import process_metadata, register_process_capability
from rai.actions.handles import SQLiteHandleStore
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.transport import normalize_request


def write_process(root: Path, start: str):
    process = root / '123'
    process.mkdir(exist_ok=True)
    executable = root / 'program'
    executable.touch()
    if not (process / 'exe').exists():
        (process / 'exe').symlink_to(executable)
    (process / 'stat').write_text('123 (name with ) brackets) ' + ' '.join(['S'] + ['0'] * 18 + [start]))


def test_process_fingerprint_includes_start_time(tmp_path):
    write_process(tmp_path, '100')
    first = process_metadata('123', tmp_path)
    write_process(tmp_path, '200')
    second = process_metadata('123', tmp_path)
    assert first['fingerprint'] != second['fingerprint']
    assert first['name'] == 'program'
    assert process_metadata('../123', tmp_path) is None
    assert process_metadata('missing', tmp_path) is None


async def test_recycled_pid_rejected_after_search(tmp_path, monkeypatch):
    import rai.actions.processes as processes
    write_process(tmp_path, '100')
    first = process_metadata('123', tmp_path)
    monkeypatch.setattr(processes, 'find_processes', lambda query: (first,))
    monkeypatch.setattr(processes, 'process_metadata', lambda pid: process_metadata(pid, tmp_path))
    registry = CapabilityRegistry()
    register_process_capability(registry, SQLiteHandleStore(tmp_path / 'handles.db'))
    descriptor = registry.descriptor('process.inspect')
    implementation = processes.ProcessInspect(SQLiteHandleStore(tmp_path / 'handles.db'))
    request = normalize_request(descriptor, {'task_id': 'inspect', 'query': 'program'})
    found = await implementation.invoke(request, CancellationToken())
    handle = found.unwrap().output['processes'][0]['handle']
    request = normalize_request(descriptor, {'task_id': 'inspect', 'handle': handle})
    observed = await implementation.invoke(request, CancellationToken())
    assert observed.unwrap().output['start_time'] == '100'
    write_process(tmp_path, '200')
    stale = await implementation.invoke(request, CancellationToken())
    assert stale.failure().code == 'STALE_RESOURCE'
