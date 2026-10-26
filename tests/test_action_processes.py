"""Process handles cannot silently follow a recycled PID."""
from pathlib import Path
from pytest import MonkeyPatch

from rai.actions import processes
from rai.actions.handles import SQLiteHandleStore
from rai.actions.processes import process_metadata, register_process_capability
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.transport import normalize_request


def write_process(root: Path, start: str) -> None:
    process = root / '123'
    process.mkdir(exist_ok=True)
    executable = root / 'program'
    executable.touch()
    if not (process / 'exe').exists():
        (process / 'exe').symlink_to(executable)
    (process / 'stat').write_text('123 (name with ) brackets) ' + ' '.join(['S'] + ['0'] * 18 + [start]))


def test_process_fingerprint_includes_start_time(tmp_path: Path) -> None:
    write_process(tmp_path, '100')
    first = process_metadata('123', tmp_path)
    write_process(tmp_path, '200')
    second = process_metadata('123', tmp_path)
    assert first is not None and second is not None
    assert first['fingerprint'] != second['fingerprint']
    assert first['name'] == 'program'
    assert process_metadata('../123', tmp_path) is None
    assert process_metadata('missing', tmp_path) is None


async def test_recycled_pid_rejected_after_search(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
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


def test_find_processes_detailed_limits(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake_procs = [tmp_path / str(i) for i in range(10)]
    for p in fake_procs:
        p.mkdir()

    monkeypatch.setattr(processes.Path, "iterdir", lambda self: fake_procs)
    monkeypatch.setattr(
        processes,
        "process_metadata",
        lambda pid: {"pid": int(pid), "name": f"proc_{pid}", "fingerprint": f"fp_{pid}"},
    )
    monkeypatch.setattr(processes, "MAX_PROCESSES", 5)

    # 1. safety guard on found matches
    limit = 2
    procs, complete, reason = processes.find_processes_detailed("proc", limit=limit)
    assert len(procs) == limit
    assert complete is False
    assert "capped at safety guard" in (reason or "")

    # 2. scan limit on visited MAX_PROCESSES
    procs2, complete2, reason2 = processes.find_processes_detailed("proc_9")
    assert len(procs2) == 0
    assert complete2 is False
    assert "scan limit" in (reason2 or "")


