"""Process inspection bounds, ownership and handle validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from returns.result import Failure, Success

import rai.actions.processes as processes
from rai.actions.capabilities import PRODUCER
from rai.actions.handles import SQLiteHandleStore
from rai.actions.processes import ProcessInspect, find_processes, process_metadata, register_process_capability
from rai.actions.records import ResourceHandle
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, _utc_now
from rai.kernel.transport import normalize_request

from datetime import timedelta


def write_process(root: Path, pid: str = "123", stat: str | None = None) -> None:
    process = root / pid
    process.mkdir(exist_ok=True)
    executable = root / "program"
    executable.touch()
    if not (process / "exe").exists():
        (process / "exe").symlink_to(executable)
    (process / "stat").write_text(stat or f"{pid} (program) " + " ".join(["S"] + ["0"] * 18 + ["100"]))


def test_metadata_ignores_processes_of_other_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_process(tmp_path)
    assert process_metadata("123", tmp_path) is not None
    monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path / "123").st_uid + 1)
    assert process_metadata("123", tmp_path) is None


def test_metadata_rejects_truncated_stat_and_start_time_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_process(tmp_path, stat="123 (program) S 0")
    assert process_metadata("123", tmp_path) is None
    write_process(tmp_path, "124")
    reads = iter(["124 (program) " + " ".join(["S"] + ["0"] * 18 + [start]) for start in ("100", "200")])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: next(reads))
    assert process_metadata("124", tmp_path) is None


def test_find_processes_matches_name_or_pid_and_respects_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    name = Path(os.readlink("/proc/self/exe")).name
    assert any(entry["pid"] == os.getpid() for entry in find_processes(name))
    assert [entry["pid"] for entry in find_processes(str(os.getpid()))] == [os.getpid()]
    assert find_processes("definitely-not-a-process-name") == ()
    monkeypatch.setattr(processes, "MAX_PROCESSES", 0)
    assert find_processes(name) == ()
    monkeypatch.setattr(processes, "MAX_PROCESSES", 4096)
    monkeypatch.setattr(processes, "MAX_RESULTS", 1)
    assert len(find_processes("")) <= 1


def setup(tmp_path: Path) -> tuple[ProcessInspect, Any, SQLiteHandleStore]:
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_process_capability(registry, handles)
    return ProcessInspect(handles), registry.descriptor("process.inspect"), handles


async def test_invalid_queries_are_rejected_without_scanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    implementation, descriptor, _ = setup(tmp_path)
    monkeypatch.setattr(processes, "find_processes", lambda _query: pytest.fail("scan must not run"))
    for query in ("  ", "q" * 257):
        request = normalize_request(descriptor, {"task_id": "t", "query": query})
        assert (await implementation.invoke(request, CancellationToken())).failure().code == "INVALID_ARGUMENT"
    request = normalize_request(descriptor, {"task_id": "t"})
    assert (await implementation.invoke(request, CancellationToken())).failure().code == "INVALID_ARGUMENT"


async def test_cancellation_during_result_issue_stops_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    implementation, descriptor, _ = setup(tmp_path)
    monkeypatch.setattr(processes, "find_processes", lambda _query: ({"pid": 1, "name": "a", "fingerprint": "f"},))
    token = CancellationToken()
    token.cancel()
    request = normalize_request(descriptor, {"task_id": "t", "query": "a"})
    assert (await implementation.invoke(request, token)).failure().code == "CANCELLED"


async def test_inspect_by_handle_checks_scope_kind_class_and_liveness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    implementation, descriptor, handles = setup(tmp_path)
    own = {"pid": os.getpid(), "name": "x", "fingerprint": process_metadata(str(os.getpid()))["fingerprint"]}
    monkeypatch.setattr(processes, "find_processes", lambda _query: (own,))
    found = await implementation.invoke(normalize_request(descriptor, {"task_id": "t", "query": "x"}), CancellationToken())
    handle = found.unwrap().output["processes"][0]["handle"]
    request = normalize_request(descriptor, {"task_id": "t", "handle": handle})
    assert (await implementation.invoke(request, CancellationToken())).unwrap().output["pid"] == os.getpid()
    wrong_task = normalize_request(descriptor, {"task_id": "other", "handle": handle})
    assert isinstance(await implementation.invoke(wrong_task, CancellationToken()), Failure)
    public = request.model_copy(update={"data_class": DataClass.PUBLIC})
    if request.data_class != DataClass.PUBLIC:
        assert (await implementation.invoke(public, CancellationToken())).failure().code == "STALE_RESOURCE"
    now = _utc_now()
    foreign_kind = handles.issue(ResourceHandle(
        producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id, task_id="t", kind="file",
        target=str(os.getpid()), fingerprint=own["fingerprint"], operations=("process.inspect",),
        expires_at=now + timedelta(minutes=5), data_class=request.data_class,
    )).unwrap()
    wrong_kind = normalize_request(descriptor, {"task_id": "t", "handle": foreign_kind})
    assert (await implementation.invoke(wrong_kind, CancellationToken())).failure().code == "STALE_RESOURCE"
    exited = handles.issue(ResourceHandle(
        producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id, task_id="t", kind="process",
        target="999999999", fingerprint="gone", operations=("process.inspect",),
        expires_at=now + timedelta(minutes=5), data_class=request.data_class,
    )).unwrap()
    gone = normalize_request(descriptor, {"task_id": "t", "handle": exited})
    assert (await implementation.invoke(gone, CancellationToken())).failure().code == "STALE_RESOURCE"
