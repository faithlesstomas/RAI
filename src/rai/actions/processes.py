"""Current-user process metadata with PID-reuse-resistant resource identity."""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, RiskClass, _utc_now
from .capabilities import HANDLE_TTL, MAX_RESULTS, PRODUCER, failure, result
from .handles import SQLiteHandleStore
from .records import ResourceHandle

MAX_PROCESSES = 4096


def process_metadata(pid: str, proc: Path = Path("/proc")) -> dict | None:
    """Only identity/state are read; never process arguments, environment or memory."""
    if not pid.isdecimal():
        return None
    path = proc / pid
    try:
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        executable = str((path / "exe").resolve(strict=True))
        confirmed = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if confirmed[19] != fields[19] or path.stat().st_uid != os.getuid():
            return None
        identity = f"{pid}:{fields[19]}:{executable}"
        return {"pid": int(pid), "name": Path(executable).name, "state": fields[0],
                "start_time": fields[19], "fingerprint": hashlib.sha256(identity.encode()).hexdigest()}
    except (OSError, IndexError):
        return None


def find_processes(query: str) -> tuple[dict, ...]:
    found = []
    for count, path in enumerate(Path("/proc").iterdir()):
        if count >= MAX_PROCESSES or len(found) >= MAX_RESULTS:
            break
        entry = process_metadata(path.name)
        if entry and (query.casefold() in entry["name"].casefold() or query == str(entry["pid"])):
            found.append(entry)
    return tuple(found)


class ProcessInspect:
    name = "process.inspect"

    def __init__(self, handles: SQLiteHandleStore) -> None:
        self.handles = handles

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:  # noqa: PLR0911
        if "handle" in request.arguments:
            resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                            task_id=request.arguments["task_id"], operation=self.name)
            if isinstance(resolved, Failure):
                return Failure(failure(request, resolved.failure()))
            handle = resolved.unwrap()
            observed = process_metadata(handle.target)
            if (handle.kind != "process" or handle.data_class != request.data_class
                    or observed is None or observed["fingerprint"] != handle.fingerprint):
                return Failure(failure(request, "STALE_RESOURCE"))
            return Success(result(request, observed, {"current_user_and_start_time": True}))
        query = request.arguments.get("query", "")
        if not query.strip() or len(query) > 256:  # noqa: PLR2004
            return Failure(failure(request, "INVALID_ARGUMENT"))
        entries = []
        for process in await asyncio.to_thread(find_processes, query):
            if cancellation.cancelled:
                return Failure(failure(request, "CANCELLED"))
            now = _utc_now()
            handle = ResourceHandle(producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id,
                                    task_id=request.arguments["task_id"], kind="process", target=str(process["pid"]),
                                    fingerprint=process["fingerprint"], operations=(self.name,),
                                    expires_at=now + HANDLE_TTL, data_class=request.data_class)
            issued = self.handles.issue(handle)
            if isinstance(issued, Failure):
                return Failure(failure(request, issued.failure()))
            entries.append({**process, "handle": issued.unwrap()})
        return Success(result(request, {"processes": entries}, {"current_user_only": True}))


def register_process_capability(registry: CapabilityRegistry, handles: SQLiteHandleStore) -> None:
    registry.register(RegisteredCapability(CapabilityDescriptor(
        name="process.inspect", description="Inspect bounded current-user process metadata (v1).",
        input_schema={"type": "object", "properties": {key: {"type": "string"} for key in ("task_id", "query", "handle")},
                      "required": ["task_id"], "additionalProperties": False},
        risk_class=RiskClass.LOW, isolation="host-api", verification_plan=("current-user-process-identity",),
    ), implementation=ProcessInspect(handles)))
