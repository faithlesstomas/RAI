"""Scoped volume control through PulseAudio/PipeWire's pactl API."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, RiskClass, _utc_now
from .capabilities import HANDLE_TTL, PRODUCER, failure, result
from .commands import run_command
from .handles import SQLiteHandleStore
from .records import ResourceHandle

MAX_VOLUME = 100
VOLUME_TOLERANCE = 1


class VolumeBackend(Protocol):
    async def read(self, token: CancellationToken) -> Result[dict, str]: ...
    async def set(self, sink: str, percent: int, token: CancellationToken) -> Result[None, str]: ...


class PactlVolumeBackend:
    async def read(self, token: CancellationToken) -> Result[dict, str]:
        default = await run_command("pactl", ("get-default-sink",), token)
        sinks = await run_command("pactl", ("--format=json", "list", "sinks"), token)
        if isinstance(default, Failure) or isinstance(sinks, Failure):
            return Failure("AUDIO_UNAVAILABLE")
        try:
            sink = next(item for item in json.loads(sinks.unwrap()) if item["name"] == default.unwrap().strip())
            volumes = [int(channel["value_percent"].rstrip("%")) for channel in sink["volume"].values()]
            identity = json.dumps({"name": sink["name"], "index": sink["index"], "card": sink.get("card")}, sort_keys=True)
            return Success({"sink": sink["name"], "volumes": volumes, "muted": sink["mute"],
                            "fingerprint": hashlib.sha256(identity.encode()).hexdigest()})
        except (ValueError, KeyError, TypeError, StopIteration):
            return Failure("INVALID_AUDIO_STATE")

    async def set(self, sink: str, percent: int, token: CancellationToken) -> Result[None, str]:
        changed = await run_command("pactl", ("set-sink-volume", sink, f"{percent}%"), token)
        return Failure(changed.failure()) if isinstance(changed, Failure) else Success(None)


class VolumeCapability:
    def __init__(self, name: str, backend: VolumeBackend, handles: SQLiteHandleStore) -> None:
        self.name, self.backend, self.handles = name, backend, handles

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:
        read = await self.backend.read(cancellation)
        if isinstance(read, Failure):
            return Failure(failure(request, read.failure()))
        state = read.unwrap()
        if self.name == "system.volume.get":
            now = _utc_now()
            handle = ResourceHandle(producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id,
                                    task_id=request.arguments["task_id"], kind="audio_sink", target=state["sink"],
                                    fingerprint=state["fingerprint"], operations=("system.volume.set",),
                                    expires_at=now + HANDLE_TTL, data_class=request.data_class)
            issued = self.handles.issue(handle)
            if isinstance(issued, Failure):
                return Failure(failure(request, issued.failure()))
            return Success(result(request, {**state, "handle": issued.unwrap()}, {"pactl_state_read": True}))
        return await self._set(request, cancellation, state)

    async def _set(self, request: CapabilityRequest, token: CancellationToken, before: dict) -> Result[ActionResult, ActionFailure]:  # noqa: PLR0911
        percent = request.arguments["percent"]
        if not 0 <= percent <= MAX_VOLUME:
            return Failure(failure(request, "INVALID_ARGUMENT"))
        resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], operation=self.name)
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        if handle.kind != "audio_sink" or handle.target != before["sink"] or handle.fingerprint != before["fingerprint"]:
            return Failure(failure(request, "STALE_RESOURCE"))
        changed = await self.backend.set(handle.target, percent, token)
        if isinstance(changed, Failure):
            return Failure(failure(request, changed.failure()))
        verified = await self.backend.read(token)
        if isinstance(verified, Failure):
            return Failure(failure(request, "UNKNOWN"))
        after = verified.unwrap()
        if after["fingerprint"] != handle.fingerprint or not after["volumes"] or any(abs(v - percent) > VOLUME_TOLERANCE for v in after["volumes"]):
            return Failure(failure(request, "POSTCONDITION_FAILED"))
        return Success(result(request, {"status": "SUCCEEDED", "percent": percent, "previous": before["volumes"]},
                              {"observed_percent": after["volumes"], "sink_fingerprint": after["fingerprint"]}))


def register_volume_capabilities(registry: CapabilityRegistry, handles: SQLiteHandleStore, backend: VolumeBackend) -> None:
    for name, fields, risk, effects in (
        ("system.volume.get", {"task_id": {"type": "string"}}, RiskClass.LOW, ()),
        ("system.volume.set", {"task_id": {"type": "string"}, "handle": {"type": "string"}, "percent": {"type": "integer"}},
         RiskClass.MODERATE, ("audio-volume",)),
    ):
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name=name, description=f"Bounded {name} (v1).",
            input_schema={"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False},
            risk_class=risk, side_effects=effects, isolation="host-api", verification_plan=("audio-state-readback",),
        ), implementation=VolumeCapability(name, backend, handles)))
