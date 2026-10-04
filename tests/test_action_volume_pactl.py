"""pactl adapter parsing and volume capability failure semantics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from returns.result import Failure, Success

import rai.actions.volume as volume
from rai.actions.handles import SQLiteHandleStore
from rai.actions.volume import PactlVolumeBackend, VolumeCapability, register_volume_capabilities
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity
from rai.kernel.transport import normalize_request

SINK = {"name": "alsa", "index": 1, "card": 2, "mute": False,
        "volume": {"front-left": {"value_percent": "30%"}, "front-right": {"value_percent": "31%"}}}


def stub_pactl(monkeypatch: pytest.MonkeyPatch, outputs: dict[str, Any]) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    async def run_command(program: str, arguments: tuple[str, ...], token: CancellationToken) -> Any:
        assert program == "pactl"
        calls.append(arguments)
        return outputs[arguments[0] if arguments[0] != "--format=json" else "list"]

    monkeypatch.setattr(volume, "run_command", run_command)
    return calls


async def test_read_parses_default_sink_and_fingerprints_stable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_pactl(monkeypatch, {"get-default-sink": Success("alsa\n"), "list": Success(json.dumps([{**SINK, "name": "other"}, SINK]))})
    state = (await PactlVolumeBackend().read(CancellationToken())).unwrap()
    assert state["sink"] == "alsa" and state["volumes"] == [30, 31] and state["muted"] is False
    again = (await PactlVolumeBackend().read(CancellationToken())).unwrap()
    assert again["fingerprint"] == state["fingerprint"]


@pytest.mark.parametrize("outputs,code", [
    ({"get-default-sink": Failure("COMMAND_FAILED"), "list": Success("[]")}, "AUDIO_UNAVAILABLE"),
    ({"get-default-sink": Success("alsa"), "list": Failure("BACKEND_UNAVAILABLE")}, "AUDIO_UNAVAILABLE"),
    ({"get-default-sink": Success("missing"), "list": Success(json.dumps([SINK]))}, "INVALID_AUDIO_STATE"),
    ({"get-default-sink": Success("alsa"), "list": Success("not json")}, "INVALID_AUDIO_STATE"),
    ({"get-default-sink": Success("alsa"), "list": Success(json.dumps([{**SINK, "volume": {"l": {"value_percent": "loud"}}}]))},
     "INVALID_AUDIO_STATE"),
])
async def test_read_reports_unavailable_or_untrusted_audio_state(
    monkeypatch: pytest.MonkeyPatch, outputs: dict, code: str,
) -> None:
    stub_pactl(monkeypatch, outputs)
    assert await PactlVolumeBackend().read(CancellationToken()) == Failure(code)


async def test_set_passes_sink_and_percent_as_separate_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = stub_pactl(monkeypatch, {"set-sink-volume": Success("")})
    assert await PactlVolumeBackend().set("alsa; reboot", 40, CancellationToken()) == Success(None)
    assert calls == [("set-sink-volume", "alsa; reboot", "40%")]
    stub_pactl(monkeypatch, {"set-sink-volume": Failure("COMMAND_FAILED")})
    assert await PactlVolumeBackend().set("alsa", 40, CancellationToken()) == Failure("COMMAND_FAILED")


class Audio:
    """Scripted backend: successive reads pop from the queue; the last value repeats."""

    def __init__(self, *reads: Any, set_result: Any = Success(None)) -> None:
        self.reads, self.set_result, self.sets = list(reads), set_result, []

    async def read(self, token: CancellationToken) -> Any:
        return self.reads.pop(0) if len(self.reads) > 1 else self.reads[0]

    async def set(self, sink: str, percent: int, token: CancellationToken) -> Any:
        self.sets.append((sink, percent))
        return self.set_result


def state(volumes: list[int], fingerprint: str = "fp", sink: str = "default") -> Success:
    return Success({"sink": sink, "fingerprint": fingerprint, "volumes": volumes, "muted": False})


async def run(tmp_path: Path, backend: Audio, percent: int = 40, actor: ProducerIdentity | None = None,
              set_task: str = "t") -> Any:
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_volume_capabilities(registry, handles, backend)
    getter = VolumeCapability("system.volume.get", backend, handles)
    setter = VolumeCapability("system.volume.set", backend, handles)
    read = normalize_request(registry.descriptor("system.volume.get"), {"task_id": "t"})
    handle = (await getter.invoke(read, CancellationToken())).unwrap().output["handle"]
    request = normalize_request(registry.descriptor("system.volume.set"),
                                {"task_id": set_task, "handle": handle, "percent": percent})
    if actor is not None:
        request = request.model_copy(update={"actor": actor})
    return await setter.invoke(request, CancellationToken())


async def test_unreadable_device_fails_before_any_write(tmp_path: Path) -> None:
    backend = Audio(state([30]), Failure("AUDIO_UNAVAILABLE"))
    outcome = await run(tmp_path, backend)
    assert outcome.failure().code == "AUDIO_UNAVAILABLE" and not backend.sets


async def test_backend_write_failure_is_reported(tmp_path: Path) -> None:
    backend = Audio(state([30]), set_result=Failure("COMMAND_FAILED"))
    assert (await run(tmp_path, backend)).failure().code == "COMMAND_FAILED"


async def test_unreadable_readback_after_write_is_unknown(tmp_path: Path) -> None:
    backend = Audio(state([30]), state([30]), Failure("AUDIO_UNAVAILABLE"))
    outcome = await run(tmp_path, backend)
    assert outcome.failure().code == "UNKNOWN" and backend.sets == [("default", 40)]


@pytest.mark.parametrize("after", [state([]), state([40], fingerprint="other"), state([45])])
async def test_readback_must_show_same_device_and_requested_level(tmp_path: Path, after: Success) -> None:
    outcome = await run(tmp_path, Audio(state([30]), state([30]), after))
    assert outcome.failure().code == "POSTCONDITION_FAILED"


async def test_level_within_tolerance_is_accepted(tmp_path: Path) -> None:
    outcome = await run(tmp_path, Audio(state([30]), state([30]), state([41, 40])))
    assert outcome.unwrap().output == {"status": "SUCCEEDED", "percent": 40, "previous": (30,)}


async def test_handle_is_bound_to_actor_and_task(tmp_path: Path) -> None:
    intruder = ProducerIdentity(producer_id="intruder", kind="user", version="1.0.0")
    backend = Audio(state([30]))
    outcome = await run(tmp_path, backend, actor=intruder)
    assert isinstance(outcome, Failure) and not backend.sets
    wrong_task = await run(tmp_path, backend, set_task="other")
    assert isinstance(wrong_task, Failure) and not backend.sets


async def test_handle_for_different_sink_name_is_stale(tmp_path: Path) -> None:
    # The sink name changes between issuing the handle and the write, even though the fingerprint is unchanged.
    backend = Audio(state([30], sink="first"), state([30], sink="second"))
    outcome = await run(tmp_path, backend)
    assert outcome.failure().code == "STALE_RESOURCE" and not backend.sets
