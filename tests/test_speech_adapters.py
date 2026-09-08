"""Tests for lazy local Piper synthesis and sounddevice playback adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from returns.result import Failure, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass
from rai.speech.contracts import (
    AudioChunk,
    SynthesisRequest,
    SynthesizedAudio,
    VoiceBinding,
)
from rai.speech.piper import (
    PiperSpeechSynthesizer,
    PiperVoiceFiles,
    resolve_local_piper_voice,
)
from rai.speech.player import SoundDeviceAudioPlayer
from rai.tts import resolve_voice_path

EXPECTED_CHUNKS = 2


@dataclass(frozen=True)
class _FakePiperChunk:
    audio_int16_bytes: bytes


@dataclass(frozen=True)
class _FakePiperConfig:
    sample_rate: int = 16000


class _FakePiperVoice:
    config = _FakePiperConfig()

    def synthesize(self, text: str) -> tuple[_FakePiperChunk, ...]:
        del text
        return (
            _FakePiperChunk(b"\x00\x00" * 160),
            _FakePiperChunk(b"\x01\x00" * 160),
        )


class _FakeStream:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exception: object) -> None:
        return None

    def write(self, data: bytes) -> object:
        self.writes.append(data)
        return None


def _request(max_audio_seconds: float | None = None) -> SynthesisRequest:
    return SynthesisRequest(
        text="Dzień dobry",
        language="pl-PL",
        profile_id="realtime_local",
        voice_profile_id="default",
        data_class=DataClass.LOCAL,
        max_audio_seconds=max_audio_seconds,
        max_first_audio_latency_seconds=2,
        max_ram_bytes=2 * 1024**3,
    )


def _files(tmp_path: Path) -> PiperVoiceFiles:
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    model.write_bytes(b"test-model")
    config.write_text("{}", encoding="utf-8")
    return PiperVoiceFiles(model_path=model, config_path=config)


def _audio() -> SynthesizedAudio:
    return SynthesizedAudio(
        backend_id="piper",
        model_id="test/piper",
        model_version="1",
        voice_id="voice/piper",
        chunks=(
            AudioChunk(
                sequence=0,
                pcm_s16le=b"\x00\x00" * 160,
                sample_rate=16000,
            ),
            AudioChunk(
                sequence=1,
                pcm_s16le=b"\x01\x00" * 160,
                sample_rate=16000,
            ),
        ),
        generation_seconds=0,
    )


def test_resolve_local_voice_never_provisions_or_uses_path_traversal(
    tmp_path: Path,
) -> None:
    voice_dir = tmp_path / "pl-test"
    voice_dir.mkdir()
    expected = _files(voice_dir)

    found = resolve_local_piper_voice("pl-test", tmp_path)
    traversal = resolve_local_piper_voice("../voice", tmp_path)
    missing = resolve_local_piper_voice("not-installed", tmp_path)

    assert isinstance(found, Success)
    assert found.unwrap() == expected
    assert isinstance(traversal, Failure)
    assert traversal.failure().code == "INVALID_REQUEST"
    assert isinstance(missing, Failure)
    assert missing.failure().code == "VOICE_UNAVAILABLE"


def test_legacy_voice_resolver_is_import_safe_and_local_only(tmp_path: Path) -> None:
    voice_dir = tmp_path / "pl-test"
    voice_dir.mkdir()
    files = _files(voice_dir)

    assert resolve_voice_path("pl-test", str(tmp_path)) == str(files.model_path)
    assert resolve_voice_path("not-installed", str(tmp_path)) is None


async def test_piper_backend_loads_lazily_and_returns_pcm_metadata(tmp_path: Path) -> None:
    load_calls: list[tuple[Path, Path]] = []

    def load(model_path: Path, config_path: Path) -> _FakePiperVoice:
        load_calls.append((model_path, config_path))
        return _FakePiperVoice()

    files = _files(tmp_path)
    backend = PiperSpeechSynthesizer(
        files,
        language="pl",
        model_id="piper/pl-test",
        model_version="fixture",
        loader=load,
    )

    result = await backend.synthesize(
        _request(),
        VoiceBinding(backend_id="piper", voice_id="pl-test"),
        CancellationToken(),
    )

    assert isinstance(result, Success)
    assert load_calls == [(files.model_path, files.config_path)]
    assert result.unwrap().backend_id == "piper"
    assert result.unwrap().voice_id == "pl-test"
    assert len(result.unwrap().chunks) == EXPECTED_CHUNKS


async def test_piper_backend_fails_typed_when_artifacts_disappear(tmp_path: Path) -> None:
    backend = PiperSpeechSynthesizer(
        PiperVoiceFiles(
            model_path=tmp_path / "missing.onnx",
            config_path=tmp_path / "missing.onnx.json",
        ),
        language="pl",
        model_id="piper/missing",
        model_version="fixture",
    )

    result = await backend.synthesize(
        _request(),
        VoiceBinding(backend_id="piper", voice_id="missing"),
        CancellationToken(),
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "BACKEND_UNAVAILABLE"


async def test_piper_backend_enforces_audio_duration_budget(tmp_path: Path) -> None:
    backend = PiperSpeechSynthesizer(
        _files(tmp_path),
        language="pl",
        model_id="piper/pl-test",
        model_version="fixture",
        loader=lambda _model, _config: _FakePiperVoice(),
    )

    result = await backend.synthesize(
        _request(max_audio_seconds=0),
        VoiceBinding(backend_id="piper", voice_id="pl-test"),
        CancellationToken(),
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "RESOURCE_LIMIT"


async def test_sounddevice_player_writes_all_chunks_before_receipt() -> None:
    stream = _FakeStream()
    factory_calls: list[tuple[int, int, str, str | None]] = []

    def factory(
        sample_rate: int, channels: int, dtype: str, device: str | None
    ) -> _FakeStream:
        factory_calls.append((sample_rate, channels, dtype, device))
        return stream

    player = SoundDeviceAudioPlayer(device="test-device", stream_factory=factory)
    audio = _audio()

    result = await player.play(audio, CancellationToken())

    assert isinstance(result, Success)
    assert factory_calls == [(16000, 1, "int16", "test-device")]
    assert stream.writes == [chunk.pcm_s16le for chunk in audio.chunks]
    assert result.unwrap().completed is True
    assert result.unwrap().chunks_played == EXPECTED_CHUNKS


async def test_sounddevice_player_rejects_inconsistent_chunks() -> None:
    first = _audio().chunks[0]
    audio = _audio().model_copy(
        update={
            "chunks": (
                first,
                AudioChunk(
                    sequence=1,
                    pcm_s16le=b"\x00\x00",
                    sample_rate=22050,
                ),
            )
        }
    )
    player = SoundDeviceAudioPlayer(stream_factory=lambda *_args: _FakeStream())

    result = await player.play(audio, CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "PLAYBACK_FAILED"


async def test_sounddevice_player_reports_device_failure_without_audio_data() -> None:
    def unavailable(*_arguments: object) -> _FakeStream:
        raise OSError("host device details must not cross the boundary")

    player = SoundDeviceAudioPlayer(stream_factory=unavailable)

    result = await player.play(_audio(), CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "PLAYBACK_FAILED"
    assert "host device" not in result.failure().message
