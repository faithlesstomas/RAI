"""Lazy, local-only Piper synthesis backend and explicit voice discovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import importlib
from pathlib import Path
import threading
import time
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken

from .contracts import (
    AudioChunk,
    SynthesisBackendMetadata,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisLocation,
    SynthesisRequest,
    SynthesizedAudio,
    VoiceBinding,
)


class PiperVoiceFiles(BaseModel):
    """Explicit local Piper artifact pair; provisioning is a separate operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_path: Path
    config_path: Path


class _PiperChunk(Protocol):
    audio_int16_bytes: bytes


class _PiperConfig(Protocol):
    sample_rate: int


class _PiperVoice(Protocol):
    config: _PiperConfig

    def synthesize(self, text: str) -> Iterable[_PiperChunk]: ...


PiperVoiceLoader = Callable[[Path, Path], _PiperVoice]


class _PiperDependencyUnavailable(RuntimeError):
    pass


class _PiperAudioBudgetExceeded(RuntimeError):
    pass


class _PiperSynthesisCancelled(RuntimeError):
    pass


def resolve_local_piper_voice(
    voice: str | Path, data_directory: Path
) -> Result[PiperVoiceFiles, SynthesisFailure]:
    """Resolve only existing local artifacts; never download during invocation."""
    direct = Path(voice)
    if direct.is_file():
        model_path = direct
    else:
        voice_id = str(voice)
        if Path(voice_id).name != voice_id or voice_id in {".", ".."}:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.INVALID_REQUEST,
                    message="Piper voice ID must be a name or an existing model path",
                    backend_id="piper",
                )
            )
        candidates = sorted((data_directory / voice_id).glob("*.onnx"))
        if not candidates:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.VOICE_UNAVAILABLE,
                    message="Piper voice artifacts are not provisioned locally",
                    backend_id="piper",
                )
            )
        model_path = candidates[0]
    config_path = Path(f"{model_path}.json")
    if model_path.suffix != ".onnx" or not config_path.is_file():
        return Failure(
            SynthesisFailure(
                code=SynthesisFailureCode.VOICE_UNAVAILABLE,
                message="Piper voice requires matching .onnx and .onnx.json files",
                backend_id="piper",
            )
        )
    return Success(PiperVoiceFiles(model_path=model_path, config_path=config_path))


def _default_loader(model_path: Path, config_path: Path) -> _PiperVoice:
    try:
        module = importlib.import_module("piper.voice")
    except ImportError as exc:
        raise _PiperDependencyUnavailable(
            "Piper is unavailable; install the 'tts' optional dependencies"
        ) from exc
    voice_type = getattr(module, "PiperVoice", None)
    if voice_type is None:
        raise _PiperDependencyUnavailable("the installed Piper package has no PiperVoice")
    loaded = voice_type.load(
        model_path=str(model_path),
        config_path=str(config_path),
    )
    return cast(_PiperVoice, loaded)


class PiperSpeechSynthesizer:
    """Run Piper off the event loop with no import-time load or network access."""

    def __init__(  # noqa: PLR0913
        self,
        files: PiperVoiceFiles,
        *,
        language: str,
        model_id: str,
        model_version: str,
        backend_version: str = "unknown",
        loader: PiperVoiceLoader | None = None,
        required_ram_bytes: int = 256 * 1024**2,
    ) -> None:
        self._files = files
        self._loader = loader or _default_loader
        self._voice: _PiperVoice | None = None
        self._load_lock = threading.Lock()
        self._metadata = SynthesisBackendMetadata(
            backend_id="piper",
            backend_version=backend_version,
            model_id=model_id,
            model_version=model_version,
            location=SynthesisLocation.LOCAL,
            languages=(language,),
            supports_streaming=False,
            supports_long_form=True,
            estimated_first_audio_latency_seconds=2,
            required_ram_bytes=required_ram_bytes,
        )

    @property
    def metadata(self) -> SynthesisBackendMetadata:
        return self._metadata

    def _loaded_voice(self) -> _PiperVoice:
        with self._load_lock:
            if self._voice is None:
                self._voice = self._loader(
                    self._files.model_path,
                    self._files.config_path,
                )
            return self._voice

    def _synthesize_blocking(
        self,
        request: SynthesisRequest,
        voice_binding: VoiceBinding,
        cancellation: CancellationToken,
    ) -> SynthesizedAudio:
        started = time.monotonic()
        voice = self._loaded_voice()
        chunks: list[AudioChunk] = []
        duration = 0.0
        for sequence, piper_chunk in enumerate(voice.synthesize(request.text)):
            if cancellation.cancelled:
                raise _PiperSynthesisCancelled
            chunk = AudioChunk(
                sequence=sequence,
                pcm_s16le=piper_chunk.audio_int16_bytes,
                sample_rate=voice.config.sample_rate,
            )
            duration += chunk.duration_seconds
            if (
                request.max_audio_seconds is not None
                and duration > request.max_audio_seconds
            ):
                raise _PiperAudioBudgetExceeded
            chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Piper generated no audio")
        return SynthesizedAudio(
            backend_id=self.metadata.backend_id,
            model_id=self.metadata.model_id,
            model_version=self.metadata.model_version,
            voice_id=voice_binding.voice_id,
            chunks=tuple(chunks),
            generation_seconds=time.monotonic() - started,
        )

    async def synthesize(  # noqa: PLR0911
        self,
        request: SynthesisRequest,
        voice: VoiceBinding,
        cancellation: CancellationToken,
    ) -> Result[SynthesizedAudio, SynthesisFailure]:
        if cancellation.cancelled:
            return Failure(self._cancelled())
        if voice.backend_id != self.metadata.backend_id:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.INVALID_REQUEST,
                    message="voice binding does not belong to the Piper backend",
                    backend_id="piper",
                )
            )
        if not self._files.model_path.is_file() or not self._files.config_path.is_file():
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.BACKEND_UNAVAILABLE,
                    message="Piper voice artifacts are unavailable",
                    backend_id="piper",
                )
            )
        try:
            audio = await asyncio.to_thread(
                self._synthesize_blocking,
                request,
                voice,
                cancellation,
            )
        except _PiperDependencyUnavailable as exc:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.BACKEND_UNAVAILABLE,
                    message=str(exc),
                    backend_id="piper",
                )
            )
        except _PiperAudioBudgetExceeded:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.RESOURCE_LIMIT,
                    message="Piper output exceeds the request audio duration budget",
                    backend_id="piper",
                )
            )
        except _PiperSynthesisCancelled:
            return Failure(self._cancelled())
        except (OSError, RuntimeError, ValueError):
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.SYNTHESIS_FAILED,
                    message="Piper failed to synthesize audio",
                    backend_id="piper",
                    retryable=True,
                )
            )
        if cancellation.cancelled:
            return Failure(self._cancelled())
        return Success(audio)

    @staticmethod
    def _cancelled() -> SynthesisFailure:
        return SynthesisFailure(
            code=SynthesisFailureCode.CANCELLED,
            message="Piper synthesis was cancelled",
            backend_id="piper",
        )
