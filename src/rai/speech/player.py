"""Lazy local audio playback adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import importlib
from typing import ContextManager, Protocol, cast

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken

from .contracts import (
    PlaybackReceipt,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesizedAudio,
)


class AudioOutputStream(Protocol):
    """Small portion of sounddevice.RawOutputStream used by the adapter."""

    def write(self, data: bytes) -> object: ...


AudioStreamFactory = Callable[
    [int, int, str, str | None], ContextManager[AudioOutputStream]
]


class _PlaybackDependencyUnavailable(RuntimeError):
    pass


class _PlaybackCancelled(RuntimeError):
    pass


def _default_stream_factory(
    sample_rate: int, channels: int, dtype: str, device: str | None
) -> ContextManager[AudioOutputStream]:
    try:
        module = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise _PlaybackDependencyUnavailable(
            "audio playback is unavailable; install the 'tts' optional dependencies"
        ) from exc
    stream_type = getattr(module, "RawOutputStream", None)
    if stream_type is None:
        raise _PlaybackDependencyUnavailable(
            "the installed sounddevice package has no RawOutputStream"
        )
    stream = stream_type(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        device=device,
    )
    return cast(ContextManager[AudioOutputStream], stream)


class SoundDeviceAudioPlayer:
    """Write PCM chunks locally while observing cancellation between chunks."""

    def __init__(
        self,
        device: str | None = None,
        *,
        stream_factory: AudioStreamFactory | None = None,
    ) -> None:
        self._device = device
        self._stream_factory = stream_factory or _default_stream_factory

    def _play_blocking(
        self, audio: SynthesizedAudio, cancellation: CancellationToken
    ) -> None:
        first = audio.chunks[0]
        if any(
            chunk.sample_rate != first.sample_rate or chunk.channels != first.channels
            for chunk in audio.chunks
        ):
            raise ValueError("audio chunks have inconsistent stream formats")
        with self._stream_factory(
            first.sample_rate,
            first.channels,
            "int16",
            self._device,
        ) as stream:
            for chunk in audio.chunks:
                if cancellation.cancelled:
                    raise _PlaybackCancelled
                if stream.write(chunk.pcm_s16le) is True:
                    raise OSError("audio stream underflow")

    async def play(
        self, audio: SynthesizedAudio, cancellation: CancellationToken
    ) -> Result[PlaybackReceipt, SynthesisFailure]:
        if cancellation.cancelled:
            return Failure(self._cancelled(audio.backend_id))
        try:
            await asyncio.to_thread(self._play_blocking, audio, cancellation)
        except _PlaybackCancelled:
            return Failure(self._cancelled(audio.backend_id))
        except _PlaybackDependencyUnavailable as exc:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.PLAYBACK_UNAVAILABLE,
                    message=str(exc),
                    backend_id=audio.backend_id,
                )
            )
        except (OSError, RuntimeError, ValueError):
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.PLAYBACK_FAILED,
                    message="local audio playback failed",
                    backend_id=audio.backend_id,
                    retryable=True,
                )
            )
        return Success(
            PlaybackReceipt(
                device_id=self._device or "audio:default",
                chunks_played=len(audio.chunks),
                audio_seconds=audio.duration_seconds,
            )
        )

    @staticmethod
    def _cancelled(backend_id: str) -> SynthesisFailure:
        return SynthesisFailure(
            code=SynthesisFailureCode.CANCELLED,
            message="audio playback was cancelled",
            backend_id=backend_id,
        )
