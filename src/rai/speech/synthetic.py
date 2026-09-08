"""Deterministic speech substitutes for tests and installations without TTS extras."""

from __future__ import annotations

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken

from .contracts import (
    AudioChunk,
    PlaybackReceipt,
    SynthesisBackendMetadata,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisRequest,
    SynthesizedAudio,
    VoiceBinding,
)


class DeterministicSpeechSynthesizer:
    """Generate bounded silence without a model, network, microphone, or speaker."""

    def __init__(self, metadata: SynthesisBackendMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> SynthesisBackendMetadata:
        return self._metadata

    async def synthesize(
        self,
        request: SynthesisRequest,
        voice: VoiceBinding,
        cancellation: CancellationToken,
    ) -> Result[SynthesizedAudio, SynthesisFailure]:
        if cancellation.cancelled:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.CANCELLED,
                    message="speech synthesis was cancelled",
                    backend_id=self.metadata.backend_id,
                )
            )
        frames = max(1, min(len(request.text), 100)) * 160
        chunk = AudioChunk(
            sequence=0,
            pcm_s16le=b"\x00\x00" * frames,
            sample_rate=16000,
        )
        return Success(
            SynthesizedAudio(
                backend_id=self.metadata.backend_id,
                model_id=self.metadata.model_id,
                model_version=self.metadata.model_version,
                voice_id=voice.voice_id,
                chunks=(chunk,),
                generation_seconds=0,
            )
        )


class DeterministicAudioPlayer:
    """A playback substitute whose receipt is the only retained output."""

    def __init__(self, device_id: str = "test:silent-player") -> None:
        self.device_id = device_id
        self.play_count = 0

    async def play(
        self, audio: SynthesizedAudio, cancellation: CancellationToken
    ) -> Result[PlaybackReceipt, SynthesisFailure]:
        if cancellation.cancelled:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.CANCELLED,
                    message="speech playback was cancelled",
                    backend_id=audio.backend_id,
                )
            )
        self.play_count += 1
        return Success(
            PlaybackReceipt(
                device_id=self.device_id,
                chunks_played=len(audio.chunks),
                audio_seconds=audio.duration_seconds,
            )
        )
