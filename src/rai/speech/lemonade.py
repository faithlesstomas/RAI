"""
Lemonade speech synthesis and transcription adapters (Kokoro TTS and Whisper STT).
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable
import wave

import httpx
from pydantic import Field
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken

from .contracts import (
    AudioChunk,
    SpeechModel,
    SpeechSynthesizer,
    SynthesisBackendMetadata,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisLocation,
    SynthesisRequest,
    SynthesizedAudio,
    VoiceBinding,
)

_HTTP_OK = 200
_HTTP_SERVER_ERROR = 500
_WAV_HEADER_MIN_SIZE = 44

logger = logging.getLogger(__name__)
DEFAULT_LEMONADE_HOST = "http://127.0.0.1:13305"


class TranscriptionResult(SpeechModel):
    """Result of speech-to-text audio transcription."""

    text: str = Field(min_length=0)
    language: Optional[str] = None
    duration_seconds: float = Field(default=0.0, ge=0.0)


@runtime_checkable
class SpeechTranscriber(Protocol):
    """Protocol for local/remote audio speech transcription."""

    async def transcribe(
        self,
        audio_data: bytes,
        model: str = "Whisper-Base",
        language: Optional[str] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> Result[TranscriptionResult, SynthesisFailure]: ...


class LemonadeSpeechSynthesizer(SpeechSynthesizer):
    """Speech synthesis adapter using Lemonade Server Kokoro TTS endpoint."""

    def __init__(
        self,
        host: str = DEFAULT_LEMONADE_HOST,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
        return self._client

    @property
    def metadata(self) -> SynthesisBackendMetadata:
        return SynthesisBackendMetadata(
            backend_id="lemonade",
            backend_version="1.0.0",
            model_id="kokoro-v1",
            model_version="1.0.0",
            location=SynthesisLocation.LOCAL,
            languages=("en", "en-us", "en-gb", "pl", "es", "fr", "de", "it", "ja", "zh"),
            supports_streaming=False,
            supports_long_form=True,
            supports_voice_cloning=False,
            available=True,
            estimated_first_audio_latency_seconds=0.2,
            estimated_cost_per_million_characters=0.0,
            required_ram_bytes=256 * 1024 * 1024,
            required_vram_bytes=0,
        )

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
                    message="Speech synthesis was cancelled before invocation",
                    backend_id="lemonade",
                )
            )

        start_time = time.monotonic()
        payload = {
            "model": "kokoro-v1",
            "input": request.text,
            "voice": voice.voice_id or "af_bella",
            "response_format": "wav",
        }

        try:
            client = self._get_client()
            resp = await client.post("/v1/audio/speech", json=payload)
            if resp.status_code != _HTTP_OK:
                return Failure(
                    SynthesisFailure(
                        code=SynthesisFailureCode.SYNTHESIS_FAILED,
                        message=f"Lemonade TTS returned HTTP {resp.status_code}: {resp.text}",
                        backend_id="lemonade",
                        retryable=resp.status_code >= _HTTP_SERVER_ERROR,
                    )
                )

            wav_bytes = resp.content
            if len(wav_bytes) < _WAV_HEADER_MIN_SIZE:
                return Failure(
                    SynthesisFailure(
                        code=SynthesisFailureCode.SYNTHESIS_FAILED,
                        message="Lemonade TTS returned invalid audio payload",
                        backend_id="lemonade",
                    )
                )

            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())

            if channels not in (1, 2):
                channels = 1

            chunk = AudioChunk(
                sequence=0,
                pcm_s16le=frames,
                sample_rate=sample_rate,
                channels=channels,  # type: ignore[arg-type]
            )

            audio = SynthesizedAudio(
                backend_id="lemonade",
                model_id="kokoro-v1",
                model_version="1.0.0",
                voice_id=voice.voice_id,
                chunks=(chunk,),
                generation_seconds=time.monotonic() - start_time,
            )
            return Success(audio)
        except Exception as exc:  # noqa: BLE001
            if cancellation.cancelled:
                return Failure(
                    SynthesisFailure(
                        code=SynthesisFailureCode.CANCELLED,
                        message="Speech synthesis cancelled",
                        backend_id="lemonade",
                    )
                )
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.BACKEND_UNAVAILABLE,
                    message=f"Lemonade TTS request failed: {exc}",
                    backend_id="lemonade",
                    retryable=True,
                )
            )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to close Lemonade TTS client: %s", exc)
            self._client = None


class LemonadeSpeechTranscriber(SpeechTranscriber):
    """Speech transcription adapter using Lemonade Server Whisper STT endpoint."""

    def __init__(
        self,
        host: str = DEFAULT_LEMONADE_HOST,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        model: str = "Whisper-Base",
        language: Optional[str] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> Result[TranscriptionResult, SynthesisFailure]:
        if cancellation and cancellation.cancelled:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.CANCELLED,
                    message="Transcription cancelled before invocation",
                    backend_id="lemonade",
                )
            )

        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        data: dict[str, Any] = {"model": model}
        if language:
            data["language"] = language

        try:
            client = self._get_client()
            resp = await client.post("/v1/audio/transcriptions", files=files, data=data)
            if resp.status_code != _HTTP_OK:
                return Failure(
                    SynthesisFailure(
                        code=SynthesisFailureCode.SYNTHESIS_FAILED,
                        message=f"Lemonade transcription failed with HTTP {resp.status_code}: {resp.text}",
                        backend_id="lemonade",
                        retryable=resp.status_code >= _HTTP_SERVER_ERROR,
                    )
                )

            res_json = resp.json()
            text = str(res_json.get("text", "")).strip()
            detected_lang = res_json.get("language")
            duration = float(res_json.get("duration", 0.0))

            return Success(
                TranscriptionResult(
                    text=text,
                    language=detected_lang,
                    duration_seconds=duration,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if cancellation and cancellation.cancelled:
                return Failure(
                    SynthesisFailure(
                        code=SynthesisFailureCode.CANCELLED,
                        message="Transcription cancelled",
                        backend_id="lemonade",
                    )
                )
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.BACKEND_UNAVAILABLE,
                    message=f"Lemonade STT request failed: {exc}",
                    backend_id="lemonade",
                    retryable=True,
                )
            )

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to close Lemonade STT client: %s", exc)
            self._client = None
