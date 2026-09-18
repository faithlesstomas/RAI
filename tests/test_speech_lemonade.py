"""
Unit tests for Lemonade TTS and STT speech adapters.
"""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from returns.result import Failure, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass
from rai.speech.contracts import (
    SynthesisFailureCode,
    SynthesisRequest,
    VoiceBinding,
)
from rai.speech.lemonade import (
    LemonadeSpeechSynthesizer,
    LemonadeSpeechTranscriber,
    TranscriptionResult,
)


def _generate_dummy_wav(sample_rate: int = 24000, duration_seconds: float = 0.05) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(sample_rate * duration_seconds)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_lemonade_speech_synthesizer_metadata() -> None:
    synthesizer = LemonadeSpeechSynthesizer()
    meta = synthesizer.metadata
    assert meta.backend_id == "lemonade"
    assert meta.model_id == "kokoro-v1"
    assert meta.supports_language("en")
    assert meta.supports_language("pl")


@pytest.mark.asyncio
async def test_lemonade_speech_synthesizer_success() -> None:
    synthesizer = LemonadeSpeechSynthesizer()
    wav_payload = _generate_dummy_wav(sample_rate=24000, duration_seconds=0.1)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        200,
        content=wav_payload,
        request=httpx.Request("POST", "http://127.0.0.1:13305/v1/audio/speech"),
    )
    mock_client.post.return_value = mock_resp

    request = SynthesisRequest(
        text="Dzień dobry!",
        language="pl",
        profile_id="realtime_local",
        voice_profile_id="default_pl",
        data_class=DataClass.PUBLIC,
        max_first_audio_latency_seconds=2.0,
    )
    voice = VoiceBinding(backend_id="lemonade", voice_id="pf_dora")

    with patch.object(synthesizer, "_get_client", return_value=mock_client):
        res = await synthesizer.synthesize(request, voice, CancellationToken())
        assert isinstance(res, Success)
        audio = res.unwrap()
        assert audio.backend_id == "lemonade"
        assert audio.model_id == "kokoro-v1"
        assert len(audio.chunks) == 1
        SAMPLE_RATE = 24000
        assert audio.chunks[0].sample_rate == SAMPLE_RATE
        assert audio.chunks[0].channels == 1
        assert audio.duration_seconds > 0.0


@pytest.mark.asyncio
async def test_lemonade_speech_synthesizer_cancellation() -> None:
    synthesizer = LemonadeSpeechSynthesizer()
    token = CancellationToken()
    token.cancel()

    request = SynthesisRequest(
        text="Cancelled text",
        language="en",
        profile_id="realtime_local",
        voice_profile_id="default_en",
        data_class=DataClass.PUBLIC,
        max_first_audio_latency_seconds=2.0,
    )
    voice = VoiceBinding(backend_id="lemonade", voice_id="af_bella")

    res = await synthesizer.synthesize(request, voice, token)
    assert isinstance(res, Failure)
    assert res.failure().code == SynthesisFailureCode.CANCELLED


@pytest.mark.asyncio
async def test_lemonade_speech_transcriber_success() -> None:
    transcriber = LemonadeSpeechTranscriber()
    dummy_wav = _generate_dummy_wav()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        200,
        json={"text": "Test transcription from Whisper", "language": "en", "duration": 1.2},
        request=httpx.Request("POST", "http://127.0.0.1:13305/v1/audio/transcriptions"),
    )
    mock_client.post.return_value = mock_resp

    with patch.object(transcriber, "_get_client", return_value=mock_client):
        res = await transcriber.transcribe(
            audio_data=dummy_wav,
            model="Whisper-Small",
            language="en",
        )
        assert isinstance(res, Success)
        result: TranscriptionResult = res.unwrap()
        assert result.text == "Test transcription from Whisper"
        assert result.language == "en"
        DURATION = 1.2
        assert result.duration_seconds == DURATION


@pytest.mark.asyncio
async def test_lemonade_speech_transcriber_failure() -> None:
    transcriber = LemonadeSpeechTranscriber()
    dummy_wav = _generate_dummy_wav()

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        500,
        text="Internal Server Error in whispercpp",
        request=httpx.Request("POST", "http://127.0.0.1:13305/v1/audio/transcriptions"),
    )
    mock_client.post.return_value = mock_resp

    with patch.object(transcriber, "_get_client", return_value=mock_client):
        res = await transcriber.transcribe(audio_data=dummy_wav)
        assert isinstance(res, Failure)
        assert res.failure().code == SynthesisFailureCode.SYNTHESIS_FAILED
