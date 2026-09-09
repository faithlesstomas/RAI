"""Policy-bound speech synthesis actuator."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import ValidationError
from returns.result import Failure, Result, Success

from rai.kernel.ports import Actuator, CancellationToken
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    ProducerIdentity,
)

from .contracts import (
    AudioPlayer,
    SpeechSynthesizer,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisProfile,
    SynthesisRequest,
    VoiceProfile,
)
from .routing import SelectedSynthesizer, select_synthesizer

SPEECH_SYNTHESIS_CAPABILITY = "speech.synthesize"
SPEECH_PRODUCER = ProducerIdentity(
    producer_id="rai.speech", kind="actuator", version="1.0.0"
)


class SpeechSynthesisActuator(Actuator):
    """Select, synthesize, and play audio before producing a terminal result."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        profiles: tuple[SynthesisProfile, ...],
        voices: tuple[VoiceProfile, ...],
        backends: tuple[SpeechSynthesizer, ...],
        player: AudioPlayer,
        default_profile_id: str = "realtime_local",
        default_voice_profile_id: str = "default",
        remote_synthesis_enabled: bool = False,
    ) -> None:
        self._profiles: Mapping[str, SynthesisProfile] = {
            profile.profile_id: profile for profile in profiles
        }
        self._voices: Mapping[str, VoiceProfile] = {
            voice.profile_id: voice for voice in voices
        }
        self._backends = backends
        self._player = player
        self._default_profile_id = default_profile_id
        self._default_voice_profile_id = default_voice_profile_id
        self._remote_synthesis_enabled = remote_synthesis_enabled

    def _failure(
        self, request: CapabilityRequest, failure: SynthesisFailure
    ) -> Failure[ActionResult, ActionFailure]:
        return Failure(
            ActionFailure(
                record_id=f"failure:{request.record_id}:{failure.code}",
                timestamp=request.timestamp,
                producer=SPEECH_PRODUCER,
                correlation_id=request.correlation_id,
                request_id=request.record_id,
                capability=SPEECH_SYNTHESIS_CAPABILITY,
                code=failure.code,
                message=failure.message,
                retryable=failure.retryable,
            )
        )

    def _request(self, source: CapabilityRequest) -> SynthesisRequest:
        profile_id = source.arguments.get("profile", self._default_profile_id)
        voice_id = source.arguments.get("voice", self._default_voice_profile_id)
        language = source.arguments.get("language", "pl-PL")
        allow_remote = source.arguments.get("allow_remote", False)
        remote_target = urlparse(source.target_resource).scheme in {"http", "https"}
        budget = source.budget
        profile = self._profiles.get(profile_id)
        default_latency = (
            profile.max_first_audio_latency_seconds if profile is not None else 30
        )
        return SynthesisRequest(
            text=source.arguments.get("text"),
            language=language,
            profile_id=profile_id,
            voice_profile_id=voice_id,
            data_class=source.data_class,
            allow_remote=(
                self._remote_synthesis_enabled and allow_remote and remote_target
            ),
            max_audio_seconds=(
                min(profile.max_audio_seconds, budget.max_audio_seconds)
                if budget is not None and profile is not None
                else (profile.max_audio_seconds if profile is not None else None)
            ),
            max_first_audio_latency_seconds=(
                budget.max_latency_seconds if budget is not None else default_latency
            ),
            max_provider_cost=(
                budget.max_provider_cost
                if budget is not None
                else (profile.max_provider_cost if profile is not None else 0)
            ),
            max_ram_bytes=(
                budget.max_ram_bytes
                if budget is not None
                else (profile.max_ram_bytes if profile is not None else 0)
            ),
            max_vram_bytes=(
                budget.max_vram_bytes
                if budget is not None
                else (profile.max_vram_bytes if profile is not None else 0)
            ),
            allowed_backends=budget.allowed_providers if budget is not None else (),
        )

    async def act(  # noqa: PLR0911
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if request.capability != SPEECH_SYNTHESIS_CAPABILITY:
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.INVALID_REQUEST,
                    message="speech actuator received an unsupported capability",
                ),
            )
        if cancellation.cancelled:
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.CANCELLED,
                    message="speech synthesis was cancelled",
                ),
            )
        try:
            synthesis_request = self._request(request)
        except (TypeError, ValidationError):
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.INVALID_REQUEST,
                    message="speech synthesis arguments are invalid",
                ),
            )
        profile = self._profiles.get(synthesis_request.profile_id)
        if profile is None:
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.PROFILE_UNAVAILABLE,
                    message="the requested synthesis profile is not configured",
                ),
            )
        voice = self._voices.get(synthesis_request.voice_profile_id)
        if voice is None:
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.VOICE_UNAVAILABLE,
                    message="the requested voice profile is not configured",
                ),
            )
        selected_result = select_synthesizer(
            profile, synthesis_request, voice, self._backends
        )
        if isinstance(selected_result, Failure):
            return self._failure(request, selected_result.failure())
        selected: SelectedSynthesizer = selected_result.unwrap()
        synthesis_result = await selected.backend.synthesize(
            synthesis_request, selected.voice, cancellation
        )
        if isinstance(synthesis_result, Failure):
            return self._failure(request, synthesis_result.failure())
        audio = synthesis_result.unwrap()
        metadata = selected.backend.metadata
        if (
            audio.backend_id != metadata.backend_id
            or audio.model_id != metadata.model_id
            or audio.model_version != metadata.model_version
            or audio.voice_id != selected.voice.voice_id
        ):
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.SYNTHESIS_FAILED,
                    message="synthesis backend returned inconsistent metadata",
                    backend_id=metadata.backend_id,
                ),
            )
        if (
            synthesis_request.max_audio_seconds is not None
            and audio.duration_seconds > synthesis_request.max_audio_seconds
        ):
            return self._failure(
                request,
                SynthesisFailure(
                    code=SynthesisFailureCode.RESOURCE_LIMIT,
                    message="synthesized audio exceeds the request duration budget",
                    backend_id=metadata.backend_id,
                ),
            )
        playback_result = await self._player.play(audio, cancellation)
        if isinstance(playback_result, Failure):
            return self._failure(request, playback_result.failure())
        receipt = playback_result.unwrap()
        return Success(
            ActionResult(
                record_id=f"result:{request.record_id}",
                timestamp=request.timestamp,
                producer=SPEECH_PRODUCER,
                correlation_id=request.correlation_id,
                request_id=request.record_id,
                capability=SPEECH_SYNTHESIS_CAPABILITY,
                output={
                    "profile": profile.profile_id,
                    "voice": voice.profile_id,
                    "backend": metadata.backend_id,
                    "model": metadata.model_id,
                    "model_version": metadata.model_version,
                    "language": synthesis_request.language,
                    "device": receipt.device_id,
                    "chunks_played": receipt.chunks_played,
                    "audio_seconds": receipt.audio_seconds,
                },
                verification={"playback-completed": receipt.completed},
            )
        )
