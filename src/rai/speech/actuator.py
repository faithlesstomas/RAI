"""Policy-bound speech synthesis actuator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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
from rai.paths import data_dir

from .contracts import (
    AudioPlayer,
    SpeechSynthesizer,
    SynthesisBackendMetadata,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisLocation,
    SynthesisProfile,
    SynthesisRequest,
    SynthesizedAudio,
    VoiceBinding,
    VoiceProfile,
)
from .normalization import normalize_speech_text
from .piper import PiperSpeechSynthesizer, resolve_local_piper_voice
from .player import SoundDeviceAudioPlayer
from .profiles import default_synthesis_profiles
from .routing import SelectedSynthesizer, select_synthesizer
from .synthetic import DeterministicAudioPlayer

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
        raw_text = source.arguments.get("text")
        clean_text = normalize_speech_text(raw_text) if isinstance(raw_text, str) else raw_text
        return SynthesisRequest(
            text=clean_text,
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


def create_default_speech_actuator(  # noqa: PLR0912
    voices_dir: Path | None = None,
    device: str | None = None,
) -> SpeechSynthesisActuator:
    """Compose the default policy-bound speech actuator with local Piper voice models."""
    effective_dir = Path(voices_dir) if voices_dir is not None else (data_dir() / "piper_voices")
    profiles = default_synthesis_profiles()

    preferred_default_voices = (
        "pl_PL-gosia-medium",
        "pl_PL-darkman-medium",
        "en_GB-cori-medium",
        "en_GB-cori-high",
    )

    discovered_voice_ids: list[str] = []
    if effective_dir.is_dir():
        for item in effective_dir.iterdir():
            if item.is_dir() and any(item.glob("*.onnx")):
                discovered_voice_ids.append(item.name)

    default_voice_name: str | None = None
    for preferred in preferred_default_voices:
        if preferred in discovered_voice_ids:
            default_voice_name = preferred
            break
    if default_voice_name is None and discovered_voice_ids:
        default_voice_name = discovered_voice_ids[0]

    if default_voice_name is not None:
        default_files_res = resolve_local_piper_voice(default_voice_name, effective_dir)
        if isinstance(default_files_res, Success):
            default_files = default_files_res.unwrap()
            langs = set()
            for vid in discovered_voice_ids:
                if vid.startswith("pl"):
                    langs.update(("pl", "pl-PL"))
                elif vid.startswith("en"):
                    langs.update(("en", "en-GB", "en-US"))
            if not langs:
                langs = {"pl", "pl-PL"}

            backend = PiperSpeechSynthesizer(
                default_files,
                language="pl-PL",
                languages=tuple(sorted(langs)),
                model_id=f"piper/{default_voice_name}",
                model_version="medium",
                voices_dir=effective_dir,
            )

            voice_profiles: list[VoiceProfile] = [
                VoiceProfile(
                    profile_id="default",
                    display_name=f"Default ({default_voice_name})",
                    bindings=(VoiceBinding(backend_id="piper", voice_id="default"),),
                )
            ]
            for vid in discovered_voice_ids:
                voice_profiles.append(
                    VoiceProfile(
                        profile_id=vid,
                        display_name=vid,
                        bindings=(VoiceBinding(backend_id="piper", voice_id=vid),),
                    )
                )

            player = SoundDeviceAudioPlayer(device=device)
            return SpeechSynthesisActuator(
                profiles=profiles,
                voices=tuple(voice_profiles),
                backends=(backend,),
                player=player,
            )

    class _UnavailableSynthesizer:
        metadata = SynthesisBackendMetadata(
            backend_id="piper",
            backend_version="unknown",
            model_id="none",
            model_version="0",
            location=SynthesisLocation.LOCAL,
            languages=("pl", "en"),
            available=False,
            unavailable_reason=f"No Piper voices provisioned in {effective_dir}",
            estimated_first_audio_latency_seconds=2,
        )

        async def synthesize(  # noqa: PLR6301
            self,
            _request: SynthesisRequest,
            _voice: VoiceBinding,
            _cancellation: CancellationToken,
        ) -> Result[SynthesizedAudio, SynthesisFailure]:
            return Failure(
                SynthesisFailure(
                    code=SynthesisFailureCode.BACKEND_UNAVAILABLE,
                    message=f"No Piper voices provisioned in {effective_dir}",
                    backend_id="piper",
                )
            )

    fallback_voice = VoiceProfile(
        profile_id="default",
        display_name="Unavailable Voice",
        bindings=(VoiceBinding(backend_id="piper", voice_id="default"),),
    )
    return SpeechSynthesisActuator(
        profiles=profiles,
        voices=(fallback_voice,),
        backends=(_UnavailableSynthesizer(),),  # type: ignore[arg-type]
        player=DeterministicAudioPlayer(),
    )
