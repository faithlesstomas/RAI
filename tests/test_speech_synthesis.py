"""Conformance and policy tests for the provider-neutral TTS foundation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from returns.result import Failure, Success

from rai.kernel.ports import Actuator, CancellationToken
from rai.kernel.records import CapabilityRequest, DataClass, ProducerIdentity
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.speech.actuator import SpeechSynthesisActuator
from rai.speech.capability import register_speech_synthesis, speech_synthesis_descriptor
from rai.speech.contracts import (
    AudioPlayer,
    SpeechSynthesizer,
    SynthesisBackendMetadata,
    SynthesisLocation,
    SynthesisProfile,
    SynthesisRequest,
    VoiceBinding,
    VoiceProfile,
)
from rai.speech.profiles import default_synthesis_profiles
from rai.speech.routing import select_synthesizer
from rai.speech.synthetic import (
    DeterministicAudioPlayer,
    DeterministicSpeechSynthesizer,
)

TEST_PRODUCER = ProducerIdentity(
    producer_id="test:speech", kind="test", version="1.0.0"
)


def _metadata(  # noqa: PLR0913
    backend_id: str,
    *,
    location: SynthesisLocation = SynthesisLocation.LOCAL,
    languages: tuple[str, ...] = ("pl", "en"),
    available: bool = True,
    latency: float = 0.1,
    cost: float = 0,
    ram: int = 0,
) -> SynthesisBackendMetadata:
    return SynthesisBackendMetadata(
        backend_id=backend_id,
        backend_version="1.0.0",
        model_id=f"test/{backend_id}",
        model_version="1",
        location=location,
        languages=languages,
        available=available,
        unavailable_reason=None if available else "not installed",
        estimated_first_audio_latency_seconds=latency,
        estimated_cost_per_million_characters=cost,
        required_ram_bytes=ram,
    )


def _voice(*backend_ids: str) -> VoiceProfile:
    return VoiceProfile(
        profile_id="default",
        display_name="Default test voice",
        bindings=tuple(
            VoiceBinding(backend_id=item, voice_id=f"voice/{item}")
            for item in backend_ids
        ),
    )


def _synthesis_request(
    *,
    profile_id: str = "quality_local",
    data_class: DataClass = DataClass.LOCAL,
    allow_remote: bool = False,
) -> SynthesisRequest:
    return SynthesisRequest(
        text="Dzień dobry",
        language="pl-PL",
        profile_id=profile_id,
        voice_profile_id="default",
        data_class=data_class,
        allow_remote=allow_remote,
        max_first_audio_latency_seconds=120,
        max_provider_cost=1,
        max_ram_bytes=16 * 1024**3,
        max_vram_bytes=12 * 1024**3,
    )


def _capability_request(**arguments: object) -> CapabilityRequest:
    return CapabilityRequest(
        record_id="request:speech",
        timestamp=datetime(2026, 9, 8, tzinfo=timezone.utc),
        producer=TEST_PRODUCER,
        actor=TEST_PRODUCER,
        capability="speech.synthesize",
        arguments={"text": "Dzień dobry", **arguments},
        data_class=DataClass.LOCAL,
        target_resource="audio:default",
        requested_side_effects=("audio.playback",),
        isolation="in-process",
        verification_plan=("playback-completed",),
    )


def test_default_profiles_never_hide_a_local_to_remote_fallback() -> None:
    profiles = {item.profile_id: item for item in default_synthesis_profiles()}

    assert profiles["realtime_local"].backend_order == ("piper",)
    assert profiles["realtime_local"].allow_remote is False
    assert profiles["quality_local"].allow_remote is False
    assert profiles["premium_remote"].allow_remote is True


def test_quality_profile_falls_back_to_first_eligible_local_backend() -> None:
    unavailable = DeterministicSpeechSynthesizer(
        _metadata("chatterbox", available=False)
    )
    xtts = DeterministicSpeechSynthesizer(_metadata("xtts", ram=8 * 1024**3))
    profile = next(
        item
        for item in default_synthesis_profiles()
        if item.profile_id == "quality_local"
    )

    result = select_synthesizer(
        profile,
        _synthesis_request(),
        _voice("chatterbox", "xtts"),
        (unavailable, xtts),
    )

    assert isinstance(result, Success)
    assert result.unwrap().backend.metadata.backend_id == "xtts"


@pytest.mark.parametrize("data_class", [DataClass.SECRET, DataClass.BLOCKED])
def test_remote_backend_is_rejected_for_sensitive_text(data_class: DataClass) -> None:
    remote = DeterministicSpeechSynthesizer(
        _metadata("elevenlabs", location=SynthesisLocation.REMOTE, cost=100)
    )
    profile = next(
        item
        for item in default_synthesis_profiles()
        if item.profile_id == "premium_remote"
    )

    result = select_synthesizer(
        profile,
        _synthesis_request(
            profile_id="premium_remote", data_class=data_class, allow_remote=True
        ),
        _voice("elevenlabs"),
        (remote,),
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "BACKEND_NOT_ALLOWED"


def test_remote_backend_requires_both_profile_and_request_consent() -> None:
    remote = DeterministicSpeechSynthesizer(
        _metadata("elevenlabs", location=SynthesisLocation.REMOTE, cost=100)
    )
    profile = SynthesisProfile(
        profile_id="remote",
        backend_order=("elevenlabs",),
        allow_remote=True,
        max_audio_seconds=60,
        max_first_audio_latency_seconds=10,
        max_provider_cost=1,
    )

    denied = select_synthesizer(
        profile,
        _synthesis_request(profile_id="remote"),
        _voice("elevenlabs"),
        (remote,),
    )
    allowed = select_synthesizer(
        profile,
        _synthesis_request(profile_id="remote", allow_remote=True),
        _voice("elevenlabs"),
        (remote,),
    )

    assert isinstance(denied, Failure)
    assert isinstance(allowed, Success)


async def test_actuator_returns_only_after_deterministic_playback() -> None:
    backend = DeterministicSpeechSynthesizer(_metadata("piper"))
    player = DeterministicAudioPlayer()
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("piper"),),
        backends=(backend,),
        player=player,
    )

    result = await actuator.act(_capability_request(), CancellationToken())

    assert isinstance(actuator, Actuator)
    assert isinstance(backend, SpeechSynthesizer)
    assert isinstance(player, AudioPlayer)
    assert isinstance(result, Success)
    assert player.play_count == 1
    assert result.unwrap().verification == {"playback-completed": True}
    assert "text" not in result.unwrap().output
    assert "audio" not in result.unwrap().output
    assert result.unwrap().output["backend"] == "piper"


async def test_actuator_does_not_treat_caller_remote_flag_as_authorization() -> None:
    remote = DeterministicSpeechSynthesizer(
        _metadata("elevenlabs", location=SynthesisLocation.REMOTE, cost=100)
    )
    player = DeterministicAudioPlayer()
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("elevenlabs"),),
        backends=(remote,),
        player=player,
    )

    result = await actuator.act(
        _capability_request(profile="premium_remote", allow_remote=True),
        CancellationToken(),
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "BACKEND_NOT_ALLOWED"
    assert player.play_count == 0


async def test_capability_service_policies_and_audits_completed_playback() -> None:
    backend = DeterministicSpeechSynthesizer(_metadata("piper"))
    player = DeterministicAudioPlayer()
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("piper"),),
        backends=(backend,),
        player=player,
    )
    registry = CapabilityRegistry()
    register_speech_synthesis(registry, actuator)
    audit = InMemoryAuditLedger()
    service = CapabilityService(registry, PolicyEngine(), audit)
    request = normalize_request(
        speech_synthesis_descriptor(),
        {"text": "Dzień dobry", "language": "pl-PL"},
        timestamp=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )

    decision, result = await service.invoke(request)

    assert decision is not None and decision.outcome == "ALLOW"
    assert isinstance(result, Success)
    assert player.play_count == 1
    assert tuple(entry.stage for entry in audit.entries) == ("DECISION", "TERMINAL")
    assert audit.entries[-1].result == result.unwrap()


async def test_private_remote_synthesis_is_stopped_by_shared_policy() -> None:
    remote = DeterministicSpeechSynthesizer(
        _metadata("elevenlabs", location=SynthesisLocation.REMOTE, cost=100)
    )
    player = DeterministicAudioPlayer()
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("elevenlabs"),),
        backends=(remote,),
        player=player,
        remote_synthesis_enabled=True,
    )
    registry = CapabilityRegistry()
    register_speech_synthesis(registry, actuator)
    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger())
    request = normalize_request(
        speech_synthesis_descriptor(),
        {
            "text": "Poufna treść",
            "language": "pl-PL",
            "profile": "premium_remote",
            "allow_remote": True,
        },
        data_class=DataClass.PRIVATE,
        target_resource="https://api.example.invalid/tts",
    )

    decision, result = await service.invoke(request)

    assert decision is not None and decision.outcome == "ESCALATE"
    assert isinstance(result, Failure)
    assert result.failure().code == "ESCALATION_REQUIRED"
    assert player.play_count == 0


async def test_actuator_cancellation_is_terminal_and_skips_playback() -> None:
    backend = DeterministicSpeechSynthesizer(_metadata("piper"))
    player = DeterministicAudioPlayer()
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("piper"),),
        backends=(backend,),
        player=player,
    )
    cancellation = CancellationToken()
    cancellation.cancel()

    result = await actuator.act(_capability_request(), cancellation)

    assert isinstance(result, Failure)
    assert result.failure().code == "CANCELLED"
    assert result.failure().terminal is True
    assert player.play_count == 0


async def test_invalid_text_is_reported_without_echoing_it() -> None:
    actuator = SpeechSynthesisActuator(
        profiles=default_synthesis_profiles(),
        voices=(_voice("piper"),),
        backends=(DeterministicSpeechSynthesizer(_metadata("piper")),),
        player=DeterministicAudioPlayer(),
    )

    result = await actuator.act(_capability_request(text=""), CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "INVALID_REQUEST"
    assert "Dzień dobry" not in result.failure().message


def test_normalize_speech_text() -> None:
    from rai.speech.normalization import normalize_speech_text

    markdown = """# Witaj!
Oto **ważne** powiadomienie z [dokumentacji](https://example.com/docs).
- Punkt 1: sprawdź `kod`
- Punkt 2: wykonaj:
```bash
echo hello
```
> Cytat na koniec
"""
    cleaned = normalize_speech_text(markdown)
    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "dokumentacji" in cleaned
    assert "Witaj!" in cleaned
    assert "ważne powiadomienie" in cleaned
    assert "sprawdź kod" in cleaned
    assert "echo hello" in cleaned


@pytest.mark.asyncio
async def test_create_default_speech_actuator_unavailable_when_empty(tmp_path: Path) -> None:
    from rai.speech.actuator import create_default_speech_actuator

    actuator = create_default_speech_actuator(voices_dir=tmp_path)
    result = await actuator.act(_capability_request(text="Dzień dobry"), CancellationToken())
    assert isinstance(result, Failure)
    assert result.failure().code == "BACKEND_UNAVAILABLE"
