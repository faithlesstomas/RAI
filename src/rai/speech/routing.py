"""Pure selection of a synthesis backend from explicit user and policy constraints."""

from __future__ import annotations

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from rai.kernel.records import DataClass

from .contracts import (
    SpeechSynthesizer,
    SynthesisFailure,
    SynthesisFailureCode,
    SynthesisLocation,
    SynthesisProfile,
    SynthesisRequest,
    VoiceBinding,
    VoiceProfile,
)


@dataclass(frozen=True)
class SelectedSynthesizer:
    """Backend plus its explicit voice binding."""

    backend: SpeechSynthesizer
    voice: VoiceBinding


def _within_limit(value: float | int, limit: float | int) -> bool:
    return value <= limit


def _remote_allowed(profile: SynthesisProfile, request: SynthesisRequest) -> bool:
    return (
        profile.allow_remote
        and request.allow_remote
        and request.data_class not in {DataClass.SECRET, DataClass.BLOCKED}
    )


def select_synthesizer(
    profile: SynthesisProfile,
    request: SynthesisRequest,
    voice_profile: VoiceProfile,
    backends: tuple[SpeechSynthesizer, ...],
) -> Result[SelectedSynthesizer, SynthesisFailure]:
    """Select the first eligible backend; never silently widen privacy constraints."""
    by_id = {backend.metadata.backend_id: backend for backend in backends}
    attempted: list[str] = []
    rejections: list[SynthesisFailureCode] = []

    for backend_id in profile.backend_order:
        attempted.append(backend_id)
        if (
            request.allowed_backends
            and backend_id not in request.allowed_backends
        ):
            rejections.append(SynthesisFailureCode.BACKEND_NOT_ALLOWED)
            continue
        backend = by_id.get(backend_id)
        if backend is None:
            rejections.append(SynthesisFailureCode.BACKEND_UNAVAILABLE)
            continue
        binding = voice_profile.binding_for(backend_id)
        if binding is None:
            rejections.append(SynthesisFailureCode.VOICE_UNAVAILABLE)
            continue
        metadata = backend.metadata
        if not metadata.available:
            rejections.append(SynthesisFailureCode.BACKEND_UNAVAILABLE)
            continue
        if not metadata.supports_language(request.language):
            rejections.append(SynthesisFailureCode.LANGUAGE_UNSUPPORTED)
            continue
        if metadata.location == SynthesisLocation.REMOTE and not _remote_allowed(
            profile, request
        ):
            rejections.append(SynthesisFailureCode.BACKEND_NOT_ALLOWED)
            continue
        if profile.require_streaming and not metadata.supports_streaming:
            rejections.append(SynthesisFailureCode.RESOURCE_LIMIT)
            continue
        latency_limit = min(
            profile.max_first_audio_latency_seconds,
            request.max_first_audio_latency_seconds,
        )
        if metadata.estimated_first_audio_latency_seconds > latency_limit:
            rejections.append(SynthesisFailureCode.RESOURCE_LIMIT)
            continue
        cost_limit = min(profile.max_provider_cost, request.max_provider_cost)
        if not _within_limit(metadata.estimated_cost(len(request.text)), cost_limit):
            rejections.append(SynthesisFailureCode.COST_LIMIT)
            continue
        ram_limit = min(profile.max_ram_bytes, request.max_ram_bytes)
        if not _within_limit(metadata.required_ram_bytes, ram_limit):
            rejections.append(SynthesisFailureCode.RESOURCE_LIMIT)
            continue
        vram_limit = min(profile.max_vram_bytes, request.max_vram_bytes)
        if not _within_limit(metadata.required_vram_bytes, vram_limit):
            rejections.append(SynthesisFailureCode.RESOURCE_LIMIT)
            continue
        return Success(SelectedSynthesizer(backend=backend, voice=binding))

    priority = (
        SynthesisFailureCode.BACKEND_NOT_ALLOWED,
        SynthesisFailureCode.COST_LIMIT,
        SynthesisFailureCode.RESOURCE_LIMIT,
        SynthesisFailureCode.LANGUAGE_UNSUPPORTED,
        SynthesisFailureCode.VOICE_UNAVAILABLE,
        SynthesisFailureCode.BACKEND_UNAVAILABLE,
    )
    code = next((item for item in priority if item in rejections), priority[-1])
    messages = {
        SynthesisFailureCode.BACKEND_NOT_ALLOWED: (
            "no synthesis backend is allowed by the active privacy constraints"
        ),
        SynthesisFailureCode.COST_LIMIT: (
            "no synthesis backend satisfies the provider cost budget"
        ),
        SynthesisFailureCode.RESOURCE_LIMIT: (
            "no synthesis backend satisfies the latency or resource budget"
        ),
        SynthesisFailureCode.LANGUAGE_UNSUPPORTED: (
            "no configured synthesis backend supports the requested language"
        ),
        SynthesisFailureCode.VOICE_UNAVAILABLE: (
            "the voice profile has no binding for an eligible synthesis backend"
        ),
        SynthesisFailureCode.BACKEND_UNAVAILABLE: (
            "no configured synthesis backend is currently available"
        ),
    }
    return Failure(
        SynthesisFailure(
            code=code,
            message=messages[code],
            attempted_backends=tuple(attempted),
        )
    )
