"""Immutable contracts for speech synthesis backends and playback devices."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from returns.result import Result

from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass


class SynthesisLocation(str, Enum):
    """Where synthesis executes and therefore where text may be disclosed."""

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class SynthesisFailureCode(str, Enum):
    """Stable failure categories exposed by speech boundaries."""

    INVALID_REQUEST = "INVALID_REQUEST"
    PROFILE_UNAVAILABLE = "PROFILE_UNAVAILABLE"
    VOICE_UNAVAILABLE = "VOICE_UNAVAILABLE"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_NOT_ALLOWED = "BACKEND_NOT_ALLOWED"
    LANGUAGE_UNSUPPORTED = "LANGUAGE_UNSUPPORTED"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    COST_LIMIT = "COST_LIMIT"
    CANCELLED = "CANCELLED"
    SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
    PLAYBACK_UNAVAILABLE = "PLAYBACK_UNAVAILABLE"
    PLAYBACK_FAILED = "PLAYBACK_FAILED"


class SpeechModel(BaseModel):
    """Strict immutable base model for speech records."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)


class VoiceBinding(SpeechModel):
    """Backend-specific voice selected by a portable voice profile."""

    backend_id: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)


class VoiceProfile(SpeechModel):
    """One user-facing voice mapped to one or more synthesis backends."""

    profile_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    bindings: tuple[VoiceBinding, ...] = Field(min_length=1)

    @field_validator("bindings")
    @classmethod
    def require_unique_backends(
        cls, bindings: tuple[VoiceBinding, ...]
    ) -> tuple[VoiceBinding, ...]:
        backend_ids = tuple(binding.backend_id for binding in bindings)
        if len(backend_ids) != len(set(backend_ids)):
            raise ValueError("voice profile backend bindings must be unique")
        return bindings

    def binding_for(self, backend_id: str) -> VoiceBinding | None:
        """Return the configured voice for a backend without inventing a fallback."""
        return next(
            (binding for binding in self.bindings if binding.backend_id == backend_id),
            None,
        )


class SynthesisProfile(SpeechModel):
    """Ordered quality, privacy, latency, cost, and resource policy."""

    profile_id: str = Field(min_length=1)
    backend_order: tuple[str, ...] = Field(min_length=1)
    allow_remote: bool = False
    require_streaming: bool = False
    max_audio_seconds: float = Field(default=60, gt=0)
    max_first_audio_latency_seconds: float = Field(gt=0)
    max_provider_cost: float = Field(default=0, ge=0)
    max_ram_bytes: int = Field(default=0, ge=0)
    max_vram_bytes: int = Field(default=0, ge=0)

    @field_validator("backend_order")
    @classmethod
    def require_unique_backends(cls, backend_order: tuple[str, ...]) -> tuple[str, ...]:
        if len(backend_order) != len(set(backend_order)):
            raise ValueError("synthesis profile backend order must be unique")
        return backend_order


class SynthesisRequest(SpeechModel):
    """Text and caller constraints passed to a selected backend."""

    text: str = Field(min_length=1, max_length=32768)
    language: str = Field(
        min_length=2,
        max_length=35,
        pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$",
    )
    profile_id: str = Field(min_length=1)
    voice_profile_id: str = Field(min_length=1)
    data_class: DataClass
    allow_remote: bool = False
    max_audio_seconds: float | None = Field(default=None, ge=0)
    max_first_audio_latency_seconds: float = Field(gt=0)
    max_provider_cost: float = Field(default=0, ge=0)
    max_ram_bytes: int = Field(default=0, ge=0)
    max_vram_bytes: int = Field(default=0, ge=0)
    allowed_backends: tuple[str, ...] = ()


class SynthesisBackendMetadata(SpeechModel):
    """Capabilities used for deterministic selection before loading a model."""

    backend_id: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    location: SynthesisLocation
    languages: tuple[str, ...] = Field(min_length=1)
    supports_streaming: bool = False
    supports_long_form: bool = False
    supports_voice_cloning: bool = False
    available: bool = True
    unavailable_reason: str | None = None
    estimated_first_audio_latency_seconds: float = Field(gt=0)
    estimated_cost_per_million_characters: float = Field(default=0, ge=0)
    required_ram_bytes: int = Field(default=0, ge=0)
    required_vram_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_availability_reason(self) -> SynthesisBackendMetadata:
        if not self.available and not self.unavailable_reason:
            raise ValueError("unavailable backends must include a reason")
        return self

    def supports_language(self, language: str) -> bool:
        """Match an exact BCP-47 tag or its primary language subtag."""
        normalized = language.casefold()
        primary = normalized.split("-", maxsplit=1)[0]
        supported = {item.casefold() for item in self.languages}
        supported_primary = {item.split("-", maxsplit=1)[0] for item in supported}
        return normalized in supported or primary in supported_primary

    def estimated_cost(self, character_count: int) -> float:
        """Return the declared upper-bound estimate for the given text."""
        return character_count * self.estimated_cost_per_million_characters / 1_000_000


class SynthesisFailure(SpeechModel):
    """Sanitized typed failure; request text is deliberately absent."""

    code: SynthesisFailureCode
    message: str = Field(min_length=1)
    backend_id: str | None = None
    retryable: bool = False
    attempted_backends: tuple[str, ...] = ()


class AudioChunk(SpeechModel):
    """One signed 16-bit little-endian PCM chunk held only in memory."""

    sequence: int = Field(ge=0)
    pcm_s16le: bytes = Field(min_length=2)
    sample_rate: int = Field(gt=0)
    channels: Literal[1, 2] = 1

    @field_validator("pcm_s16le")
    @classmethod
    def require_whole_frames(cls, payload: bytes) -> bytes:
        if len(payload) % 2:
            raise ValueError("PCM payload must contain complete 16-bit samples")
        return payload

    @model_validator(mode="after")
    def require_complete_frames(self) -> AudioChunk:
        if len(self.pcm_s16le) % (2 * self.channels):
            raise ValueError("PCM payload must contain complete channel frames")
        return self

    @property
    def duration_seconds(self) -> float:
        """Duration derived from the PCM frame count."""
        return len(self.pcm_s16le) / (2 * self.channels * self.sample_rate)


class SynthesizedAudio(SpeechModel):
    """Ephemeral backend output that must not cross the terminal result boundary."""

    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    chunks: tuple[AudioChunk, ...] = Field(min_length=1)
    generation_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def require_ordered_chunks(self) -> SynthesizedAudio:
        sequences = tuple(chunk.sequence for chunk in self.chunks)
        if sequences != tuple(range(len(self.chunks))):
            raise ValueError("audio chunk sequence must be contiguous and zero-based")
        return self

    @property
    def duration_seconds(self) -> float:
        """Total audio duration without retaining a second representation."""
        return sum(chunk.duration_seconds for chunk in self.chunks)


class PlaybackReceipt(SpeechModel):
    """Proof that playback, rather than only synthesis, reached completion."""

    device_id: str = Field(min_length=1)
    chunks_played: int = Field(gt=0)
    audio_seconds: float = Field(gt=0)
    completed: Literal[True] = True


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Optional-dependency-safe synthesis backend boundary."""

    @property
    def metadata(self) -> SynthesisBackendMetadata: ...

    async def synthesize(
        self,
        request: SynthesisRequest,
        voice: VoiceBinding,
        cancellation: CancellationToken,
    ) -> Result[SynthesizedAudio, SynthesisFailure]: ...


@runtime_checkable
class AudioPlayer(Protocol):
    """Playback boundary used by real devices and deterministic substitutes."""

    async def play(
        self, audio: SynthesizedAudio, cancellation: CancellationToken
    ) -> Result[PlaybackReceipt, SynthesisFailure]: ...
