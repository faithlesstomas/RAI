"""Built-in speech profiles; callers may replace every value from configuration."""

from __future__ import annotations

from .contracts import SynthesisProfile


def default_synthesis_profiles() -> tuple[SynthesisProfile, ...]:
    """Return conservative defaults with no implicit local-to-cloud fallback."""
    gibibyte = 1024**3
    return (
        SynthesisProfile(
            profile_id="realtime_local",
            backend_order=("piper",),
            max_audio_seconds=60,
            max_first_audio_latency_seconds=2,
            max_ram_bytes=2 * gibibyte,
        ),
        SynthesisProfile(
            profile_id="quality_local",
            backend_order=("chatterbox", "xtts", "piper"),
            max_audio_seconds=3600,
            max_first_audio_latency_seconds=120,
            max_ram_bytes=16 * gibibyte,
            max_vram_bytes=12 * gibibyte,
        ),
        SynthesisProfile(
            profile_id="premium_remote",
            backend_order=("elevenlabs", "gemini", "chatterbox", "xtts", "piper"),
            allow_remote=True,
            max_audio_seconds=3600,
            max_first_audio_latency_seconds=30,
            max_provider_cost=1,
            max_ram_bytes=16 * gibibyte,
            max_vram_bytes=12 * gibibyte,
        ),
    )
