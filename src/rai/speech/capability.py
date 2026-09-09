"""Capability-registry composition for speech synthesis."""

from __future__ import annotations

from returns.result import Result

from rai.kernel.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    RegisteredCapability,
)
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, RiskClass

from .actuator import SPEECH_SYNTHESIS_CAPABILITY, SpeechSynthesisActuator


class SpeechSynthesisCapability:
    """Expose an actuator through the shared policy and audit service path."""

    name = SPEECH_SYNTHESIS_CAPABILITY

    def __init__(self, actuator: SpeechSynthesisActuator) -> None:
        self._actuator = actuator

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        return await self._actuator.act(request, cancellation)


def speech_synthesis_descriptor() -> CapabilityDescriptor:
    """Return the transport-neutral schema and policy metadata for local speech."""
    return CapabilityDescriptor(
        name=SPEECH_SYNTHESIS_CAPABILITY,
        description="Synthesize text and play it through an authorized audio device.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "language": {"type": "string"},
                "profile": {"type": "string"},
                "voice": {"type": "string"},
                "allow_remote": {"type": "boolean"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk_class=RiskClass.LOW,
        side_effects=("audio.playback", "network"),
        isolation="in-process",
        verification_plan=("playback-completed",),
    )


def register_speech_synthesis(
    registry: CapabilityRegistry, actuator: SpeechSynthesisActuator
) -> None:
    """Register speech without importing an optional backend or audio library."""
    registry.register(
        RegisteredCapability(
            speech_synthesis_descriptor(),
            implementation=SpeechSynthesisCapability(actuator),
        )
    )
