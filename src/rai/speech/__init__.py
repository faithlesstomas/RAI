"""Provider-neutral speech synthesis contracts and policy-aware routing."""

from .actuator import SpeechSynthesisActuator
from .capability import register_speech_synthesis, speech_synthesis_descriptor
from .profiles import default_synthesis_profiles

__all__ = [
    "SpeechSynthesisActuator",
    "default_synthesis_profiles",
    "register_speech_synthesis",
    "speech_synthesis_descriptor",
]
