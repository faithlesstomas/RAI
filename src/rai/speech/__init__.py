"""Provider-neutral speech synthesis contracts and policy-aware routing."""

from .actuator import SpeechSynthesisActuator, create_default_speech_actuator
from .capability import register_speech_synthesis, speech_synthesis_descriptor
from .normalization import normalize_speech_text
from .profiles import default_synthesis_profiles

__all__ = [
    "SpeechSynthesisActuator",
    "create_default_speech_actuator",
    "default_synthesis_profiles",
    "normalize_speech_text",
    "register_speech_synthesis",
    "speech_synthesis_descriptor",
]
