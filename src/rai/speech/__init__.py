"""Provider-neutral speech synthesis contracts and policy-aware routing."""

from .actuator import SpeechSynthesisActuator, create_default_speech_actuator
from .capability import register_speech_synthesis, speech_synthesis_descriptor
from .lemonade import (
    LemonadeSpeechSynthesizer,
    LemonadeSpeechTranscriber,
    SpeechTranscriber,
    TranscriptionResult,
)
from .normalization import normalize_speech_text
from .profiles import default_synthesis_profiles

__all__ = [
    "LemonadeSpeechSynthesizer",
    "LemonadeSpeechTranscriber",
    "SpeechSynthesisActuator",
    "SpeechTranscriber",
    "TranscriptionResult",
    "create_default_speech_actuator",
    "default_synthesis_profiles",
    "normalize_speech_text",
    "register_speech_synthesis",
    "speech_synthesis_descriptor",
]
