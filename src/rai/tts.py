"""
This module handles Text-to-Speech (TTS) functionality using Piper TTS.
"""

from __future__ import annotations

import asyncio
import threading
import wave
from pathlib import Path
from typing import Optional

import logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import sounddevice as sd
    from piper.voice import PiperVoice
except ImportError:
    np = None
    sd = None
    PiperVoice = None


# pylint: disable=too-few-public-methods
class TTS:
    """A class to handle Piper TTS operations."""

    def __init__(self, model_path: str) -> None:
        logger.debug("Initializing TTS instance...")
        self._voice = self._load_voice(model_path)

    def _load_voice(self, model_path: str) -> Optional[PiperVoice]:
        """Loads the Piper voice model, handling potential ImportError."""
        if PiperVoice is None:
            logger.error(
                "TTS Error: dependencies are not installed. "
                "Please run 'pip install rich-ai[tts]'"
            )
            return None
        try:
            logger.debug("Loading model from: %s", model_path)
            config_path = f"{model_path}.json"
            logger.debug("Using config path: %s", config_path)
            voice = PiperVoice.load(model_path=model_path, config_path=config_path)
            logger.debug("Model loaded successfully.")
            return voice
        except (IOError, ValueError, RuntimeError) as e:
            logger.error("TTS Error loading model: %s", e)
            return None

    async def synthesize(self, text: str, output_path: str | None = None) -> None:
        """
        Synthesizes text to speech and either plays it or saves it to a file.
        """
        if not self._voice:
            logger.warning("Aborting synthesis: voice not loaded.")
            return

        if output_path:
            with wave.open(output_path, "wb") as wav_file:
                self._voice.synthesize(text, wav_file)
        else:
            logger.debug("Synthesizing for direct playback.")
            await self._play_audio(text)

    async def _play_audio(self, text: str) -> None:
        """Synthesizes audio and plays it using sounddevice in a non-blocking way."""
        if np is None or sd is None:
            logger.error(
                "TTS Error: playback dependencies are not installed. "
                "Please run 'pip install rich-ai[tts]'"
            )
            return
        logger.debug("[dim][TTS Debug] TTS._play_audio() called.[/dim]")

        stop_event = threading.Event()

        try:

            def blocking_synthesis() -> bytes:
                logger.debug("[dim][TTS Debug] Generating audio bytes...[/dim]")
                audio_chunks = self._voice.synthesize(text)
                return b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)

            audio_data = await asyncio.to_thread(blocking_synthesis)

            if not audio_data:
                logger.error("TTS Error: No audio data was generated.")
                return

            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            def playback_thread_func() -> None:
                """
                This function runs in a separate thread. It starts playback and then
                polls a stop_event to allow for graceful interruption.
                """
                try:
                    sd.play(audio_array, samplerate=self._voice.config.sample_rate)
                    logger.debug(
                        "Playing audio with sample rate %s...",
                        self._voice.config.sample_rate,
                    )

                    # Poll until playback is finished or a stop is requested
                    while sd.get_stream().active:
                        if stop_event.is_set():
                            sd.stop()
                            logger.debug("Playback stopped by event.")
                            break
                        # Use the event's wait method for a non-busy, interruptible wait
                        stop_event.wait(0.1)  # Poll every 100ms

                    logger.debug("Playback thread finished.")
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("[red]Error in playback thread: %s[/red]", e)

            # Run the blocking playback function in a separate thread
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, playback_thread_func)
            logger.debug("Playback executor task finished.")

        except asyncio.CancelledError:
            logger.debug("[TTS Debug] Playback cancelled. Signaling thread to stop.")
            stop_event.set()
            raise
        except ImportError:
            logger.error(
                "TTS Error: playback dependencies are not installed. "
                "Please run 'pip install rich-ai[tts]'"
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("TTS Error playing audio: %s", e)


def resolve_voice_path(voice_input: str, data_dir_str: str) -> str | None:
    """
    Resolves the voice input to a valid .onnx model path.
    1. Checks if the input is a direct path to a file.
    2. If not, treats it as a voice ID and searches for an existing .onnx file.

    Voice provisioning is deliberately separate; this function never downloads.
    """
    data_dir = Path(data_dir_str)
    direct = Path(voice_input)
    if direct.is_file() and Path(f"{direct}.json").is_file():
        return str(direct)

    voice_dir = data_dir / voice_input
    if voice_dir.is_dir():
        onnx_files = sorted(voice_dir.glob("*.onnx"))
        for model_path in onnx_files:
            if Path(f"{model_path}.json").is_file():
                return str(model_path)

    logger.error(
        "Voice '%s' is not provisioned locally; provide matching .onnx and .onnx.json files.",
        voice_input,
    )
    return None
