"""
Protocols and Data Structures for RAI Local Inference.

This module defines the core abstractions for local model execution using
structural subtyping (Protocols) and functional error handling (Returns).
"""

from typing import Any, AsyncIterator, List, Protocol, runtime_checkable, Dict, Optional
from dataclasses import dataclass

from returns.result import Result


@dataclass(frozen=True)
class GenerationStats:
    """Statistics for a generation request."""
    input_tokens: int
    output_tokens: int
    total_time_sec: float
    tokens_per_sec: float
    model_load_time_sec: Optional[float] = None


@dataclass(frozen=True)
class InferenceResult:
    """
    Standardized result from an inference engine.
    Wraps the raw text and telemetry data.
    """
    text: str
    stats: Optional[GenerationStats] = None
    finish_reason: str = "stop"  # stop, length, error


@runtime_checkable
class InferenceEngine(Protocol):
    """
    Protocol for a low-level local inference engine.
    
    Implementations (IREE, Llama.cpp) must satisfy this interface.
    All methods must be efficient and side-effect free where possible.
    """

    def generate(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Result[InferenceResult, Exception]:
        """
        Synchronously generates text from a prompt.
        
        Args:
            prompt: The input text.
            stop: List of stop sequences.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Result[InferenceResult, Exception]: Success with text/stats or Failure.
        """
        ...

    def stream(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> AsyncIterator[Result[str, Exception]]:
        """
        Asynchronously streams generated tokens.
        
        Args:
            prompt: The input text.
            stop: List of stop sequences.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Yields:
             AsyncIterator[Result[str, Exception]]: Stream of token chunks or failures.
        """
        ...

    def unload(self) -> None:
        """Free resources associated with the model."""
        ...
