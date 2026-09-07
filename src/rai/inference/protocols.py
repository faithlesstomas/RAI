"""
Protocols and Data Structures for RAI Local Inference.

This module defines the core abstractions for local model execution using
structural subtyping (Protocols) and functional error handling (Returns).
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from returns.result import Result

from rai.kernel.ports import LifecycleState


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


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a local model artifact or provider."""
    model_name: str
    backend: str  # "ollama", "llama", "custom"
    context_window: int = 4096
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    format: Optional[str] = None
    family: Optional[str] = None


@dataclass(frozen=True)
class ProcessorHealth:
    """Live health and resource reporting for a local processor."""
    name: str
    state: LifecycleState
    loaded_model: Optional[str] = None
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    last_active_timestamp: Optional[float] = None
    available_backends: tuple[str, ...] = ()


@runtime_checkable
class LocalTextEngine(Protocol):
    """
    Async lifecycle-aware engine protocol for local text generation.
    Implementations must run heavy compute outside the asyncio event loop.
    """

    @property
    def model_name(self) -> str:
        """The identifier or file path of the underlying model."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether the model weights are currently resident in memory."""
        ...

    async def load(self) -> Result[None, Exception]:
        """Loads model weights into memory/VRAM."""
        ...

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Asynchronously generates text without blocking the main event loop."""
        ...

    async def unload(self) -> Result[None, Exception]:
        """Frees model weights and resources."""
        ...


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
        ...

    def stream(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> AsyncIterator[Result[str, Exception]]:
        ...

    def unload(self) -> None:
        ...
