"""Framework-neutral engine contracts used by the sidecar service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from ..contracts import GenerationRequest, NeuralObservation


@dataclass(frozen=True)
class ModelDescription:
    """Public model discovery and exact revision metadata."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    loaded: bool
    device: str
    dtype: str

    def to_wire(self) -> dict[str, object]:
        return {
            "model-id": self.model_id,
            "model-revision": self.model_revision,
            "tokenizer-revision": self.tokenizer_revision,
            "loaded": self.loaded,
            "device": self.device,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class EngineDelta:
    """One generated token and any compact observations from its forward pass."""

    token_id: int | str
    token_text: str
    observations: tuple[NeuralObservation, ...] = ()


class GenerationEngine(Protocol):
    """Long-lived model owner used exclusively by the neural process."""

    @property
    def description(self) -> ModelDescription: ...

    async def stream(
        self,
        request: GenerationRequest,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[EngineDelta]: ...

    async def unload(self) -> None: ...
