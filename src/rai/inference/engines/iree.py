"""
IREE implementation of InferenceEngine (frozen pending conformance tests).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import List, Optional

from returns.result import Failure, Result

from ..protocols import InferenceEngine, InferenceResult


def is_iree_available() -> bool:
    """IREE runtime is frozen in Stage 4 until conformance tests and owners exist."""
    return False


class IreeEngine(InferenceEngine):
    """
    Stub for IREE runtime.
    Frozen per ROADMAP.md Stage 4 WP 4.1 until owners and conformance tests exist.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)

    @property
    def model_name(self) -> str:
        return self.model_path

    @property
    def is_loaded(self) -> bool:
        return False

    async def load(self) -> Result[None, Exception]:
        return Failure(NotImplementedError("IREE runtime is frozen in Stage 4"))

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        return Failure(NotImplementedError("IREE runtime is frozen in Stage 4"))

    async def generate_async(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        return Failure(NotImplementedError("IREE runtime is frozen in Stage 4"))

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[Result[str, Exception]]:
        async def _fail() -> AsyncIterator[Result[str, Exception]]:
            yield Failure(NotImplementedError("IREE runtime is frozen in Stage 4"))

        return _fail()

    def unload(self) -> None:
        pass

