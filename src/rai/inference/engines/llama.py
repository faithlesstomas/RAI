"""
Llama.cpp implementation of LocalTextEngine and InferenceEngine.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import functools
import importlib.util
from pathlib import Path
import time
from typing import Any, List, Optional

from returns.result import Failure, Result, Success, safe

from ..protocols import GenerationStats, InferenceEngine, InferenceResult, LocalTextEngine


def is_llama_cpp_available() -> bool:
    """Check if llama-cpp-python is installed without raising an ImportError."""
    return importlib.util.find_spec("llama_cpp") is not None


class LlamaCppEngine:
    """
    Non-blocking adapter for llama-cpp-python.
    Satisfies LocalTextEngine and offloads blocking generation to worker threads.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.llm: Any = None

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    async def load(self) -> Result[None, Exception]:
        """Loads the model in a worker thread to avoid blocking the event loop."""
        if self.is_loaded:
            return Success(None)

        if not is_llama_cpp_available():
            return Failure(
                ImportError(
                    "llama-cpp-python is not installed. Install with: uv sync --extra inference-llama"
                )
            )

        def _load_sync() -> Any:
            from llama_cpp import Llama  # noqa: PLC0415
            return Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
            )

        try:
            self.llm = await asyncio.to_thread(_load_sync)
            return Success(None)
        except Exception as exc:
            return Failure(exc)

    def generate_sync(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Synchronous generation for low-level or test execution."""
        if not self.is_loaded:
            if not is_llama_cpp_available():
                return Failure(
                    ImportError(
                        "llama-cpp-python is not installed. Install with: uv sync --extra inference-llama"
                    )
                )
            from llama_cpp import Llama  # noqa: PLC0415
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    verbose=self.verbose,
                )
            except Exception as exc:
                return Failure(exc)

        start_time = time.monotonic()
        try:
            output = self.llm.create_completion(
                prompt=prompt,
                stop=stop or [],
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            duration = time.monotonic() - start_time
            text = output["choices"][0]["text"]
            usage = output.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            tps = completion_tokens / duration if duration > 0 else 0.0

            stats = GenerationStats(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_time_sec=duration,
                tokens_per_sec=tps,
            )
            return Success(
                InferenceResult(
                    text=text,
                    stats=stats,
                    finish_reason=output["choices"][0].get("finish_reason", "stop"),
                )
            )
        except Exception as exc:
            return Failure(exc)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Asynchronously generates text offloaded to a worker thread (non-blocking)."""
        load_res = await self.load()
        if isinstance(load_res, Failure):
            return load_res

        return await asyncio.to_thread(
            self.generate_sync,
            prompt,
            stop,
            max_tokens,
            temperature,
        )

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[Result[str, Exception]]:
        """Streams generated tokens without blocking the event loop between tokens."""
        load_res = await self.load()
        if isinstance(load_res, Failure):
            yield Failure(load_res.failure())
            return

        def _create_stream() -> Any:
            return self.llm.create_completion(
                prompt=prompt,
                stop=stop or [],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        try:
            stream_gen = await asyncio.to_thread(_create_stream)
            while True:
                chunk = await asyncio.to_thread(lambda: next(stream_gen, None))
                if chunk is None:
                    break
                text = chunk["choices"][0]["text"]
                yield Success(text)
        except Exception as exc:
            yield Failure(exc)

    async def unload(self) -> Result[None, Exception]:
        """Unload the model and free resources."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None
        return Success(None)

