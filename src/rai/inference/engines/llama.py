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

    @property
    def required_ram_bytes(self) -> Optional[int]:
        """Return the GGUF size as a conservative lower bound for host RAM."""
        try:
            return Path(self.model_path).stat().st_size
        except OSError:
            return None

    @property
    def required_vram_bytes(self) -> int:
        """The default adapter does not request GPU offload explicitly."""
        return 0

    def load(self) -> Result[None, Exception]:
        """Synchronously loads model weights."""
        if self.is_loaded:
            return Success(None)

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
            return Success(None)
        except Exception as exc:
            return Failure(exc)

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Synchronously generates text from a prompt satisfying InferenceEngine."""
        if not self.is_loaded:
            load_res = self.load()
            if isinstance(load_res, Failure):
                return load_res

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

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[Result[str, Exception]]:
        """Streams generated tokens without blocking the event loop between tokens."""
        if not self.is_loaded:
            load_res = self.load()
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

    def unload(self) -> None:
        """Synchronously unloads model weights and frees resources."""
        if hasattr(self, "llm") and self.llm is not None:
            del self.llm
            self.llm = None


class AsyncLlamaEngine(LocalTextEngine):
    """
    Asynchronous LocalTextEngine adapter for LlamaCppEngine.
    Offloads synchronous model loading and generation to worker threads via asyncio.to_thread.
    """

    def __init__(
        self,
        engine_or_path: Any,
        n_ctx: int = 0,
        verbose: bool = False,
    ) -> None:
        if isinstance(engine_or_path, LlamaCppEngine):
            self._engine = engine_or_path
        else:
            self._engine = LlamaCppEngine(
                model_path=str(engine_or_path),
                n_ctx=n_ctx,
                verbose=verbose,
            )

    @property
    def model_name(self) -> str:
        return self._engine.model_name

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def required_ram_bytes(self) -> Optional[int]:
        return self._engine.required_ram_bytes

    @property
    def required_vram_bytes(self) -> int:
        return self._engine.required_vram_bytes

    async def load(self) -> Result[None, Exception]:
        return await asyncio.to_thread(self._engine.load)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        if not self.is_loaded:
            load_res = await self.load()
            if isinstance(load_res, Failure):
                return load_res

        return await asyncio.to_thread(
            self._engine.generate,
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
        async for chunk in self._engine.stream(prompt, stop, max_tokens, temperature):
            yield chunk

    async def unload(self) -> Result[None, Exception]:
        await asyncio.to_thread(self._engine.unload)
        return Success(None)

