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
from typing import Any, Dict, List, Optional

from returns.result import Failure, Result, Success, safe

from ..cot_utils import extract_reasoning_and_content
from ..protocols import (
    GenerationStats,
    InferenceEngine,
    InferenceResult,
    LocalTextEngine,
)


def is_llama_cpp_available() -> bool:
    """Check if llama-cpp-python is installed without raising an ImportError."""
    return importlib.util.find_spec("llama_cpp") is not None


class ReasoningBudgetLogitsProcessor:
    """Forces emission of closing thinking tag when reasoning budget is exhausted."""

    def __init__(self, open_token_ids: set[int], close_token_id: int, budget: int) -> None:
        self.open_token_ids = open_token_ids
        self.close_token_id = close_token_id
        self.budget = budget

    def __call__(self, input_ids: Any, logits: Any) -> Any:  # noqa: ANN401
        import numpy as np  # noqa: PLC0415

        tokens = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        last_open_idx = -1
        last_close_idx = -1
        for idx, tok in enumerate(tokens):
            if tok in self.open_token_ids:
                last_open_idx = idx
            elif tok == self.close_token_id:
                last_close_idx = idx
        if last_open_idx > last_close_idx:
            tokens_in_thinking = len(tokens) - 1 - last_open_idx
            if tokens_in_thinking >= self.budget:
                forced_logits = np.full_like(logits, -1e9)
                forced_logits[self.close_token_id] = 1e9
                return forced_logits
        return logits


def _build_budget_processor(llm: Any, budget: int) -> Optional[Any]:  # noqa: ANN401
    """Construct a ReasoningBudgetLogitsProcessor using tokenizer tokens if available."""
    try:
        open_tokens: set[int] = set()
        close_token: Optional[int] = None
        for tok_bytes in (b"<think>", b"<|think|>", b"<thought>"):
            tok_ids = llm.tokenize(tok_bytes, add_bos=False)
            if len(tok_ids) == 1:
                open_tokens.add(tok_ids[0])
        for tok_bytes in (b"</think>", b"<|/think|>", b"</thought>"):
            tok_ids = llm.tokenize(tok_bytes, add_bos=False)
            if len(tok_ids) == 1:
                close_token = tok_ids[0]
                break
        if open_tokens and close_token is not None:
            return ReasoningBudgetLogitsProcessor(open_tokens, close_token, budget)
    except Exception:
        pass
    return None


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

    def generate(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        prompt: str = "",
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
    ) -> Result[InferenceResult, Exception]:
        """Synchronously generates text from a prompt or messages satisfying InferenceEngine."""
        if not self.is_loaded:
            load_res = self.load()
            if isinstance(load_res, Failure):
                return load_res

        start_time = time.monotonic()
        try:
            logits_processors: list[Any] = []
            if enable_thinking and thinking_budget is not None and thinking_budget > 0:
                budget_proc = _build_budget_processor(self.llm, thinking_budget)
                if budget_proc is not None:
                    logits_processors.append(budget_proc)

            raw_text = ""
            explicit_reasoning: Optional[str] = None
            finish_reason = "stop"

            if messages or hasattr(self.llm, "create_chat_completion"):
                chat_messages = (
                    list(messages)
                    if messages
                    else [{"role": "user", "content": prompt}]
                )
                if not enable_thinking or thinking_budget == 0:
                    # Append assistant prefix to close think block immediately for models expecting CoT
                    is_thinking_model = any(
                        keyword in self.model_name.lower()
                        for keyword in ("qwen3.5", "qwen-3.5", "deepseek", "r1")
                    )
                    if is_thinking_model:
                        chat_messages.append(
                            {"role": "assistant", "content": "<think>\n</think>\n", "prefix": True}
                        )

                chat_kwargs: dict[str, Any] = {
                    "messages": chat_messages,
                    "stop": stop or [],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if logits_processors:
                    chat_kwargs["logits_processor"] = logits_processors

                try:
                    output = self.llm.create_chat_completion(**chat_kwargs)
                    choice = output["choices"][0]
                    msg = choice.get("message", {})
                    raw_text = msg.get("content", "") or ""
                    explicit_reasoning = msg.get("reasoning_content")
                    finish_reason = choice.get("finish_reason", "stop")
                except Exception:
                    comp_kwargs: dict[str, Any] = {
                        "prompt": prompt,
                        "stop": stop or [],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "echo": False,
                    }
                    if logits_processors:
                        comp_kwargs["logits_processor"] = logits_processors
                    output = self.llm.create_completion(**comp_kwargs)
                    choice = output["choices"][0]
                    raw_text = choice.get("text", "")
                    finish_reason = choice.get("finish_reason", "stop")
            else:
                comp_kwargs = {
                    "prompt": prompt,
                    "stop": stop or [],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "echo": False,
                }
                if logits_processors:
                    comp_kwargs["logits_processor"] = logits_processors
                output = self.llm.create_completion(**comp_kwargs)
                choice = output["choices"][0]
                raw_text = choice.get("text", "")
                finish_reason = choice.get("finish_reason", "stop")

            duration = time.monotonic() - start_time
            usage = output.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            tps = completion_tokens / duration if duration > 0 else 0.0

            clean_text, reasoning = extract_reasoning_and_content(
                raw_text, explicit_reasoning=explicit_reasoning
            )

            stats = GenerationStats(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_time_sec=duration,
                tokens_per_sec=tps,
            )
            return Success(
                InferenceResult(
                    text=clean_text,
                    reasoning_content=reasoning,
                    stats=stats,
                    finish_reason=finish_reason,
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

    async def generate(  # noqa: PLR0913
        self,
        prompt: str = "",
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
    ) -> Result[InferenceResult, Exception]:
        if not self.is_loaded:
            load_res = await self.load()
            if isinstance(load_res, Failure):
                return load_res

        return await asyncio.to_thread(
            self._engine.generate,
            prompt=prompt,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
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
