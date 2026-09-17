"""
Ollama implementation of LocalTextEngine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, List, Optional

from returns.result import Failure, Result, Success

from ..protocols import GenerationStats, InferenceResult, LocalTextEngine

logger = logging.getLogger(__name__)


class OllamaEngine:
    """
    Non-blocking HTTP-backed adapter for local Ollama daemon.
    Satisfies LocalTextEngine.
    """

    def __init__(
        self,
        model_name: str,
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        self._model_name = model_name
        self.host = host
        self._client: Any = None
        self._is_loaded = False
        self.model_artifact_version: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama  # noqa: PLC0415

            self._client = ollama.AsyncClient(host=self.host)
        return self._client

    async def load(self) -> Result[None, Exception]:
        """Verify model availability in Ollama."""
        try:
            client = self._get_client()
            # Test model availability via show
            model_description = await client.show(model=self._model_name)
            if hasattr(model_description, "model_dump"):
                model_description = model_description.model_dump(mode="json")
            serialized = json.dumps(
                model_description,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            self.model_artifact_version = (
                "ollama-show-sha256:" + hashlib.sha256(serialized).hexdigest()
            )
            self._is_loaded = True
            return Success(None)
        except Exception as exc:
            self._is_loaded = False
            return Failure(exc)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Asynchronously generates text from Ollama."""
        start_time = time.monotonic()
        try:
            client = self._get_client()
            options: dict[str, Any] = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            if stop:
                options["stop"] = stop

            response = await client.generate(
                model=self._model_name,
                prompt=prompt,
                # Reasoning-only output is not a user-visible assistant answer. Small
                # reasoning models can otherwise consume the entire token budget in
                # the hidden `thinking` field and return an empty `response`.
                think=False,
                options=options,
            )
            duration = time.monotonic() - start_time
            self._is_loaded = True

            text = response.get("response", "")
            eval_count = response.get("eval_count", 0)
            prompt_eval_count = response.get("prompt_eval_count", 0)
            tps = eval_count / duration if duration > 0 else 0.0

            stats = GenerationStats(
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                total_time_sec=duration,
                tokens_per_sec=tps,
            )
            return Success(
                InferenceResult(
                    text=text,
                    stats=stats,
                    finish_reason="stop" if response.get("done", True) else "length",
                )
            )
        except Exception as exc:
            return Failure(exc)

    async def unload(self) -> Result[None, Exception]:
        """Request Ollama to unload model from memory using keep_alive=0."""
        try:
            client = self._get_client()
            # In Ollama API, keep_alive=0 tells the runner to immediately unload the model
            await client.generate(
                model=self._model_name,
                prompt="",
                keep_alive=0,
            )
            self._is_loaded = False
            return Success(None)
        except Exception as exc:
            self._is_loaded = False
            return Failure(exc)

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and hasattr(self._client, "_client"):
            try:
                await self._client._client.aclose()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to close Ollama HTTP client: %s", exc)
            self._client = None
