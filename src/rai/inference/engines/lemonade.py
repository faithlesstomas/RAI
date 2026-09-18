"""
Lemonade implementation of LocalTextEngine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from returns.result import Failure, Result, Success

from ..protocols import GenerationStats, InferenceResult, LocalTextEngine

logger = logging.getLogger(__name__)

DEFAULT_LEMONADE_HOST = "http://127.0.0.1:13305"


class LemonadeEngine(LocalTextEngine):
    """
    Non-blocking HTTP-backed adapter for local Lemonade daemon.
    Satisfies LocalTextEngine.
    """

    def __init__(
        self,
        model_name: str,
        host: str = DEFAULT_LEMONADE_HOST,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self.model_artifact_version: Optional[str] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
        return self._client

    async def load(self) -> Result[None, Exception]:
        """Request Lemonade server to load model weights into memory."""
        try:
            client = self._get_client()
            resp = await client.post(
                "/api/v1/load",
                json={"model_name": self._model_name},
            )
            if resp.status_code not in (200, 201):
                err_data = resp.text
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("error", err_data)
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", str(err_msg))
                except Exception:  # noqa: BLE001
                    err_msg = err_data
                self._is_loaded = False
                return Failure(RuntimeError(f"Failed to load model {self._model_name}: {err_msg}"))

            # Calculate fingerprint / artifact version from model metadata
            version_source = f"{self._model_name}:{self.host}"
            try:
                models_resp = await client.get("/api/v1/models?show_all=true")
                if models_resp.status_code == 200:
                    models_data = models_resp.json().get("data", [])
                    for m in models_data:
                        if m.get("id") == self._model_name or m.get("checkpoint") == self._model_name:
                            serialized = json.dumps(m, sort_keys=True, separators=(",", ":"))
                            version_source = serialized
                            break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to retrieve extended metadata for %s: %s", self._model_name, exc)

            self.model_artifact_version = (
                "lemonade-sha256:" + hashlib.sha256(version_source.encode("utf-8")).hexdigest()
            )
            self._is_loaded = True
            return Success(None)
        except Exception as exc:  # noqa: BLE001
            self._is_loaded = False
            return Failure(exc)

    @staticmethod
    def _parse_prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
        """Parse structured plain-text prompt into system and user messages."""
        user_match = re.search(
            r"\n(?:Użytkownik|User):\s*(.*?)(?:\n(?:Asystent|Assistant):|$)",
            prompt,
            re.DOTALL,
        )
        if user_match:
            user_content = user_match.group(1).strip()
            system_content = prompt[: user_match.start()].strip()
            if system_content.startswith("System:"):
                system_content = system_content[len("System:") :].strip()
            messages: list[dict[str, Any]] = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})
            return messages
        return [{"role": "user", "content": prompt}]

    async def generate(
        self,
        prompt: str = "",
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Result[InferenceResult, Exception]:
        """Asynchronously generates text from Lemonade server."""
        start_time = time.monotonic()
        try:
            client = self._get_client()
            if messages:
                chat_messages = [dict(m) for m in messages]
            elif prompt:
                chat_messages = self._parse_prompt_to_messages(prompt)
            else:
                chat_messages = []

            # For models with reasoning tokens (like Qwen 3.5), close thinking phase immediately
            # with an assistant prefix so the model does not exhaust output token budget inside <think>.
            is_thinking_model = any(
                keyword in self._model_name.lower()
                for keyword in ("qwen3.5", "qwen-3.5", "deepseek", "r1")
            )
            if is_thinking_model:
                chat_messages.append(
                    {"role": "assistant", "content": "<think>\n</think>\n", "prefix": True}
                )

            chat_payload: dict[str, Any] = {
                "model": self._model_name,
                "messages": chat_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if stop:
                chat_payload["stop"] = stop

            response = await client.post("/api/v1/chat/completions", json=chat_payload)
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                first_choice = choices[0] if choices else {}
                raw_text = first_choice.get("message", {}).get("content", "")
                text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                finish_reason = first_choice.get("finish_reason", "stop")
            else:
                # Fallback to plain text completions endpoint
                payload: dict[str, Any] = {
                    "model": self._model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if stop:
                    payload["stop"] = stop
                response = await client.post("/api/v1/completions", json=payload)
                if response.status_code != 200:
                    return Failure(
                        RuntimeError(
                            f"Lemonade completion error {response.status_code}: {response.text}"
                        )
                    )
                data = response.json()
                choices = data.get("choices", [])
                first_choice = choices[0] if choices else {}
                raw_text = first_choice.get("text", "")
                text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                finish_reason = first_choice.get("finish_reason", "stop")

            duration = time.monotonic() - start_time
            self._is_loaded = True

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
            completion_tokens = usage.get("completion_tokens", len(text.split()))
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
                    finish_reason=finish_reason or "stop",
                )
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(exc)

    async def generate_multimodal(
        self,
        prompt: str,
        image_base64: str,
        image_format: str = "png",
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        """Asynchronously generates text from multimodal input (e.g. Qwen-VL) via Lemonade."""
        start_time = time.monotonic()
        try:
            client = self._get_client()
            content: list[dict[str, Any]] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
                },
            ]
            payload: dict[str, Any] = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if stop:
                payload["stop"] = stop

            response = await client.post("/api/v1/chat/completions", json=payload)
            if response.status_code != 200:
                return Failure(
                    RuntimeError(
                        f"Lemonade multimodal completion error {response.status_code}: {response.text}"
                    )
                )
            data = response.json()
            choices = data.get("choices", [])
            first_choice = choices[0] if choices else {}
            text = first_choice.get("message", {}).get("content", "")
            finish_reason = first_choice.get("finish_reason", "stop")

            duration = time.monotonic() - start_time
            self._is_loaded = True

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
            completion_tokens = usage.get("completion_tokens", len(text.split()))
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
                    finish_reason=finish_reason or "stop",
                )
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(exc)

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
    ) -> Result[bytes, Exception]:
        """Asynchronously generates an image using Lemonade's sd-cpp/Flux endpoint."""
        import base64  # noqa: PLC0415

        try:
            client = self._get_client()
            payload = {
                "model": model or self._model_name,
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "response_format": "b64_json",
            }
            resp = await client.post("/v1/images/generations", json=payload)
            if resp.status_code != 200:
                return Failure(
                    RuntimeError(f"Lemonade image generation error {resp.status_code}: {resp.text}")
                )
            data = resp.json().get("data", [])
            if data and "b64_json" in data[0]:
                image_bytes = base64.b64decode(data[0]["b64_json"])
                return Success(image_bytes)
            return Failure(RuntimeError("No image data returned from Lemonade"))
        except Exception as exc:  # noqa: BLE001
            return Failure(exc)

    async def unload(self) -> Result[None, Exception]:
        """Request Lemonade to unload model from memory."""
        try:
            client = self._get_client()
            resp = await client.post(
                "/api/v1/unload",
                json={"model_name": self._model_name},
            )
            # Accept 200 or 404 (if already unloaded)
            if resp.status_code in (200, 404):
                self._is_loaded = False
                return Success(None)
            return Failure(RuntimeError(f"Failed to unload model {self._model_name}: {resp.text}"))
        except Exception as exc:  # noqa: BLE001
            self._is_loaded = False
            return Failure(exc)

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to close Lemonade HTTP client: %s", exc)
            self._client = None
