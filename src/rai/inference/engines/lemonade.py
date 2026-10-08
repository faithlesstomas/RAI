"""
Lemonade implementation of LocalTextEngine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from returns.result import Failure, Result, Success

from rai.diagnostics import trace, trace_payload
from rai.inference.cot_utils import extract_reasoning_and_content
from rai.inference.protocols import GenerationStats, InferenceResult, LocalTextEngine

logger = logging.getLogger(__name__)

DEFAULT_LEMONADE_HOST = "http://127.0.0.1:13305"
DEFAULT_LEMONADE_URL = f"{DEFAULT_LEMONADE_HOST}/api/v1"
DEFAULT_LEMONADE_TIMEOUT = 300.0
_HTTP_OK: int = 200


async def resolve_lemonade_model(
    base_url: str = DEFAULT_LEMONADE_URL,
    explicit_model: Optional[str] = None,
) -> str:
    """Dynamically resolve the active model on Lemonade daemon, never hardcoding."""
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("RAI_LEMONADE_WORKER_MODEL") or os.environ.get("LEMONADE_MODEL")
    if env_model:
        return env_model
    try:
        host = base_url.replace("/api/v1", "").rstrip("/")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{host}/api/v1/models")
            if resp.status_code == 200:  # noqa: PLR2004
                data = resp.json()
                models = data.get("data", [])
                if models and isinstance(models, list):
                    first_model = models[0].get("id") or models[0].get("checkpoint")
                    if first_model:
                        return str(first_model)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to query Lemonade for active models: %s", exc)
    return "default"


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
        timeout_seconds: float = DEFAULT_LEMONADE_TIMEOUT,
    ) -> None:
        self._model_name = model_name
        self._host = host.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self.model_artifact_version: Optional[str] = None

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        normalized = value.rstrip("/")
        if self._host != normalized:
            self._host = normalized
            if self._client and not self._client.is_closed:
                self._client = None

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        if self._api_key != value:
            self._api_key = value
            if self._client and not self._client.is_closed:
                self._client = None

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: float) -> None:
        if self._timeout_seconds != value:
            self._timeout_seconds = value
            if self._client and not self._client.is_closed:
                self._client = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._host,
                headers=self._get_headers(),
                timeout=httpx.Timeout(self._timeout_seconds, connect=5.0),
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
                if models_resp.status_code == _HTTP_OK:
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
        except httpx.TimeoutException:
            self._is_loaded = False
            return Failure(
                TimeoutError(
                    f"Lemonade load request timed out after {self._timeout_seconds:.1f}s for model {self._model_name} on {self._host}"
                )
            )
        except httpx.ConnectError as exc:
            self._is_loaded = False
            return Failure(
                ConnectionError(f"Could not connect to Lemonade server at {self._host}: {exc}")
            )
        except Exception as exc:  # noqa: BLE001
            self._is_loaded = False
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Failed to load model {self._model_name}: {msg}"))

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

    async def generate(  # noqa: PLR0912, PLR0913, PLR0915
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

            # Fallback assistant prefix if thinking is explicitly disabled
            if not enable_thinking or thinking_budget == 0:
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
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking and (thinking_budget is None or thinking_budget > 0))},
            }
            if stop:
                chat_payload["stop"] = stop

            trace_payload(
                "lemonade.chat.request",
                chat_payload,
                model=self._model_name,
                messages_count=len(chat_messages),
            )
            response = await client.post("/api/v1/chat/completions", json=chat_payload)
            if response.status_code == _HTTP_OK:
                data = response.json()
                trace_payload("lemonade.chat.response", data, model=self._model_name)
                choices = data.get("choices", [])
                first_choice = choices[0] if choices else {}
                msg = first_choice.get("message", {})
                raw_text = msg.get("content", "") or ""
                explicit_reasoning = msg.get("reasoning_content")
                finish_reason = first_choice.get("finish_reason", "stop")
            else:
                trace_payload(
                    "lemonade.chat.error",
                    {"status_code": response.status_code, "body": response.text},
                    model=self._model_name,
                )
                # Fallback to plain text completions endpoint
                payload: dict[str, Any] = {
                    "model": self._model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if stop:
                    payload["stop"] = stop
                trace_payload("lemonade.completions.request", payload, model=self._model_name)
                response = await client.post("/api/v1/completions", json=payload)
                if response.status_code != _HTTP_OK:
                    trace_payload(
                        "lemonade.completions.error",
                        {"status_code": response.status_code, "body": response.text},
                        model=self._model_name,
                    )
                    return Failure(
                        RuntimeError(
                            f"Lemonade completion error {response.status_code}: {response.text}"
                        )
                    )
                data = response.json()
                trace_payload("lemonade.completions.response", data, model=self._model_name)
                choices = data.get("choices", [])
                first_choice = choices[0] if choices else {}
                raw_text = first_choice.get("text", "")
                explicit_reasoning = None
                finish_reason = first_choice.get("finish_reason", "stop")

            duration = time.monotonic() - start_time
            self._is_loaded = True

            clean_text, reasoning = extract_reasoning_and_content(
                raw_text, explicit_reasoning=explicit_reasoning
            )
            # If model produced no clean text but had reasoning and thinking was suppressed, use reasoning as text
            if not clean_text and reasoning and not enable_thinking:
                clean_text = reasoning
                reasoning = None

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
            completion_tokens = usage.get("completion_tokens", len(clean_text.split()))
            tps = completion_tokens / duration if duration > 0 else 0.0

            trace(
                "lemonade.generation.stats",
                model=self._model_name,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                duration_sec=round(duration, 3),
                tps=round(tps, 1),
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
        except httpx.TimeoutException:
            return Failure(
                TimeoutError(
                    f"Lemonade request timed out after {self._timeout_seconds:.1f}s while waiting for response from {self._host}"
                )
            )
        except httpx.ConnectError as exc:
            return Failure(
                ConnectionError(f"Could not connect to Lemonade server at {self._host}: {exc}")
            )
        except httpx.HTTPError as exc:
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade HTTP error: {msg}"))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade generation failed: {msg}"))

    async def generate_multimodal(  # noqa: PLR0913
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
            if response.status_code != _HTTP_OK:
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
        except httpx.TimeoutException:
            return Failure(
                TimeoutError(
                    f"Lemonade request timed out after {self._timeout_seconds:.1f}s while waiting for multimodal response from {self._host}"
                )
            )
        except httpx.ConnectError as exc:
            return Failure(
                ConnectionError(f"Could not connect to Lemonade server at {self._host}: {exc}")
            )
        except httpx.HTTPError as exc:
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade multimodal HTTP error: {msg}"))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade multimodal generation failed: {msg}"))

    async def generate_image(  # noqa: PLR0911
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
            if resp.status_code != _HTTP_OK:
                return Failure(
                    RuntimeError(f"Lemonade image generation error {resp.status_code}: {resp.text}")
                )
            data = resp.json().get("data", [])
            if data and "b64_json" in data[0]:
                image_bytes = base64.b64decode(data[0]["b64_json"])
                return Success(image_bytes)
            return Failure(RuntimeError("No image data returned from Lemonade"))
        except httpx.TimeoutException:
            return Failure(
                TimeoutError(
                    f"Lemonade request timed out after {self._timeout_seconds:.1f}s while waiting for image response from {self._host}"
                )
            )
        except httpx.ConnectError as exc:
            return Failure(
                ConnectionError(f"Could not connect to Lemonade server at {self._host}: {exc}")
            )
        except httpx.HTTPError as exc:
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade image HTTP error: {msg}"))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip() or type(exc).__name__
            return Failure(RuntimeError(f"Lemonade image generation failed: {msg}"))

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
