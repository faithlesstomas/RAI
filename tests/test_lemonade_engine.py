"""
Tests for LemonadeEngine and its integration with inference factory and supervisor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from returns.result import Failure, Success

from rai.inference.engines.lemonade import DEFAULT_LEMONADE_HOST, LemonadeEngine
from rai.inference.factory import (
    get_available_backends,
    is_backend_available,
    is_lemonade_available,
    load_local_model,
)
from rai.inference.protocols import GenerationStats, InferenceResult, LocalTextEngine
from rai.inference.supervisor import ProcessorSupervisor


@pytest.mark.asyncio
async def test_lemonade_engine_properties() -> None:
    engine = LemonadeEngine(
        model_name="Qwen3.5-2B-GGUF",
        host="http://localhost:13305/",
        api_key="secret-token",
        timeout_seconds=45.0,
    )
    assert engine.model_name == "Qwen3.5-2B-GGUF"
    assert engine.host == "http://localhost:13305"
    assert engine.api_key == "secret-token"
    assert engine.timeout_seconds == 45.0
    assert not engine.is_loaded
    assert engine.model_artifact_version is None

    headers = engine._get_headers()
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_lemonade_engine_load_success() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    load_resp = httpx.Response(
        200,
        json={"status": "success", "message": "Model loaded"},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/load"),
    )
    models_resp = httpx.Response(
        200,
        json={"data": [{"id": "Qwen3.5-2B-GGUF", "recipe": "llamacpp", "size": 1.8}]},
        request=httpx.Request(
            "GET", f"{DEFAULT_LEMONADE_HOST}/api/v1/models?show_all=true"
        ),
    )

    mock_client.post.return_value = load_resp
    mock_client.get.return_value = models_resp

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.load()
        assert isinstance(res, Success)
        assert engine.is_loaded
        assert engine.model_artifact_version is not None
        assert engine.model_artifact_version.startswith("lemonade-sha256:")

        mock_client.post.assert_called_once_with(
            "/api/v1/load",
            json={"model_name": "Qwen3.5-2B-GGUF"},
        )


@pytest.mark.asyncio
async def test_lemonade_engine_load_failure() -> None:
    engine = LemonadeEngine(model_name="Missing-Model")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    error_resp = httpx.Response(
        404,
        json={"error": {"message": "Model 'Missing-Model' was not found."}},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/load"),
    )
    mock_client.post.return_value = error_resp

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.load()
        assert isinstance(res, Failure)
        assert not engine.is_loaded
        assert "Missing-Model" in str(res.failure())


@pytest.mark.asyncio
async def test_lemonade_engine_generate_chat_success() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    chat_resp = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello from Lemonade!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
        request=httpx.Request(
            "POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/chat/completions"
        ),
    )
    mock_client.post.return_value = chat_resp

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.generate(
            prompt="Hello there",
            stop=["\n"],
            max_tokens=64,
            temperature=0.3,
        )
        assert isinstance(res, Success)
        result: InferenceResult = res.unwrap()
        assert result.text == "Hello from Lemonade!"
        assert result.finish_reason == "stop"
        assert result.stats is not None
        assert result.stats.input_tokens == 8
        assert result.stats.output_tokens == 4
        assert result.stats.total_time_sec >= 0.0
        assert engine.is_loaded

        mock_client.post.assert_called_once_with(
            "/api/v1/chat/completions",
            json={
                "model": "Qwen3.5-2B-GGUF",
                "messages": [
                    {"role": "user", "content": "Hello there"},
                    {
                        "role": "assistant",
                        "content": "<think>\n</think>\n",
                        "prefix": True,
                    },
                ],
                "max_tokens": 64,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": ["\n"],
            },
        )


@pytest.mark.asyncio
async def test_lemonade_engine_preserves_structured_messages() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "Pamiętam."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        },
        request=httpx.Request(
            "POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/chat/completions"
        ),
    )
    messages = [
        {"role": "system", "content": "Pomagaj zwięźle."},
        {"role": "user", "content": "Pamiętasz?"},
    ]

    with patch.object(engine, "_get_client", return_value=mock_client):
        result = await engine.generate(messages=messages)

    assert isinstance(result, Success)
    payload = mock_client.post.await_args.kwargs["json"]
    assert payload["messages"][:2] == messages


@pytest.mark.asyncio
async def test_lemonade_engine_generate_completions_fallback() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    # 404 on chat completions, 200 on text completions fallback
    chat_404 = httpx.Response(
        404,
        text="Not found",
        request=httpx.Request(
            "POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/chat/completions"
        ),
    )
    comp_200 = httpx.Response(
        200,
        json={
            "choices": [
                {"text": "Completions fallback response", "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/completions"),
    )
    mock_client.post.side_effect = [chat_404, comp_200]

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.generate(prompt="Explain quantum computing")
        assert isinstance(res, Success)
        result = res.unwrap()
        assert result.text == "Completions fallback response"
        assert result.stats is not None
        assert result.stats.output_tokens == 5


@pytest.mark.asyncio
async def test_lemonade_engine_generate_multimodal() -> None:
    engine = LemonadeEngine(model_name="Qwen2.5-VL-7B-Instruct-GGUF")
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    chat_200 = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I see a cat in the image.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 7},
        },
        request=httpx.Request(
            "POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/chat/completions"
        ),
    )
    mock_client.post.return_value = chat_200

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.generate_multimodal(
            prompt="What is in this image?",
            image_base64="aGVsbG8=",
            image_format="jpeg",
        )
        assert isinstance(res, Success)
        result = res.unwrap()
        assert result.text == "I see a cat in the image."
        assert result.stats.input_tokens == 15
        assert result.stats.output_tokens == 7


@pytest.mark.asyncio
async def test_lemonade_engine_generate_image() -> None:
    import base64

    engine = LemonadeEngine(model_name="SD-Turbo")
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b64_str = base64.b64encode(raw_bytes).decode("ascii")

    img_200 = httpx.Response(
        200,
        json={"data": [{"b64_json": b64_str}]},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/v1/images/generations"),
    )
    mock_client.post.return_value = img_200

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.generate_image(prompt="A beautiful sunset")
        assert isinstance(res, Success)
        assert res.unwrap() == raw_bytes


@pytest.mark.asyncio
async def test_lemonade_engine_unload() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")
    engine._is_loaded = True

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    unload_resp = httpx.Response(
        200,
        json={"message": "Model unloaded successfully", "status": "success"},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/unload"),
    )
    mock_client.post.return_value = unload_resp

    with patch.object(engine, "_get_client", return_value=mock_client):
        res = await engine.unload()
        assert isinstance(res, Success)
        assert not engine.is_loaded

        mock_client.post.assert_called_once_with(
            "/api/v1/unload",
            json={"model_name": "Qwen3.5-2B-GGUF"},
        )


@pytest.mark.asyncio
async def test_is_lemonade_available() -> None:
    with patch("httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(
            200,
            json={"status": "ok", "version": "10.6.0"},
            request=httpx.Request("GET", "http://127.0.0.1:13305/api/v1/health"),
        )
        assert is_lemonade_available() is True
        assert is_backend_available("lemonade") is True

    with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
        assert is_lemonade_available() is False
        assert is_backend_available("lemonade") is False


def test_factory_lemonade_load() -> None:
    res = load_local_model("Qwen3.5-2B-GGUF", backend="lemonade")
    assert isinstance(res, Success)
    engine = res.unwrap()
    assert isinstance(engine, LemonadeEngine)
    assert engine.model_name == "Qwen3.5-2B-GGUF"


@pytest.mark.asyncio
async def test_supervisor_integration_with_lemonade() -> None:
    engine = LemonadeEngine(model_name="Qwen3.5-2B-GGUF")
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    load_resp = httpx.Response(
        200,
        json={"status": "success"},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/load"),
    )
    models_resp = httpx.Response(
        200,
        json={"data": [{"id": "Qwen3.5-2B-GGUF"}]},
        request=httpx.Request(
            "GET", f"{DEFAULT_LEMONADE_HOST}/api/v1/models?show_all=true"
        ),
    )
    comp_resp = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "text": '{"intent": "query", "confidence": 0.95}',
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        },
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/completions"),
    )
    unload_resp = httpx.Response(
        200,
        json={"status": "success"},
        request=httpx.Request("POST", f"{DEFAULT_LEMONADE_HOST}/api/v1/unload"),
    )

    mock_client.post.side_effect = [load_resp, comp_resp, unload_resp]
    mock_client.get.return_value = models_resp

    with (
        patch.object(engine, "_get_client", return_value=mock_client),
        patch(
            "rai.inference.supervisor.get_available_backends",
            return_value=("lemonade", "ollama"),
        ),
    ):
        supervisor = ProcessorSupervisor(
            engine=engine,
            model_name="Qwen3.5-2B-GGUF",
            backend="lemonade",
        )
        start_res = await supervisor.start()
        assert isinstance(start_res, Success)

        health = supervisor.health()
        assert health.name == "local-processor"
        assert health.loaded_model is None
        assert "lemonade" in health.available_backends

        await supervisor.stop()


def test_assistant_runtime_lemonade_config() -> None:
    from rai.assistant.runtime import build_assistant_backend, resolve_assistant_config

    resolved = resolve_assistant_config(
        {
            "assistant": {
                "backend": "lemonade",
                "model": "Qwen3.5-4B-GGUF",
                "lemonade_host": "http://192.168.1.10:13305",
                "lemonade_api_key": "custom-key",
            }
        }
    )
    assert resolved.backend == "lemonade"
    assert resolved.model == "Qwen3.5-4B-GGUF"
    assert resolved.lemonade_host == "http://192.168.1.10:13305"
    assert resolved.lemonade_api_key == "custom-key"

    backend = build_assistant_backend(resolved)
    assert backend.backend_name == "lemonade"
    assert backend.model_name == "Qwen3.5-4B-GGUF"
    assert isinstance(backend.engine, LemonadeEngine)
    assert backend.engine.host == "http://192.168.1.10:13305"
    assert backend.engine.api_key == "custom-key"
