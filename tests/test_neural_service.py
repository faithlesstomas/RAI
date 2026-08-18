"""Neural sidecar service lifecycle and failure tests."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from rai.neural.artifacts import LensRegistry
from rai.neural.contracts import (
    Concept,
    EventType,
    GenerationRequest,
    NeuralObservation,
    NcsiErrorCode,
    NcsiRuntimeError,
)
from rai.neural.engines.base import EngineDelta, ModelDescription
from rai.neural.server import create_neural_app
from rai.neural.service import NeuralService


class FakeEngine:
    """Deterministic framework-free engine for protocol integration tests."""

    def __init__(self, *, delay: float = 0.0, failure: NcsiRuntimeError | None = None) -> None:
        self.delay = delay
        self.failure = failure
        self.unloaded = False

    @property
    def description(self) -> ModelDescription:
        return ModelDescription("fixture-model", "revision", "tokenizer", True, "cpu", "float32")

    async def stream(
        self,
        request: GenerationRequest,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[EngineDelta]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if cancel_event.is_set():
            raise NcsiRuntimeError(NcsiErrorCode.CANCELLED, "generation was cancelled")
        if self.failure:
            raise self.failure
        observation = NeuralObservation(
            request_id=request.request_id,
            forward_pass_id="forward",
            model_id="fixture-model",
            model_revision="revision",
            tokenizer_revision="tokenizer",
            lens_id=request.lens_id or "fixture-lens",
            lens_revision="lens-revision",
            layer=1,
            position=0,
            concepts=(Concept(7, "hello", 0.9),),
            readout_method="jlens-sparse",
            parameters={"top-k": 1},
            reconstruction_error=0.1,
            timestamp=time.time(),
        )
        yield EngineDelta(7, "hello", (observation,) if request.lens_id else ())

    async def unload(self) -> None:
        self.unloaded = True


@pytest.mark.asyncio
async def test_generation_emits_valid_complete_lifecycle() -> None:
    service = NeuralService(FakeEngine())
    request = GenerationRequest(
        prompt="test", request_id="request", lens_id="fixture-lens", max_new_tokens=1
    )

    events = [event async for event in service.generate(request)]

    assert [event.event_type for event in events] == [
        EventType.GENERATION_STARTED,
        EventType.TOKEN_DELTA,
        EventType.NEURAL_STATE_OBSERVED,
        EventType.GENERATION_COMPLETED,
    ]
    assert events[-1].payload == {"final-text": "hello", "token-count": 1}
    assert service.telemetry()["completed-count"] == 1


@pytest.mark.asyncio
async def test_timeout_is_a_typed_terminal_event() -> None:
    service = NeuralService(FakeEngine(delay=0.05))
    request = GenerationRequest(
        prompt="test", request_id="timeout", timeout_seconds=0.001
    )

    events = [event async for event in service.generate(request)]

    assert events[-1].event_type is EventType.GENERATION_FAILED
    assert events[-1].payload["error-code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_cancellation_is_a_typed_terminal_event() -> None:
    service = NeuralService(FakeEngine(delay=0.02))
    request = GenerationRequest(prompt="test", request_id="cancel")

    async def collect() -> list[object]:
        return [event async for event in service.generate(request)]

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    assert await service.cancel("cancel")
    events = await task

    assert events[-1].payload["error-code"] == "CANCELLED"
    assert not await service.cancel("missing")


@pytest.mark.asyncio
async def test_concurrency_limit_fails_explicitly() -> None:
    service = NeuralService(FakeEngine(delay=0.02), max_concurrency=1)

    async def collect(request_id: str) -> list[object]:
        request = GenerationRequest(prompt="test", request_id=request_id)
        return [event async for event in service.generate(request)]

    first_task = asyncio.create_task(collect("first"))
    await asyncio.sleep(0)
    second = await collect("second")
    first = await first_task

    assert first[-1].event_type is EventType.GENERATION_COMPLETED
    assert second[-1].payload["error-code"] == "CONCURRENCY_LIMIT"


@pytest.mark.asyncio
async def test_request_id_reservation_rejects_duplicate() -> None:
    service = NeuralService(FakeEngine())
    assert await service.reserve("reserved")
    assert not await service.reserve("reserved")

    request = GenerationRequest(prompt="test", request_id="reserved")
    events = [event async for event in service.generate(request, reserved=True)]

    assert events[-1].event_type is EventType.GENERATION_COMPLETED


@pytest.mark.asyncio
async def test_sidecar_http_surface_streams_ndjson(tmp_path: Path) -> None:
    engine = FakeEngine()
    app = create_neural_app(engine, LensRegistry(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        capabilities = await client.get("/api/v1/neural/capabilities")
        response = await client.post(
            "/api/v1/neural/generate",
            json={
                "request-id": "http-request",
                "prompt": "test",
                "model-id": "fixture-model",
                "max-new-tokens": 1,
            },
        )
        invalid = await client.post("/api/v1/neural/generate", json={"prompt": "missing id"})
        unloaded = await client.post("/api/v1/neural/models/unload")

    events = [json.loads(line) for line in response.text.splitlines()]
    assert capabilities.json()["schema-version"] == "gcas.ncsi.v1"
    assert capabilities.json()["intervention"] is False
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events[-1]["event-type"] == "GenerationCompleted"
    assert invalid.status_code == 422  # noqa: PLR2004
    assert unloaded.json() == {"status": "unloaded"}
    assert engine.unloaded


@pytest.mark.asyncio
async def test_unknown_model_fails_without_silent_fallback() -> None:
    service = NeuralService(FakeEngine())
    request = GenerationRequest(prompt="test", request_id="wrong", model_id="other-model")

    events = [event async for event in service.generate(request)]

    assert events[-1].payload["error-code"] == "MODEL_NOT_FOUND"


@pytest.mark.asyncio
async def test_sidecar_api_requires_shared_rai_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "sidecar-test-token")
    app = create_neural_app(FakeEngine(), LensRegistry(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        rejected = await client.get("/api/v1/neural/capabilities")
        accepted = await client.get(
            "/api/v1/neural/capabilities",
            headers={"Authorization": "Bearer sidecar-test-token"},
        )

    assert rejected.status_code == 401  # noqa: PLR2004
    assert accepted.status_code == 200  # noqa: PLR2004
