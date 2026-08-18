"""Separately startable FastAPI/UDS server for the optional NCSI runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from rai import __version__
from rai.paths import data_dir
from rai.tools.security.auth import is_authorized

from .artifacts import LensRegistry
from .contracts import GenerationRequest, NCSI_SCHEMA_VERSION, NcsiContractError
from .engines.base import GenerationEngine
from .engines.transformers import TransformersEngine
from .service import NeuralService

NDJSON_MEDIA_TYPE = "application/x-ndjson"


class NeuralAuthenticationMiddleware:
    """Require the ordinary RAI bearer/API token for every neural API call."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/"):
            if not is_authorized(Headers(scope=scope)):
                response = JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_neural_app(
    engine: GenerationEngine,
    lens_registry: LensRegistry,
    *,
    max_concurrency: int = 1,
) -> FastAPI:
    """Build a sidecar app around an injected engine (including test engines)."""
    service = NeuralService(engine, max_concurrency=max_concurrency)
    app = FastAPI(
        title="RAI NCSI Neural Sidecar",
        version=__version__,
        description="Optional read-only Transformers/J-lens runtime for GAIA.",
    )
    app.state.neural_service = service
    app.state.lens_registry = lens_registry
    app.add_middleware(NeuralAuthenticationMiddleware)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "schema-version": NCSI_SCHEMA_VERSION}

    @app.get("/api/v1/neural/capabilities")
    async def capabilities() -> dict[str, object]:
        return {
            "schema-version": NCSI_SCHEMA_VERSION,
            "transport": "ndjson",
            "observation": True,
            "intervention": False,
            "cancellation": True,
            "raw-tensors-exposed": False,
            "max-concurrency": service.max_concurrency,
            "max-concepts": 64,
            "event-types": [
                "GenerationStarted",
                "TokenDelta",
                "NeuralStateObserved",
                "GenerationCompleted",
                "GenerationFailed",
            ],
        }

    @app.get("/api/v1/neural/models")
    async def models() -> dict[str, object]:
        return {"models": [engine.description.to_wire()]}

    @app.get("/api/v1/neural/lenses")
    async def lenses() -> dict[str, object]:
        return {"lenses": [manifest.to_public_dict() for manifest in lens_registry.manifests()]}

    @app.get("/api/v1/neural/telemetry")
    async def telemetry() -> dict[str, object]:
        return service.telemetry()

    @app.post("/api/v1/neural/generate", response_model=None)
    async def generate(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
            generation_request = GenerationRequest.from_wire(body)
        except (json.JSONDecodeError, NcsiContractError, TypeError) as exc:
            return JSONResponse(
                status_code=422,
                content={"error-code": "INVALID_REQUEST", "detail": str(exc)},
            )
        if not await service.reserve(generation_request.request_id):
            return JSONResponse(
                status_code=409,
                content={"error-code": "INVALID_REQUEST", "detail": "request-id is active"},
            )

        async def event_bytes() -> AsyncIterator[bytes]:
            async for event in service.generate(generation_request, reserved=True):
                yield (json.dumps(event.to_wire(), separators=(",", ":")) + "\n").encode()

        return StreamingResponse(event_bytes(), media_type=NDJSON_MEDIA_TYPE)

    @app.post("/api/v1/neural/requests/{request_id}/cancel")
    async def cancel(request_id: str) -> JSONResponse:
        cancelled = await service.cancel(request_id)
        if not cancelled:
            return JSONResponse(status_code=404, content={"detail": "request is not active"})
        return JSONResponse(content={"request-id": request_id, "status": "cancellation-requested"})

    @app.post("/api/v1/neural/models/unload")
    async def unload() -> JSONResponse:
        if not await service.unload():
            return JSONResponse(
                status_code=409, content={"detail": "model has active requests"}
            )
        return JSONResponse(content={"status": "unloaded"})

    return app


def app_from_environment() -> FastAPI:
    """Create the production sidecar without importing Transformers at module import."""
    model_id = os.environ.get("RAI_NEURAL_MODEL")
    model_revision = os.environ.get("RAI_NEURAL_MODEL_REVISION")
    if not model_id or not model_revision:
        raise RuntimeError("RAI_NEURAL_MODEL and RAI_NEURAL_MODEL_REVISION are required")
    registry = LensRegistry(
        Path(os.environ.get("RAI_NEURAL_LENS_DIR", data_dir() / "neural" / "lenses"))
    )
    engine = TransformersEngine(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=os.environ.get("RAI_NEURAL_TOKENIZER_REVISION"),
        lens_registry=registry,
        device=os.environ.get("RAI_NEURAL_DEVICE", "auto"),
        dtype=os.environ.get("RAI_NEURAL_DTYPE", "auto"),
        quantization=os.environ.get("RAI_NEURAL_QUANTIZATION"),
    )
    return create_neural_app(
        engine,
        registry,
        max_concurrency=int(os.environ.get("RAI_NEURAL_MAX_CONCURRENCY", "1")),
    )
