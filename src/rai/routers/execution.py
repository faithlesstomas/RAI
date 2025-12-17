"""
API endpoints for executing agent chains.
"""
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Success

from .. import config_manager
from ..engine import run_chain, stream_chain
from ..dependencies import get_config, get_model_registry
from ..services.model_registry import ModelRegistry

try:
    from agno.exceptions import ModelProviderError # pylint: disable=unused-import
except ImportError:
    # Agno might not be installed or version mismatch, define dummy
    class ModelProviderError(Exception):
        """Dummy exception when agno is not installed."""

try:
    from ollama import ResponseError # pylint: disable=unused-import
except ImportError:
    # Ollama might not be installed
    class ResponseError(Exception):
        """Dummy exception when ollama is not installed."""

router = APIRouter(
    tags=["AI Engine"],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic Models ---

class ChainRequest(BaseModel):
    """Request model for running a chain of agents."""
    chain_input: str
    chain_configs: List[Dict[str, Any]]
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# --- Dependencies ---



# --- Endpoints ---

@router.post("/api/v1/run")
async def execute_chain(
        request: ChainRequest,
        app_config: Dict[str, Any] = Depends(get_config)
) -> JSONResponse:
    """
    Runs a chain of agents with the given input and configurations.
    """
    result = await run_chain(
        chain_input=request.chain_input,
        chain_configs=request.chain_configs,
        app_config=app_config
    )

    match result:
        case Success(payload):
            return JSONResponse(content={"status": "success", "payload": payload})
        case Failure(error):
            # The ChainExecutionError strings the original exception,
            # but we might want to check the original exception if possible.

            error_str = str(error)

            if "429" in error_str and (
                "Too Many Requests" in error_str or "RESOURCE_EXHAUSTED" in error_str
            ):
                logging.warning("Backend rate limit exceeded: %s", error_str)
                raise HTTPException(status_code=429, detail=error_str)

            if "404" in error_str and "not found" in error_str.lower():
                logging.error("Backend resource not found: %s", error_str)
                raise HTTPException(status_code=404, detail=error_str)

            # Check for ResponseError patterns (Ollama)
            if "ResponseError" in error_str:
                if "404" in error_str:
                    logging.error("Ollama model not found: %s", error_str)
                    raise HTTPException(status_code=404, detail=error_str)
                if "400" in error_str:
                    logging.error("Ollama bad request: %s", error_str)
                    raise HTTPException(status_code=400, detail=error_str)

            logging.error("Error during chain execution: %s", error, exc_info=error)
            raise HTTPException(status_code=500, detail=str(error))


@router.post("/api/v1/stream")
async def stream_chain_endpoint(
        request: ChainRequest,
        app_config: Dict[str, Any] = Depends(get_config)
) -> StreamingResponse:
    """
    Streams the result of a chain execution using Server-Sent Events (SSE).
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in stream_chain(
            chain_input=request.chain_input,
            chain_configs=request.chain_configs,
            session_id=request.session_id,
            app_config=app_config
        ):
            if isinstance(chunk, Failure):
                # Send error as a specific event or data
                error_msg = str(chunk.failure())
                yield f"event: error\ndata: {json.dumps({'detail': error_msg})}\n\n"
                return

            # Assuming chunk is an object with delta or content
            # We need to serialize it to JSON
            # Agno chunks might be objects, let's try to get content
            content = ""
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, str):
                content = chunk
            else:
                content = str(chunk)

            # Send data
            yield f"data: {json.dumps({'content': content})}\n\n"

        # End of stream
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/v1/models")
async def get_all_models(
    registry: ModelRegistry = Depends(get_model_registry)
) -> JSONResponse:
    """
    Returns a list of available models for all backends.
    """
    models = await registry.get_all_models()
    return JSONResponse(content={"models": models})


@router.get("/api/v1/models/{backend}")
async def get_models_for_backend(
    backend: str,
    registry: ModelRegistry = Depends(get_model_registry)
) -> JSONResponse:
    """
    Returns a list of available models for a given backend.
    """
    logging.info("Fetching models for backend: %s", backend)
    models = await registry.get_models(backend)

    # Fallback for other backends as per original behavior
    if not models and backend != "ollama":
        return JSONResponse(content={"models": ["default-model"]})

    return JSONResponse(content={"models": models})


@router.websocket("/ws/v1/chat")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handles WebSocket connections for real-time chat."""
    await websocket.accept()
    app_config = config_manager.load_config()  # Load config manually
    try:
        while True:
            data = await websocket.receive_json()
            try:
                request = ChainRequest.model_validate(data)
            except ValidationError as e:
                await websocket.send_json({"status": "error", "detail": e.errors()})
                continue

            result = await run_chain(
                chain_input=request.chain_input,
                chain_configs=request.chain_configs,
                session_id=request.session_id,
                app_config=app_config
            )

            match result:
                case Success(payload):
                    await websocket.send_json({"type": "response", "payload": payload})
                case Failure(error):
                    logging.error("Error during WebSocket chain execution: %s", error, exc_info=error)
                    await websocket.send_json({"type": "error", "detail": str(error)})

    except WebSocketDisconnect:
        logging.info("Client disconnected from WebSocket.")
    except Exception as e: # pylint: disable=broad-exception-caught
        logging.error("Unexpected error in WebSocket: %s", e, exc_info=True)
        # Try to send a final error message if possible
        if not websocket.client_state == "DISCONNECTED":
            await websocket.send_json({"type": "error", "detail": "An unexpected server error occurred."})
    finally:
        # Ensure the websocket is closed
        try:
            if websocket.client_state != "DISCONNECTED":
                await websocket.close(code=1000)
        except Exception as e: # pylint: disable=broad-exception-caught
            logging.warning("Error closing websocket: %s", e)
