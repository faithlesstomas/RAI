import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Success

from .. import config_manager
from ..engine import run_chain
from ..dependencies import get_config, get_model_registry
from ..services.model_registry import ModelRegistry

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
            logging.error("Error during chain execution: %s", error, exc_info=error)
            raise HTTPException(status_code=500, detail=str(error))


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
        except Exception as e:
             logging.warning(f"Error closing websocket: {e}")
