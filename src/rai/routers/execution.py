import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Success

from .. import config_manager
from ..engine import run_chain

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

def get_config() -> Dict[str, Any]:
    """Dependency to load and provide the application configuration."""
    return config_manager.load_config()

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


@router.get("/api/v1/models/{backend}")
async def get_models_for_backend(
    backend: str,
    app_config: Dict[str, Any] = Depends(get_config)
) -> JSONResponse:
    """
    Returns a list of available models for a given backend.
    """
    print(f"--- Entering get_models_for_backend for backend: {backend} ---")
    if backend == "ollama":
        try:
            # pylint: disable=import-outside-toplevel
            import ollama  # noqa: PLC0415

            # Get host from the active session config, with a fallback
            active_session_name = app_config.get("active_session", "default")
            session_config = app_config.get("sessions", {}).get(active_session_name, {})
            host = session_config.get("ollama_host", "http://127.0.0.1:11434")
            logging.warning("Attempting to connect to Ollama at host: %s", host)

            # Use the async client inside an async function
            client = ollama.AsyncClient(host=host)
            models = await client.list()
            print(f"--- Ollama models response: {models} ---")
            model_names = [m.get("model") for m in models.get("models", [])]
            return JSONResponse(content={"models": model_names})
        except ImportError:
            raise HTTPException(status_code=501, detail="Ollama support is not installed.")
        except Exception as e:
            error_msg = f"Failed to get models from Ollama: {type(e).__name__} - {e}"
            logging.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
    # Placeholder for other backends
    return JSONResponse(content={"models": ["default-model"]})


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
    except Exception as e:
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
