"""
Main file for the FastAPI server.
"""
import logging
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Success

from . import config_manager
from .engine import run_chain

app = FastAPI(
    title="RAI - Rich AI Assistant",
    description="Backend server for the RAI CLI and other clients.",
    version="0.1.0",
)


# --- Pydantic Models ---

class ChainRequest(BaseModel):
    """Request model for running a chain of agents."""
    chain_input: str
    chain_configs: List[Dict[str, Any]]


# --- Dependencies ---

def get_config() -> Dict[str, Any]:
    """Dependency to load and provide the application configuration."""
    return config_manager.load_config()


# --- API Endpoints ---

@app.get("/health", tags=["Server"])
async def health_check() -> JSONResponse:
    """
    Simple health check endpoint to confirm the server is running.
    """
    return JSONResponse(content={"status": "ok"})


@app.post("/api/v1/run", tags=["AI Engine"])
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


@app.websocket("/ws/v1/chat")
async def websocket_endpoint(
        websocket: WebSocket,
        app_config: Dict[str, Any] = Depends(get_config)
) -> None:
    """Handles WebSocket connections for real-time chat."""
    await websocket.accept()
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
                app_config=app_config
            )

            match result:
                case Success(payload):
                    await websocket.send_json({"status": "success", "payload": payload})
                case Failure(error):
                    logging.error("Error during WebSocket chain execution: %s", error, exc_info=error)
                    await websocket.send_json({"status": "error", "detail": str(error)})

    except WebSocketDisconnect:
        logging.info("Client disconnected from WebSocket.")
    except Exception as e:
        logging.error("Unexpected error in WebSocket: %s", e, exc_info=True)
        raise e
