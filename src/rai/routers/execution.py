"""
API endpoints for executing agent chains.
"""
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Success

from .. import config_manager
from ..services.chat import ChatService
from ..dependencies import get_config, get_model_registry
from ..services.model_registry import ModelRegistry
from ..tools.security.auth import is_authorized

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

class AgentExecutionRequest(BaseModel):
    """Simplified request model for running an agent execution."""
    prompt: Optional[str] = None
    chain_input: Optional[str] = None  # Backward compatibility
    agent_id: Optional[str] = None
    session_id: Optional[str] = None  # Conversation session ID
    context: Optional[Dict[str, Any]] = None
    chain_configs: Optional[List[Dict[str, Any]]] = None  # Backward compatibility

# Keep ChainRequest alias for backward compatibility
ChainRequest = AgentExecutionRequest

# --- Endpoints ---

@router.post("/api/v1/run")
async def execute_chain(
        request: AgentExecutionRequest,
        app_config: Dict[str, Any] = Depends(get_config)
) -> JSONResponse:
    """
    Runs an agent with the given input and configurations.
    """
    chat_service = ChatService()
    prompt = request.prompt or request.chain_input
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt or chain_input.")

    result = await chat_service.run_chain(
        chain_input=prompt,
        chain_configs=request.chain_configs,
        session_id=request.session_id,
        context=request.context,
        agent_id=request.agent_id,
    )

    match result:
        case Success(payload):
            return JSONResponse(content={"status": "success", "payload": payload})
        case Failure(error):
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

            logging.error("Error during agent execution: %s", error, exc_info=error)
            raise HTTPException(status_code=500, detail=str(error))


@router.post("/api/v1/stream")
async def stream_chain_endpoint(
        request: AgentExecutionRequest,
        app_config: Dict[str, Any] = Depends(get_config)
) -> StreamingResponse:
    """
    Streams the result of an agent execution using Server-Sent Events (SSE).
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        chat_service = ChatService()
        prompt = request.prompt or request.chain_input
        if not prompt:
            yield f"event: error\ndata: {json.dumps({'detail': 'Missing prompt or chain_input.'})}\n\n"
            return

        async for chunk in chat_service.stream_chain(
            chain_input=prompt,
            chain_configs=request.chain_configs,
            session_id=session_id,
            context=request.context,
            agent_id=request.agent_id,
        ):
            if isinstance(chunk, Failure):
                # Send error as a specific event or data
                error_msg = str(chunk.failure())
                yield f"event: error\ndata: {json.dumps({'detail': error_msg})}\n\n"
                return

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
        yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


from ..tools.security.hitl import get_approval_manager

class ApprovalResolutionRequest(BaseModel):
    """Request model to resolve a pending HITL authorization request."""
    approved: bool

@router.get("/api/v1/approvals")
async def list_approvals() -> JSONResponse:
    """
    Lists all pending HITL tool execution approval requests.
    """
    manager = get_approval_manager()
    return JSONResponse(content={"approvals": manager.list_pending()})


@router.post("/api/v1/approvals/{approval_id}/resolve")
async def resolve_approval(
    approval_id: str,
    request: ApprovalResolutionRequest
) -> JSONResponse:
    """
    Resolves a pending HITL authorization request (approves or denies execution).
    """
    manager = get_approval_manager()
    success = manager.resolve_request(approval_id, request.approved)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Pending approval request '{approval_id}' not found or already resolved."
        )
    return JSONResponse(content={"status": "success", "message": f"Approval '{approval_id}' resolved to: {request.approved}"})


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
    if not is_authorized(websocket.headers, websocket.query_params.get("token")):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await websocket.accept()
    app_config = config_manager.load_config()  # Load config manually
    session_id = None  # Preserve the stateful session ID across turns!
    try:
        while True:
            data = await websocket.receive_json()
            try:
                request = AgentExecutionRequest.model_validate(data)
            except ValidationError as e:
                await websocket.send_json({"status": "error", "detail": e.errors()})
                continue

            prompt = request.prompt or request.chain_input
            if not prompt:
                await websocket.send_json({"status": "error", "detail": ["Missing prompt or chain_input."]})
                continue

            current_session_id = request.session_id or session_id

            chat_service = ChatService()
            result = await chat_service.run_chain(
                chain_input=prompt,
                chain_configs=request.chain_configs,
                session_id=current_session_id,
                context=request.context,
                agent_id=request.agent_id,
            )

            match result:
                case Success(payload):
                    # Store/update the active session ID to preserve session propagation!
                    session_id = payload.get("session_id")
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
