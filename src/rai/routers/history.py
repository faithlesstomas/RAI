"""
History router for RAI.
"""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from returns.result import Success, Failure

from ..services.history import HistoryService

router = APIRouter(
    prefix="/api/v1/history",
    tags=["History"],
    responses={404: {"description": "Not found"}},
)

# Instantiate service (singleton-like for now, or per request if needed)
# Ideally this should come from dependencies.
history_service = HistoryService()

# --- Endpoints ---

@router.get("/sessions")
async def list_history_sessions() -> Dict[str, Any]:
    """
    List past conversation sessions.
    """
    result = await history_service.list_sessions()
    if isinstance(result, Failure):
        raise HTTPException(status_code=500, detail=str(result.failure()))
    
    return {"sessions": result.unwrap()}

@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """
    Get full chat history for a specific session.
    """
    result = await history_service.get_session_history(session_id)
    if isinstance(result, Failure):
        raise HTTPException(status_code=500, detail=str(result.failure()))
        
    messages = result.unwrap()
    # If no messages, maybe return 404 or just empty list? 
    # Current implementation returns empty list which is fine.
    
    return {"id": session_id, "messages": messages}


@router.delete("/sessions/{session_id}")
async def clear_session_history(session_id: str) -> Dict[str, Any]:
    """
    Clear chat history for a specific session.
    """
    result = await history_service.clear_history(session_id)
    if isinstance(result, Failure):
        raise HTTPException(status_code=500, detail=str(result.failure()))
        
    return {"status": "success", "message": f"History cleared for session '{session_id}'."}

