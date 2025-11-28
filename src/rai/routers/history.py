"""
History router for RAI.
"""
from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1/history",
    tags=["History"],
    responses={404: {"description": "Not found"}},
)

# --- Endpoints ---

@router.get("/sessions")
async def list_history_sessions() -> Dict[str, Any]:
    """
    List past conversation sessions.
    (Placeholder: Currently returns mock data until persistence is implemented)
    """
    # TODO: Implement actual history reading from a database or log files.
    return {
        "sessions": [
            {
                "id": "mock-session-1",
                "timestamp": "2023-10-27T10:00:00",
                "summary": "Discussion about API design",
            },
            {
                "id": "mock-session-2",
                "timestamp": "2023-10-27T11:30:00",
                "summary": "Debugging WebSocket issues",
            },
        ]
    }

@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """
    Get full chat history for a specific session.
    """
    if session_id == "mock-session-1":
        return {
            "id": "mock-session-1",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I am fine, thank you. How can I help you?"}
            ]
        }
    return {"id": session_id, "messages": []}
