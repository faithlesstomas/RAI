import pytest
import uuid
from rai.services.history import HistoryService
from rai.routers.history import list_history_sessions, get_session_history, history_service
from returns.result import Success

@pytest.mark.asyncio
async def test_history_flow():
    # Setup
    session_id = str(uuid.uuid4())
    # Override db path for test safety if possible, or assume it uses default/env
    # For this test we use the single instance imported from router
    
    # 1. Add messages
    await history_service.add_message(session_id, "user", "Hello Test")
    await history_service.add_message(session_id, "assistant", "Hi there")
    
    # 2. Test list_sessions via router
    sessions_response = await list_history_sessions()
    sessions = sessions_response["sessions"]
    
    assert any(s["id"] == session_id for s in sessions)
    
    # 3. Test get_session_history via router
    history_response = await get_session_history(session_id)
    messages = history_response["messages"]
    
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello Test"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there"

    # Cleanup (Optional)
    await history_service.delete_session(session_id)
