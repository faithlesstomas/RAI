import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import uuid

from returns.result import Success
from rai.services.chat import ChatService
from rai.services.history import HistoryService

class MockResponse:
    def __init__(self, content):
        self._content = content

    async def text(self):
        return self._content

    @property
    def tool_calls(self):
        async def empty_gen():
            for x in []:
                yield x
        return empty_gen()

    def __aiter__(self):
        # Allow streaming tests if needed
        async def gen():
            yield self._content
        return gen()

class MockAgent:
    def __init__(self, config):
        self.config = config
        self.conversation_id = "test-conv-id-1234"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def chat(self, prompt):
        return MockResponse("This is a mock reply")

@pytest.mark.asyncio
async def test_run_chain_persists_conversation_id():
    """Test that run_chain correctly associates the conversation ID with the session after ag.chat() completes."""
    session_id = f"test-session-{uuid.uuid4()}"
    chat_service = ChatService()
    
    # We patch Agent and the config managers
    with patch("rai.services.chat.Agent", side_effect=MockAgent) as mock_agent_class, \
         patch("rai.services.chat.load_config", return_value={"active_agent": "default"}), \
         patch("rai.services.chat.load_agents", return_value={"default": {}}), \
         patch("rai.services.chat.set_conversation_id_for_session") as mock_set_conv_id:
         
        result = await chat_service.run_chain(
            chain_input="Hello",
            chain_configs=[{"model": "test-model"}],
            session_id=session_id
        )
        
        assert isinstance(result, Success)
        # Verify the new conversation ID got set
        mock_set_conv_id.assert_called_once_with(session_id, "test-conv-id-1234")

@pytest.mark.asyncio
async def test_run_chain_loads_existing_conversation_id():
    """Test that run_chain loads the existing conversation ID from state and passes it to LocalAgentConfig."""
    session_id = f"test-session-{uuid.uuid4()}"
    chat_service = ChatService()
    existing_conv_id = "existing-conv-id-5678"
    
    with patch("rai.services.chat.Agent", side_effect=MockAgent) as mock_agent_class, \
         patch("rai.services.chat.load_config", return_value={"active_agent": "default"}), \
         patch("rai.services.chat.load_agents", return_value={"default": {}}), \
         patch("rai.services.chat.get_conversation_id_for_session", return_value=existing_conv_id), \
         patch("os.path.exists", return_value=True), \
         patch("rai.services.chat.set_conversation_id_for_session"):
         
        result = await chat_service.run_chain(
            chain_input="Hello again",
            chain_configs=[{"model": "test-model"}],
            session_id=session_id
        )
        
        assert isinstance(result, Success)
        
        # Verify config constructor was called with the existing conversation_id
        mock_agent_class.assert_called_once()
        config_arg = mock_agent_class.call_args[0][0]
        assert config_arg.conversation_id == existing_conv_id

@pytest.mark.asyncio
async def test_history_deduplication():
    """Test that run_chain does NOT write duplicate user messages, and assistant writes are handled by run_chain."""
    session_id = f"test-session-{uuid.uuid4()}"
    chat_service = ChatService()
    
    # Let's clean up after testing
    try:
        with patch("rai.services.chat.Agent", side_effect=MockAgent), \
             patch("rai.services.chat.load_config", return_value={"active_agent": "default"}), \
             patch("rai.services.chat.load_agents", return_value={"default": {}}), \
             patch("rai.services.chat.set_conversation_id_for_session"):
             
            # 1. Simulator: caller (like CLI loop) writes the user message
            await chat_service._history_service.add_message(session_id, "user", "User message")
            
            # 2. run_chain is called
            result = await chat_service.run_chain(
                chain_input="User message",
                chain_configs=[{"model": "test-model"}],
                session_id=session_id
            )
            
            assert isinstance(result, Success)
            
            # 3. Retrieve history and check message list
            history_res = await chat_service.get_session_history(session_id)
            assert isinstance(history_res, Success)
            messages = history_res.unwrap()
            
            # Should have exactly 2 messages: 1 user, 1 assistant (no duplicates!)
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "User message"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "This is a mock reply"
            
    finally:
        await chat_service._history_service.delete_session(session_id)
