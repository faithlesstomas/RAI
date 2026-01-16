"""Tests for the AgnoAdapter."""
from unittest.mock import MagicMock, patch

from rai.adapters.agno import AgnoAdapter


def test_get_history_with_complex_mock_storage() -> None:
    """
    Test the get_history method with a mocked storage that returns a
    realistic session object.
    """
    # 1. Setup
    mock_agent = MagicMock()
    mock_agent.session_id = "test-session"
    mock_agent.storage = MagicMock()

    # Mock the return value of storage.get_session
    mock_session = MagicMock()
    
    # Create mock runs with object attributes, as preferred by new logging
    run1 = MagicMock()
    run1.messages = [
        MagicMock(role="user", content="Hello"),
        MagicMock(role="assistant", content="Hi there!")
    ]
    # Set dict access for robustness check (optional, but good)
    # run1.get = lambda k, d=None: ... # Too complex, stick to attributes which code checks first
    
    run2 = MagicMock()
    run2.messages = [
        MagicMock(role="user", content="How are you?"),
        MagicMock(role="assistant", content="I am fine, thank you.")
    ]
    
    mock_session.runs = [run1, run2]
    
    # Support both checks (storage or db)
    mock_agent.db = MagicMock()
    mock_agent.db.get_session.return_value = mock_session
    mock_agent.storage = None # Force use of db path or ensure logic handles it

    # We don't need to patch creation anymore, just inject into the cache
    adapter = AgnoAdapter(agent_config={"session_id": "test-session"})
    adapter.agents["test-session"] = mock_agent

    # 2. Execute
    history = adapter.get_history()

    # 3. Assert
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
    ]
    mock_agent.db.get_session.assert_called_once_with(session_id="test-session", session_type="agent")
