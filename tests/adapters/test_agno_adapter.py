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

    # Mock the return value of storage.read
    mock_session = MagicMock()
    mock_session.memory = {
        "runs": [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I am fine, thank you."},
                ]
            },
        ]
    }
    mock_agent.storage.read.return_value = mock_session

    # We need to patch the adapter's _create_agent_from_config method
    # to inject our mock agent.
    with patch.object(AgnoAdapter, "_create_agent_from_config", return_value=mock_agent):
        adapter = AgnoAdapter(agent_config={})

        # 2. Execute
        history = adapter.get_history()

        # 3. Assert
        assert history == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
        ]
        mock_agent.storage.read.assert_called_once_with(session_id="test-session")
