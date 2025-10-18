"""Tests for the rai.cli module."""
from unittest.mock import MagicMock, patch, call
import pytest
from agno.agent import Agent
from rai.cli import setup_agent

# Fixtures for mocking common dependencies
@pytest.fixture(name="mock_console")
def fixture_mock_console():
    """Fixture for mocking the rich console."""
    with patch('rai.cli.console') as mock:
        yield mock

@pytest.fixture(name="mock_error_console")
def fixture_mock_error_console():
    """Fixture for mocking the rich error console."""
    with patch('rai.core.error_console') as mock:
        yield mock

@pytest.fixture(name="mock_agent")
def fixture_mock_agent():
    """Fixture for mocking the Agno agent."""
    with patch('rai.cli.Agent', spec=Agent) as mock:
        yield mock

@pytest.fixture(name="mock_sys_exit")
def fixture_mock_sys_exit():
    """Fixture for mocking sys.exit."""
    with patch('rai.core.sys.exit') as mock:
        yield mock

@pytest.fixture(name="mock_os_getenv")
def fixture_mock_os_getenv():
    """Fixture for mocking os.getenv."""
    with patch('rai.cli.os.getenv') as mock:
        yield mock

@pytest.fixture(name="mock_load_dotenv")
def fixture_mock_load_dotenv():
    """Fixture for mocking load_dotenv."""
    with patch('rai.cli.load_config') as mock: # Updated to mock load_config which is used by setup_agent
        mock.return_value = {
            "active_session": "default",
            "sessions": {
                "default": {
                    "model": "test_model",
                    "backend": "ollama",
                    "system": "test prompt"
                }
            }
        }
        yield mock

@pytest.fixture(name="mock_panel")
def fixture_mock_panel():
    """Fixture for mocking rich.panel.Panel."""
    with patch('rai.cli.Panel') as mock:
        yield mock

# --- Test Cases ---

class TestSetupAgent:

    """Tests for the setup_agent function."""

    @patch('rai.core.Agent', spec=Agent)
    @patch('rai.core._setup_model')
    @patch('rai.core._setup_tools')
    def test_setup_agent_logic(self, mock_setup_tools, mock_setup_model, mock_agent, mock_load_dotenv) -> None:
        """Test the core logic of agent setup."""
        mock_setup_model.return_value = (MagicMock(), [])
        mock_setup_tools.return_value = ([], [])

        with patch.dict('rai.core.RAI_CONFIG', {
            'backend': 'ollama',
            'model': 'test_model',
            'system': 'test prompt'
        }):
            agent, messages = setup_agent()

        mock_setup_model.assert_called_once_with('ollama', 'test_model', False)
        mock_setup_tools.assert_called_once_with(True, False, None)
        mock_agent.assert_called_once()
        assert agent is not None
        assert isinstance(messages, list)

    @patch('rai.core.sys.exit')
    @patch('rai.core.Agent', side_effect=Exception("General error"))
    @patch('rai.core._setup_tools')
    @patch('rai.core._setup_model')
    def test_setup_agent_exception(self, mock_setup_model, mock_setup_tools,  # noqa: PLR0913
                                   mock_agent_ctor, mock_sys_exit, mock_load_dotenv, mock_error_console) -> None:
        """Test handling of a general exception during agent initialization."""

        mock_setup_model.return_value = (MagicMock(), [])
        mock_setup_tools.return_value = ([], [])

        with patch.dict('rai.core.RAI_CONFIG', {
            'backend': 'ollama',
            'model': 'test_model',
            'system': 'test prompt'
        }):
            setup_agent()

        expected_calls = [
            call('[bold red]ERROR: Failed to initialize agent: General error[/bold red]'),
            call('[yellow]Is the Ollama server running and is the model pulled?[/yellow]')
        ]

        mock_error_console.print.assert_has_calls(expected_calls, any_order=True)
        mock_sys_exit.assert_called_once_with(1)
