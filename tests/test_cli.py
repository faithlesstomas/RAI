"""Tests for the rai.cli module."""
from typing import Generator
from unittest.mock import MagicMock, patch, call
import pytest
from agno.agent import Agent
from rai.cli import setup_agent

# Fixtures for mocking common dependencies
@pytest.fixture(name="mock_console")
def fixture_mock_console() -> Generator:
    """Fixture for mocking the rich console."""
    with patch('rai.cli.console') as mock:
        yield mock

@pytest.fixture(name="mock_error_console")
def fixture_mock_error_console() -> Generator:
    """Fixture for mocking the rich error console."""
    with patch('rai.core.error_console') as mock:
        yield mock

@pytest.fixture(name="mock_agent")
def fixture_mock_agent() -> Generator:
    """Fixture for mocking the Agno agent."""
    with patch('rai.cli.Agent', spec=Agent) as mock:
        yield mock

@pytest.fixture(name="mock_sys_exit")
def fixture_mock_sys_exit() -> Generator:
    """Fixture for mocking sys.exit."""
    with patch('rai.core.sys.exit') as mock:
        yield mock

@pytest.fixture(name="mock_os_getenv")
def fixture_mock_os_getenv() -> Generator:
    """Fixture for mocking os.getenv."""
    with patch('rai.cli.os.getenv') as mock:
        yield mock

@pytest.fixture(name="mock_load_dotenv")
def fixture_mock_load_dotenv() -> Generator:
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
def fixture_mock_panel() -> Generator:
    """Fixture for mocking rich.panel.Panel."""
    with patch('rai.cli.Panel') as mock:
        yield mock

# --- Test Cases ---

class TestSetupAgent:

    """Tests for the setup_agent function."""

    @patch('rai.core.Agent', spec=Agent)
    @patch('rai.core.setup_model')
    @patch('rai.core.setup_tools')
    def test_setup_agent_logic(self, mock_setup_tools: MagicMock, mock_setup_model: MagicMock, mock_agent: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test the core logic of agent setup."""
        mock_setup_model.return_value = (MagicMock(), [])
        mock_setup_tools.return_value = ([], [])

        with patch.dict('rai.core.RAI_CONFIG', {
            'backend': 'ollama',
            'model': 'test_model',
            'system': 'test prompt',
            'tools': None
        }):
            agent, messages = setup_agent()

        mock_setup_model.assert_called_once_with('ollama', 'test_model', quiet=True)
        mock_setup_tools.assert_called_once_with(enable_tools=False, quiet=True, enabled_tool_names=None, has_prompt=True)
        mock_agent.assert_called_once()
        assert agent is not None
        assert isinstance(messages, list)

    @patch('rai.core.sys.exit')
    @patch('rai.core.Agent', side_effect=Exception("General error"))
    @patch('rai.core.setup_tools')
    @patch('rai.core.setup_model')
    def test_setup_agent_exception(self, mock_setup_model: MagicMock, mock_setup_tools: MagicMock,  # noqa: PLR0913
                                   mock_agent_ctor: MagicMock, mock_sys_exit: MagicMock, mock_load_dotenv: MagicMock, mock_error_console: MagicMock) -> None:
        """Test handling of a general exception during agent initialization."""

        mock_setup_model.return_value = (MagicMock(), [])
        mock_setup_tools.return_value = ([], [])

        with patch.dict('rai.core.RAI_CONFIG', {
            'backend': 'ollama',
            'model': 'test_model',
            'system': 'test prompt',
            'tools': None
        }):
            setup_agent(quiet=True)

        expected_calls = [
            call('[bold red]ERROR: Failed to initialize agent: General error[/bold red]'),
            call('[yellow]Is the Ollama server running and is the model pulled?[/yellow]')
        ]

        mock_error_console.print.assert_has_calls(expected_calls, any_order=True)
        mock_sys_exit.assert_called_once_with(1)
