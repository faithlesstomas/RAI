"""Tests for the rai.cli module."""
from unittest.mock import MagicMock, patch, call
import pytest
from click.testing import CliRunner
from agno.agent import Agent, Message
from rai.cli import main, setup_agent, run_single_query

# Fixtures for mocking common dependencies
@pytest.fixture(name="mock_console")
def fixture_mock_console():
    """Fixture for mocking the rich console."""
    with patch('rai.cli.console') as mock:
        yield mock

@pytest.fixture(name="mock_agent")
def fixture_mock_agent():
    """Fixture for mocking the Agno agent."""
    with patch('rai.cli.Agent', spec=Agent) as mock:
        yield mock

@pytest.fixture(name="mock_sys_exit")
def fixture_mock_sys_exit():
    """Fixture for mocking sys.exit."""
    with patch('rai.cli.sys.exit') as mock:
        yield mock

@pytest.fixture(name="mock_os_getenv")
def fixture_mock_os_getenv():
    """Fixture for mocking os.getenv."""
    with patch('rai.cli.os.getenv') as mock:
        yield mock

@pytest.fixture(name="mock_load_dotenv")
def fixture_mock_load_dotenv():
    """Fixture for mocking load_dotenv."""
    with patch('rai.cli.load_dotenv') as mock:
        yield mock

@pytest.fixture(name="mock_panel")
def fixture_mock_panel():
    """Fixture for mocking rich.panel.Panel."""
    with patch('rai.cli.Panel') as mock:
        yield mock

# --- Test Cases ---

class TestSetupAgent:
    """Tests for the setup_agent function."""

    @patch('rai.cli.base_tools', [])
    @patch('rai.cli.Ollama')
    def test_setup_agent_no_tavily(self, mock_ollama, mock_tools, mock_os_getenv,  # noqa: PLR0913
                                   mock_load_dotenv, mock_agent, mock_console):
        """Test successful agent setup without Tavily API key."""
        mock_os_getenv.return_value = None
        agent = setup_agent(system_prompt="test prompt", model_id="test_model")
        mock_load_dotenv.assert_called_once()
        mock_console.print.assert_called_with(
            "[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. "
            "Tavily will be disabled![/bold yellow]"
        )
        mock_agent.assert_called_once()
        assert agent is not None

    @patch('rai.cli.base_tools', [])
    @patch('rai.cli.Ollama')
    def test_setup_agent_with_tavily(self, mock_ollama, mock_tools, mock_os_getenv,
                                     mock_load_dotenv, mock_agent):
        """Test successful agent setup with Tavily API key."""
        mock_os_getenv.return_value = "TAVILY_KEY"
        setup_agent(system_prompt="test prompt", model_id="test_model")
        mock_agent.assert_called_once()
        _args, kwargs = mock_agent.call_args
        assert any(isinstance(tool, MagicMock) for tool in kwargs['tools'])

    @patch('rai.cli.base_tools', [])
    @patch('rai.cli.Ollama')
    def test_setup_agent_exception(self, mock_ollama, mock_tools, mock_os_getenv,  # noqa: PLR0913
                                   mock_load_dotenv, mock_agent, mock_console, mock_sys_exit):
        """Test handling of a general exception during agent initialization."""
        mock_os_getenv.return_value = None
        mock_agent.side_effect = Exception("General error")
        setup_agent(system_prompt="test prompt", model_id="test_model")
        mock_console.print.assert_any_call(
            "[bold red]ERROR: Failed to initialize agent: General error[/bold red]"
        )
        mock_sys_exit.assert_called_once_with(1)


class TestRunSingleQuery:
    """Tests for the run_single_query function."""

    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_simple_query(self, _redirect, mock_stringio, mock_console, mock_agent, mock_panel):
        """Test a simple query without tool usage."""
        mock_agent.run.return_value = Message(content="Simple response")
        mock_stringio.return_value.getvalue.return_value = ""  # No tool output

        run_single_query(mock_agent, "hello")

        mock_agent.run.assert_called_once_with("hello", stream=False)
        mock_console.out.assert_called_once_with("Simple response", style="bright_blue")
        mock_panel.assert_not_called()

    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_text_tool_output(self, _redirect, mock_stringio, mock_console, mock_agent, mock_panel):
        """Test a query that results in simple text output from a tool."""
        mock_agent.run.return_value = Message(content="Response after tool")
        mock_stringio.return_value.getvalue.return_value = "Some tool text"

        run_single_query(mock_agent, "what's the weather?")

        mock_agent.run.assert_called_once_with("what's the weather?", stream=False)
        mock_panel.assert_called_once_with(
            "Some tool text", title="[bold yellow]Tool Call[/bold yellow]", border_style="yellow"
        )
        mock_console.out.assert_called_once_with("Response after tool", style="bright_blue")

    @patch('rai.cli.Image')
    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_image_tool_output(self, _redirect, mock_stringio, mock_image,  # noqa: PLR0913
                               mock_console, mock_agent, mock_panel):
        """Test a query that triggers the screenshot tool and a second analysis run."""
        mock_agent.run.return_value = Message(content="")
        image_json = '{"type": "image_data", "format": "png", "base64": "fake_base64_data"}'
        mock_stringio.return_value.getvalue.return_value = image_json

        second_run_stream = [Message(content="I see a cat.")]
        mock_agent.run.side_effect = [
            Message(content=""),
            second_run_stream
        ]

        run_single_query(mock_agent, "take a screenshot")

        call1 = call("take a screenshot", stream=False)
        mock_image.assert_called_once_with(source="fake_base64_data", mime_type="image/png")
        call2 = call("take a screenshot", images=[mock_image.return_value], stream=True)

        mock_agent.run.assert_has_calls([call1, call2])
        mock_panel.assert_called_once_with(
            "Screenshot captured successfully.",
            title="[bold yellow]Tool Call[/bold yellow]",
            border_style="yellow"
        )
        mock_console.out.assert_called_once_with("I see a cat.", end="", style="bright_blue")

class TestMainExecution:
    """High-level tests for the main CLI execution path."""

    @patch('rai.cli.setup_agent')
    @patch('rai.cli.ollama.list')
    def test_model_not_available(self, mock_ollama_list, _mock_setup_agent,
                                 _mock_console, mock_sys_exit):
        """Test that the CLI exits if the specified model is not available."""
        mock_ollama_list.return_value = {"models": [{"model": "some-other-model:latest"}]}
        runner = CliRunner()
        result = runner.invoke(main, ["-m", "nonexistent-model", "test prompt"])

        assert result.exit_code == 1
        mock_sys_exit.assert_called_with(1)
        assert "Error: Model 'nonexistent-model' is not available" in result.output

    @patch('rai.cli.setup_agent')
    @patch('rai.cli.ollama.list')
    def test_model_name_handling(self, mock_ollama_list, mock_setup_agent, _mock_console):
        """Test that model names without tags are correctly identified."""
        mock_ollama_list.return_value = {"models": [{"model": "llama3.2:latest"}]}
        with patch('rai.cli.run_single_query'):
            runner = CliRunner()
            result = runner.invoke(main, ["-m", "llama3.2", "test prompt"])

        assert result.exit_code == 0
        assert "Error: Model 'llama3.2' is not available" not in result.output
        mock_setup_agent.assert_called_once_with(
            system_prompt="You are a versatile and helpful AI assistant.",
            model_id="llama3.2"
        )
