import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

# Import the main function from cli.py
from rai.cli import main, setup_agent, run_single_query, run_interactive_chat
from ollama import ResponseError

# Mock the global console object
@pytest.fixture
def mock_console():
    with patch('rai.cli.console') as mock_console:
        # Mock console.input to provide predefined inputs for interactive chat
        mock_console.input.side_effect = ["test query", "exit"]
        yield mock_console

# Mock the base_tools list to prevent actual tool initialization
@pytest.fixture
def mock_base_tools():
    with patch('rai.cli.base_tools', new=[]) as mock_tools:
        yield mock_tools

# Mock load_dotenv to prevent actual .env loading
@pytest.fixture
def mock_load_dotenv():
    with patch('rai.cli.load_dotenv') as mock_load:
        yield mock_load

# Mock ollama.list to control available models
@pytest.fixture
def mock_ollama_list():
    with patch('rai.cli.ollama.list') as mock_list:
        mock_list.return_value = {'models': [{'model': 'gemma2:9b'}]}
        yield mock_list

# Mock agno.agent.Agent
@pytest.fixture
def mock_agent():
    with patch('rai.cli.Agent') as mock_agent_class:
        mock_instance = MagicMock()
        # Ensure agent.run returns an iterable that doesn't raise an error when list() is called
        mock_instance.run.return_value = [MagicMock(content="")] # A list containing a mock response chunk
        mock_agent_class.return_value = mock_instance
        yield mock_agent_class

# Mock agno.models.ollama.Ollama
@pytest.fixture
def mock_ollama_model():
    with patch('rai.cli.Ollama') as mock_ollama_model_class:
        mock_ollama_model_class.return_value = MagicMock()
        yield mock_ollama_model_class

# Mock sys.exit
@pytest.fixture
def mock_sys_exit():
    with patch('rai.cli.sys.exit') as mock_exit:
        yield mock_exit

# Mock os.getenv
@pytest.fixture
def mock_os_getenv():
    with patch('rai.cli.os.getenv') as mock_getenv:
        yield mock_getenv

@pytest.fixture
def mock_tavily_tools():
    with patch('rai.cli.TavilyTools') as MockTavilyTools:
        mock_instance = MagicMock()
        MockTavilyTools.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_panel():
    with patch('rai.cli.Panel') as MockPanel:
        mock_instance = MagicMock()
        MockPanel.return_value = mock_instance
        yield MockPanel


class TestMain:
    def test_main_default_call(self, mock_console, mock_ollama_list, mock_agent, mock_ollama_model, mock_sys_exit, mock_load_dotenv, mock_base_tools):
        runner = CliRunner()
        result = runner.invoke(main)

        assert result.exit_code == 0
        mock_console.print.assert_any_call("[dim]Using model: [bold]gemma2:9b[/bold][/dim]")
        mock_ollama_list.assert_called_once()
        mock_agent.assert_called_once()
        mock_ollama_model.assert_called_once()

    def test_run_interactive_chat(self, mock_console, mock_agent):
        # Mock agent.run to return a simple response
        mock_agent.run.return_value = [MagicMock(content="Agent response")]

        # Call the interactive chat function
        run_interactive_chat(mock_agent)

        # Assertions
        mock_console.input.assert_any_call("[bold green]You:[/]")
        mock_agent.run.assert_called_with("test query", stream=True)
        mock_console.out.assert_called_with("Agent response", end="", style="bright_blue")
        mock_console.print.assert_any_call("\n[yellow]Goodbye![/yellow]")


class TestSetupAgent:
    def test_setup_agent_with_tavily_key(self, mock_os_getenv, mock_load_dotenv, mock_agent, mock_ollama_model, mock_base_tools, mock_tavily_tools):
        mock_os_getenv.side_effect = lambda key: "TAVILY_KEY" if key == "TAVILY_API_KEY" else None # Simulate TAVILY_API_KEY present
        agent = setup_agent(system_prompt="test prompt", model_id="test_model")
        mock_load_dotenv.assert_called_once()
        mock_os_getenv.assert_any_call("TAVILY_API_KEY") # Use assert_any_call
        # Assert that TavilyTools was added to base_tools (this is tricky with new=[])
        # For now, we'll just assert agent was created
        expected_tools = mock_base_tools + [mock_tavily_tools] # Use the mocked TavilyTools instance
        mock_agent.assert_called_once_with(
            model=mock_ollama_model.return_value,
            tools=expected_tools, # Assert that TavilyTools was added
            show_tool_calls=True,
            markdown=True,
            add_history_to_messages=True,
            session_id="my_chat_session",
            instructions="test prompt",
        )
        assert agent == mock_agent.return_value

    def test_setup_agent_without_tavily_key(self, mock_os_getenv, mock_load_dotenv, mock_agent, mock_ollama_model, mock_base_tools, mock_console):
        mock_os_getenv.return_value = None # Simulate TAVILY_API_KEY missing
        agent = setup_agent(system_prompt="test prompt", model_id="test_model")
        mock_load_dotenv.assert_called_once()
        mock_os_getenv.assert_called_with("TAVILY_API_KEY")
        mock_console.print.assert_called_with("[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. Tavily will be disabled![/bold yellow]")
        mock_agent.assert_called_once_with(
            model=mock_ollama_model.return_value,
            tools=mock_base_tools, # This will be the patched empty list
            show_tool_calls=True,
            markdown=True,
            add_history_to_messages=True,
            session_id="my_chat_session",
            instructions="test prompt",
        )
        assert agent == mock_agent.return_value

    def test_setup_agent_model_no_tools_support(self, mock_os_getenv, mock_load_dotenv, mock_agent, mock_ollama_model, mock_base_tools, mock_console):
        mock_os_getenv.return_value = "TAVILY_KEY"
        # Simulate ResponseError indicating no tool support
        from ollama import ResponseError # Ensure ResponseError is imported
        mock_error = ResponseError("Mock error message", 500) # Create a real instance
        mock_error.error = "does not support tools" # Set the error attribute
        mock_agent.return_value.run.side_effect = mock_error

        agent = setup_agent(system_prompt="test prompt", model_id="test_model")

        mock_console.print.assert_called_with("[bold yellow]WARNING: Model 'test_model' does not support tools. Running in no-tools mode.[/bold yellow]")
        # Assert agent was re-initialized without tools
        mock_agent.assert_called_with(
            model=mock_ollama_model.return_value,
            tools=[], # Should be re-initialized with empty tools
            show_tool_calls=True,
            markdown=True,
            add_history_to_messages=True,
            session_id="my_chat_session",
        )
        assert agent == mock_agent.return_value

    def test_setup_agent_general_exception(self, mock_os_getenv, mock_load_dotenv, mock_agent, mock_ollama_model, mock_base_tools, mock_console, mock_sys_exit):
        mock_os_getenv.return_value = "TAVILY_KEY"
        mock_agent.return_value.run.side_effect = Exception("General error") # Simulate a general exception

        setup_agent(system_prompt="test prompt", model_id="test_model")

        mock_console.print.assert_any_call("[bold red]ERROR: Failed to initialize agent: General error[/bold red]")
        mock_sys_exit.assert_called_once_with(1)


class TestRunSingleQuery:
    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_run_single_query_success_with_tool_output(self, mock_redirect_stdout, mock_string_io, mock_console, mock_agent, mock_panel):
        # Mock StringIO to capture tool output
        mock_string_io_instance = MagicMock()
        mock_string_io.return_value = mock_string_io_instance
        mock_string_io_instance.getvalue.return_value = "Tool output captured"

        # Mock agent.run to return a response with content
        mock_agent.run.return_value = [MagicMock(content="Agent response")]

        run_single_query(mock_agent, prompt="test prompt")

        mock_console.print.assert_any_call("\n[bold blue]AI Assistant:[/]")
        mock_string_io.assert_called_once()
        mock_redirect_stdout.assert_called_once_with(mock_string_io_instance)
        mock_agent.run.assert_called_with("test prompt", stream=True)
        mock_string_io_instance.getvalue.assert_called_once()
        mock_panel.assert_called_once_with("Tool output captured", title="[bold yellow]Tool Call[/bold yellow]", border_style="yellow")
        mock_console.print.assert_any_call(mock_panel.return_value)
        mock_console.out.assert_called_with("Agent response", end="", style="bright_blue")
        mock_console.print.assert_any_call() # For the final newline

    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_run_single_query_success_no_tool_output(self, mock_redirect_stdout, mock_string_io, mock_console, mock_agent, mock_panel):
        # Mock StringIO to capture no tool output
        mock_string_io_instance = MagicMock()
        mock_string_io.return_value = mock_string_io_instance
        mock_string_io_instance.getvalue.return_value = ""

        # Mock agent.run to return a response with content
        mock_agent.run.return_value = [MagicMock(content="Agent response")]

        run_single_query(mock_agent, prompt="test prompt")

        mock_console.print.assert_any_call("\n[bold blue]AI Assistant:[/]")
        mock_string_io_instance.getvalue.assert_called_once()
        mock_panel.assert_not_called() # Assert that Panel was NOT called
        mock_console.out.assert_called_with("Agent response", end="", style="bright_blue")
        mock_console.print.assert_any_call() # For the final newline

    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_run_single_query_ollama_response_error(self, mock_redirect_stdout, mock_string_io, mock_console, mock_agent):
        # Simulate ResponseError
        error = ResponseError("Ollama error", 400)
        mock_agent.run.side_effect = error

        run_single_query(mock_agent, prompt="test prompt")

        mock_console.print.assert_any_call(f"\n[bold red]Ollama API error (status: {error.status_code}): {error.error}[/bold red]")

    @patch('rai.cli.io.StringIO')
    @patch('rai.cli.redirect_stdout')
    def test_run_single_query_general_exception(self, mock_redirect_stdout, mock_string_io, mock_console, mock_agent):
        error = Exception("General error")
        mock_agent.run.side_effect = error

        run_single_query(mock_agent, prompt="test prompt")

        mock_console.print.assert_any_call(f"\n[bold red]An unexpected error occurred during streaming: {error}[/bold red]")