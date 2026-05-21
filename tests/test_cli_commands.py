# ruff: noqa: ANN001
"""Tests for the main CLI commands in rai.cli."""
import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import click.testing
from rich.panel import Panel
from rich.markdown import Markdown
from returns.result import Success, Failure

from rai import cli as rai_cli


@pytest.mark.asyncio
@patch.dict("rai.cli._SLASH_COMMAND_HANDLERS", {"config": AsyncMock()})
async def test_slash_command_dispatches_to_config() -> None:
    """Test that /config command is dispatched to the config handler."""
    agent = MagicMock()
    run_config = {}
    mock_config_handler = rai_cli._SLASH_COMMAND_HANDLERS["config"]

    user_input = "/config set model gemma"
    should_exit = await rai_cli._handle_slash_command(user_input, run_config, agent)

    assert not should_exit
    mock_config_handler.assert_awaited_once_with(
        ["set", "model", "gemma"], run_config, agent
    )


@pytest.mark.asyncio
@patch("rai.cli.console.print")
async def test_config_command_show_implementation(mock_print) -> None:
    """Test the implementation of the '/config show' command."""
    agent = MagicMock()
    run_config = {"model": "test-model", "backend": "test-backend"}

    # Test the 'show' subcommand (default)
    user_input_show = "/config"
    await rai_cli._handle_slash_command(user_input_show, run_config, agent)

    mock_print.assert_called_once()
    args, _ = mock_print.call_args
    panel_arg = args[0]
    assert isinstance(panel_arg, Panel)
    assert panel_arg.title == "Current Session Config"
    assert json.loads(panel_arg.renderable) == run_config


@pytest.mark.asyncio
@patch("rai.cli.console.print")
async def test_config_command_get_implementation(mock_print) -> None:
    """Test the implementation of the '/config get' command."""
    agent = MagicMock()
    run_config = {"model": "test-model"}

    # Test the 'get' subcommand
    user_input_get = "/config get model"
    await rai_cli._handle_slash_command(user_input_get, run_config, agent)

    mock_print.assert_called_once_with("model: test-model")


@pytest.mark.asyncio
async def test_config_command_set_implementation() -> None:
    """Test the implementation of the '/config set' command."""
    agent = MagicMock()
    run_config = {"model": "old-model"}

    # Test the 'set' subcommand
    user_input_set = "/config set model new-model"
    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_slash_command(user_input_set, run_config, agent)
        # Check that a confirmation message was printed
        mock_print.assert_any_call(
            "Set 'model' to 'new-model' (persisted)."
        )

    # Check that the run_config was modified
    assert run_config["model"] == "new-model"


@patch("rai.cli.click.launch")
@patch("rai.cli.config_manager.get_config_path")
@patch("rai.cli.console.print")
def test_edit_config_command(mock_console_print, mock_get_config_path, mock_click_launch) -> None:
    """Test the 'rai config edit' command."""
    mock_get_config_path.return_value = "/fake/path/to/config.json"
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, ["config", "edit"])

    mock_get_config_path.assert_called_once()
    mock_click_launch.assert_called_once_with("/fake/path/to/config.json")
    mock_console_print.assert_called_once_with("[dim]Opening configuration file: /fake/path/to/config.json[/dim]")
    assert result.exit_code == 0


@patch("rai.cli.config_manager.get_config_path")
@patch("rai.cli.error_console.print")
def test_edit_config_command_no_config_path(mock_error_console_print, mock_get_config_path) -> None:
    """Test the 'rai config edit' command when config path cannot be determined."""
    mock_get_config_path.return_value = None
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, ["config", "edit"])

    mock_get_config_path.assert_called_once()
    mock_error_console_print.assert_called_once_with("[red]Could not determine configuration file path.[/red]")
    assert result.exit_code == 0


@patch("rai.cli.async_run_standalone")
def test_cli_standalone_one_shot(mock_async_run_standalone) -> None:
    """Test 'rai --prompt "hello"' executes in standalone one-shot mode."""
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, ["--prompt", "hello"])

    mock_async_run_standalone.assert_called_once()
    options = mock_async_run_standalone.call_args.args[0]
    assert options.prompt == "hello"
    assert result.exit_code == 0


@patch("rai.cli.async_run_standalone")
def test_cli_standalone_interactive(mock_async_run_standalone) -> None:
    """Test 'rai' (no prompt) executes in standalone interactive mode."""
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, [])

    mock_async_run_standalone.assert_called_once()
    options = mock_async_run_standalone.call_args.args[0]
    assert options.prompt == ""
    assert result.exit_code == 0


@patch("rai.cli.async_main_client")
def test_cli_connect_one_shot(mock_async_main_client) -> None:
    """Test 'rai --connect --prompt "hello"' executes in client one-shot mode."""
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, ["--connect", "--prompt", "hello"])

    assert result.exit_code == 0
    mock_async_main_client.assert_called_once()
    options = mock_async_main_client.call_args.args[0]
    assert options.prompt == "hello"
    assert options.connect_uri == "_auto_"


@patch("rai.cli.async_main_client")
def test_cli_connect_interactive(mock_async_main_client) -> None:
    """Test 'rai --connect' (no prompt) executes in client interactive mode."""
    runner = click.testing.CliRunner()
    result = runner.invoke(rai_cli.cli, ["--connect"])

    assert result.exit_code == 0
    mock_async_main_client.assert_called_once()
    options = mock_async_main_client.call_args.args[0]
    assert options.prompt == ""
    assert options.connect_uri == "_auto_"


@patch("rai.cli.async_main_client")
def test_cli_connect_with_uri(mock_async_main_client) -> None:
    """Test 'rai --connect <uri>' executes in client interactive mode."""
    runner = click.testing.CliRunner()
    uri = "ws://test.host:1234"
    result = runner.invoke(rai_cli.cli, ["--connect", uri])

    assert result.exit_code == 0
    mock_async_main_client.assert_called_once()
    options = mock_async_main_client.call_args.args[0]
    assert options.prompt == ""
    assert options.connect_uri == uri


@patch("rai.cli.PromptSession")
@patch("rai.cli.console.print")
async def test_run_interactive_chat_success(mock_console_print, mock_prompt_session_cls) -> None:
    """Test the interactive chat loop with a successful response."""
    mock_prompt_session = mock_prompt_session_cls.return_value
    mock_prompt_session.prompt_async = AsyncMock(side_effect=["hello", EOFError])

    mock_processor = MagicMock()
    mock_processor.arun = AsyncMock(return_value=Success({"content": "Test response"}))
    mock_processor.close = AsyncMock()

    # Create mock chat service
    mock_chat_service = MagicMock()
    mock_chat_service.get_session_history = AsyncMock(return_value=Success([]))
    mock_chat_service._history_service.add_message = AsyncMock(return_value=Success(None))
    mock_chat_service.add_message_to_history = AsyncMock(return_value=Success(None))

    run_config = {
        "chat_service": mock_chat_service,
        "session_id": "test-session"
    }

    await rai_cli.run_interactive_chat(mock_processor, run_config, rai_cli.CliOptions())

    # Verify history interaction
    mock_chat_service.get_session_history.assert_awaited_with("test-session")
    # Verify user message add
    mock_chat_service.add_message_to_history.assert_awaited() 
    
    mock_processor.arun.assert_awaited_once() # Called with history now

    # Find the call to print with the Markdown object
    found_markdown = False
    for call in mock_console_print.call_args_list:
        arg = call.args[0]
        if isinstance(arg, Markdown) and arg.markup == "Test response":
            found_markdown = True
            break
    assert found_markdown, "Markdown response was not printed"

    mock_processor.close.assert_awaited_once()


@patch("rai.cli.LocalProcessor", autospec=True)
@patch("rai.cli.console.print")
@patch("rai.cli._setup_standalone_processor")
@patch("rai.cli._build_run_config")
async def test_async_run_standalone_one_shot_success(
    mock_build_run_config, mock_setup_processor, mock_console_print, MockLocalProcessor
) -> None:
    """Test async_run_standalone in one-shot mode with a successful response."""
    mock_build_run_config.return_value = ({}, {}, "default", {})
    
    mock_processor_instance = MockLocalProcessor.return_value
    mock_processor_instance.arun = AsyncMock(return_value=Success({"content": "Standalone one-shot response"}))
    mock_processor_instance.close = AsyncMock()
    
    mock_chat_service = MagicMock()
    mock_chat_service.get_session_history = AsyncMock(return_value=Success([]))
    mock_chat_service._history_service.add_message = AsyncMock(return_value=Success(None))
    mock_chat_service.add_message_to_history = AsyncMock(return_value=Success(None))

    mock_setup_processor.return_value = (mock_processor_instance, mock_chat_service)

    options = rai_cli.CliOptions(prompt="test prompt")
    await rai_cli.async_run_standalone(options)

    # Verify history interaction
    mock_chat_service.get_session_history.assert_awaited()
    # verify user message saved
    mock_chat_service.add_message_to_history.assert_awaited()
    
    # Verify arun called with history
    mock_processor_instance.arun.assert_awaited_once() 

    # Find the call to print with the Markdown object
    found_markdown = False
    for call in mock_console_print.call_args_list:
        arg = call.args[0]
        if isinstance(arg, Markdown) and arg.markup == "Standalone one-shot response":
            found_markdown = True
            break
    assert found_markdown, "Markdown response was not printed"

    mock_processor_instance.close.assert_awaited_once()


def test_config_manager_migration(tmp_path) -> None:
    """Test that load_agents automatically migrates legacy configuration fields."""
    import yaml
    from rai import config_manager

    # Create a temporary agents.yaml file with legacy keys and tools
    agents_file = tmp_path / "agents.yaml"
    legacy_data = {
        "custom_agent": {
            "name": "custom_agent",
            "model": "gpt-4",
            "framework": "agno",
            "tools": [
                "CalculatorTools",
                "GnomeNotificationTool",
                "GnomeScreenshotTool",
                "GnomeWeatherTool",
                "WikipediaTools"
            ]
        }
    }

    with open(agents_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(legacy_data, f)

    # Call load_agents
    loaded = config_manager.load_agents(path=str(agents_file))

    # Assert in-memory migration
    assert "framework" not in loaded["custom_agent"]
    assert "DesktopNotificationTool" in loaded["custom_agent"]["tools"]
    assert "DesktopScreenshotTool" in loaded["custom_agent"]["tools"]
    assert "DesktopWeatherTool" in loaded["custom_agent"]["tools"]
    assert "GnomeNotificationTool" not in loaded["custom_agent"]["tools"]
    assert "GnomeScreenshotTool" not in loaded["custom_agent"]["tools"]
    assert "GnomeWeatherTool" not in loaded["custom_agent"]["tools"]

    # Assert on-disk migration (it should have written back to agents_file)
    with open(agents_file, "r", encoding="utf-8") as f:
        on_disk = yaml.safe_load(f)

    assert "framework" not in on_disk["custom_agent"]
    assert "DesktopNotificationTool" in on_disk["custom_agent"]["tools"]
    assert "DesktopScreenshotTool" in on_disk["custom_agent"]["tools"]
    assert "DesktopWeatherTool" in on_disk["custom_agent"]["tools"]

