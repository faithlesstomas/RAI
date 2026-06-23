"""Tests for the slash commands in rai.cli."""
from unittest.mock import MagicMock, patch, mock_open, AsyncMock
import pytest
from rich.panel import Panel

from rai import cli as rai_cli


@pytest.mark.asyncio
async def test_handle_history_command_with_history() -> None:
    """Test the /history command when history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])

    # No chat_service in config, fallbacks to processor.get_history
    run_config = {}

    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_history_command([], run_config, mock_processor)

        assert mock_print.call_count == 2 # noqa: PLR2004
        # Check the first call (user message)
        panel_user = mock_print.call_args_list[0].args[0]
        assert isinstance(panel_user, Panel)
        assert panel_user.title == "User"
        assert panel_user.renderable == "Hello"

        # Check the second call (assistant message)
        panel_assistant = mock_print.call_args_list[1].args[0]
        assert isinstance(panel_assistant, Panel)
        assert panel_assistant.title == "Assistant"
        assert panel_assistant.renderable == "Hi there!"


@pytest.mark.asyncio
async def test_handle_history_command_with_no_history() -> None:
    """Test the /history command when no history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history = AsyncMock(return_value=[])
    run_config = {}

    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_history_command([], run_config, mock_processor)

        mock_print.assert_called_once_with("[dim]No history available.[/dim]")


@pytest.mark.asyncio
async def test_handle_save_command_with_history() -> None:
    """Test the /save command when history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    run_config = {}
    m = mock_open()
    with patch("builtins.open", m):
        with patch("rai.cli.console.print") as mock_print:
            await rai_cli._handle_save_command(["test.md"], run_config, mock_processor)

            m.assert_called_once_with("test.md", "w", encoding="utf-8")
            handle = m()
            handle.write.assert_any_call("**User**\n\nHello\n\n---\n\n")
            handle.write.assert_any_call("**Assistant**\n\nHi there!\n\n---\n\n")
            mock_print.assert_called_once_with("[green]Conversation saved to test.md[/green]")


@pytest.mark.asyncio
async def test_handle_save_command_no_history() -> None:
    """Test the /save command when no history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history = AsyncMock(return_value=[])
    run_config = {}

    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_save_command(["test.md"], run_config, mock_processor)

        mock_print.assert_called_once_with("[dim]No history available to save.[/dim]")


@pytest.mark.asyncio
async def test_handle_save_command_no_filename() -> None:
    """Test the /save command when no filename is provided."""
    mock_processor = MagicMock()
    mock_processor.get_history = AsyncMock(return_value=[{"role": "user", "content": "Hello"}])
    run_config = {}

    with patch("rai.cli.error_console.print") as mock_print:
        await rai_cli._handle_save_command([], run_config, mock_processor)

        mock_print.assert_called_once_with("[red]Usage: /save <filename.md>[/red]")


@pytest.mark.asyncio
async def test_handle_clear_command() -> None:
    """Test the /clear command."""
    mock_processor = MagicMock()
    mock_processor.clear_history = AsyncMock()
    run_config = {}
    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_clear_command([], run_config, mock_processor)

        mock_processor.clear_history.assert_awaited_once()
        mock_print.assert_called_once_with("[green]Chat history cleared.[/green]")


@pytest.mark.asyncio
async def test_handle_model_command_show_current_model() -> None:
    """Test the /model command to show the current model."""
    run_config = {"model": "current-model"}
    mock_processor = MagicMock()
    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_model_command([], run_config, mock_processor)

        mock_print.assert_called_once_with("Current model: current-model")


@pytest.mark.asyncio
async def test_handle_model_command_set_new_model() -> None:
    """Test the /model command to set a new model."""
    run_config = {"model": "old-model"}
    mock_processor = MagicMock()
    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_model_command(["new-model"], run_config, mock_processor)

        assert run_config["model"] == "new-model"
        mock_print.assert_any_call("Set model to 'new-model' (persisted).")
        mock_print.assert_any_call("[dim]Note: New settings will be used on the next interaction.[/dim]")


@pytest.mark.asyncio
async def test_handle_config_command_filtering() -> None:
    """Test that /config show filters out internal objects like chat_service."""
    mock_processor = MagicMock()
    run_config = {
        "model": "test-model",
        "chat_service": MagicMock(),  # Not JSON serializable
        "active_tts_task": MagicMock(), # Not JSON serializable
        "other_setting": "value"
    }

    with patch("rai.cli.console.print") as mock_print:
        await rai_cli._handle_config_command(["show"], run_config, mock_processor)

        # Check that json.dumps was called without error
        # We can check the arguments to print
        assert mock_print.called
        panel = mock_print.call_args[0][0]
        assert isinstance(panel, Panel)
        # The content should be a JSON string
        import json
        content = json.loads(panel.renderable)
        assert content["model"] == "test-model"
        assert content["other_setting"] == "value"
        assert "chat_service" not in content
        assert "active_tts_task" not in content