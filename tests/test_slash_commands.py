"""Tests for the slash commands in rai.cli."""
from unittest.mock import MagicMock, patch, mock_open

from rich.panel import Panel

from rai import cli as rai_cli


def test_handle_history_command_with_history():
    """Test the /history command when history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_history_command([], {}, mock_processor)

        assert mock_print.call_count == 2
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


def test_handle_history_command_with_no_history():
    """Test the /history command when no history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history.return_value = []

    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_history_command([], {}, mock_processor)

        mock_print.assert_called_once_with("[dim]No history available.[/dim]")


def test_handle_save_command_with_history():
    """Test the /save command when history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    m = mock_open()
    with patch("builtins.open", m):
        with patch("rai.cli.console.print") as mock_print:
            rai_cli._handle_save_command(["test.md"], {}, mock_processor)

            m.assert_called_once_with("test.md", "w", encoding="utf-8")
            handle = m()
            handle.write.assert_any_call("**User**\n\nHello\n\n---\n\n")
            handle.write.assert_any_call("**Assistant**\n\nHi there!\n\n---\n\n")
            mock_print.assert_called_once_with("[green]Conversation saved to test.md[/green]")


def test_handle_save_command_no_history():
    """Test the /save command when no history is available."""
    mock_processor = MagicMock()
    mock_processor.get_history.return_value = []

    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_save_command(["test.md"], {}, mock_processor)

        mock_print.assert_called_once_with("[dim]No history available to save.[/dim]")


def test_handle_save_command_no_filename():
    """Test the /save command when no filename is provided."""
    mock_processor = MagicMock()
    mock_processor.get_history.return_value = [{"role": "user", "content": "Hello"}]

    with patch("rai.cli.error_console.print") as mock_print:
        rai_cli._handle_save_command([], {}, mock_processor)

        mock_print.assert_called_once_with("[red]Usage: /save <filename.md>[/red]")


def test_handle_clear_command():
    """Test the /clear command."""
    mock_processor = MagicMock()
    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_clear_command([], {}, mock_processor)

        mock_processor.clear_history.assert_called_once()
        mock_print.assert_called_once_with("[green]Chat history cleared.[/green]")


def test_handle_model_command_show_current_model():
    """Test the /model command to show the current model."""
    run_config = {"model": "current-model"}
    mock_processor = MagicMock()
    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_model_command([], run_config, mock_processor)

        mock_print.assert_called_once_with("Current model: current-model")


def test_handle_model_command_set_new_model():
    """Test the /model command to set a new model."""
    run_config = {"model": "old-model"}
    mock_processor = MagicMock()
    with patch("rai.cli.console.print") as mock_print:
        rai_cli._handle_model_command(["new-model"], run_config, mock_processor)

        assert run_config["model"] == "new-model"
        mock_print.assert_any_call("Temporarily set model to 'new-model' for this session.")
        mock_print.assert_any_call("[dim]Note: New settings will be used on the next interaction.[/dim]")