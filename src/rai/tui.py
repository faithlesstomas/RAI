# pylint: disable=no-name-in-module,import-error
"""A basic Textual app for Rai."""

import io
import json
from contextlib import redirect_stdout

import click
from agno.media import Image
from ollama import ResponseError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Input, LoadingIndicator, RichLog, Static

from rai.cli import check_model_tool_support
from rai.config_screen import ConfigScreen
from rai.core import RAI_CONFIG, setup_agent

console = Console()


class RaiTUI(App[None]):
    """A basic Textual app for Rai."""

    SCREENS = {"config": ConfigScreen}

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
        ("c", "push_screen('config')", "Configure"),
    ]

    CSS = """
    #chat-area {
        height: 1fr; /* Take all available vertical space */
        layout: vertical;
    }
    #output {
        height: 1fr; /* RichLog takes all available space within chat-area */
        border: round $panel;
        margin: 1;
    }
    #input {
        height: auto; /* Input takes only as much height as needed */
        dock: bottom;
        margin: 1;
    }
    #status_bar {
        height: auto; /* Status bar takes only as much height as needed */
        dock: bottom; /* Dock to the bottom */
        background: $panel;
        color: $text;
        padding: 0 1;
    }
    #loading_indicator {
        dock: top;
        height: 1;
        width: 100%;
        background: $accent;
        layer: overlay;
    }
    .-hidden {
        display: none;
    }
    """

    def __init__(
        self,
        model_id: str,
        backend: str,
        system_prompt: str,
        no_markdown: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.backend = backend
        self.system_prompt = system_prompt
        self.no_markdown = no_markdown
        self.agent = None

    def reinitialize_agent(self) -> None:
        """Reinitializes the AI agent with the current configuration."""
        self._write_message(
            "Reinitializing AI agent with new configuration...", message_type="info"
        )
        self.agent = None  # Clear existing agent

        has_tools = True
        if self.backend == "ollama":
            has_tools = check_model_tool_support(self.model_id)
            if not has_tools:
                self._write_message(
                    f"[yellow]Warning: Model '{self.model_id}' may not support tools. "
                    "Proceeding in text-only mode.[/yellow]"
                )

        self.query_one("#status_bar", Static).update(
            f"Model: {self.model_id} | Backend: {self.backend} | "
            f"Tools: {'Enabled' if has_tools else 'Disabled'}"
        )
        try:
            # Setup the global config that setup_agent uses

            RAI_CONFIG.update({
                "system": self.system_prompt,
                "model": self.model_id,
                "backend": self.backend,
                "tools": None, # Let setup_agent decide the default tools
            })

            agent, messages = setup_agent(
                enable_tools=has_tools,
                use_markdown=not self.no_markdown,
            )
            self.agent = agent

            for msg in messages:
                self._write_message(msg, message_type="info")

            self._write_message(
                "AI agent reinitialized successfully.", message_type="info"
            )
        except Exception as e:
            self._write_message(
                f"Failed to reinitialize agent: {e}", message_type="error"
            )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="chat-area"):
            yield RichLog(id="output")
            yield Input(placeholder="Type your message here...", id="input")
            yield LoadingIndicator(id="loading_indicator", classes="-hidden")
        yield Static("", id="status_bar")

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.query_one("#input", Input).focus()
        self.reinitialize_agent()
        self._write_message("AI agent initialized. Type your query below.", "info")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Called when the user submits text in the input field."""
        user_message = message.value
        message.input.value = ""

        if not user_message.strip():
            return

        self._write_message(user_message, message_type="user")
        await self._process_query(user_message)

    async def _process_query(self, prompt: str) -> None:
        """
        Executes a single query, handling potential image data from tools,
        and updates TUI.
        """
        self.query_one("#loading_indicator", LoadingIndicator).remove_class("-hidden")

        try:
            string_io = io.StringIO()
            with redirect_stdout(string_io):
                initial_response = self.agent.run(prompt, stream=False)

            tool_output_str = string_io.getvalue().strip()
            image_data = None

            if tool_output_str:
                panel_title = "[bold yellow]Tool Call[/bold yellow]"
                panel_content = "Screenshot captured successfully."
                try:
                    tool_json = json.loads(tool_output_str)
                    if tool_json.get("type") == "image_data":
                        image_data = tool_json
                    else:
                        panel_content = tool_output_str
                except (json.JSONDecodeError, AttributeError):
                    panel_content = tool_output_str

                self._write_message(
                    Panel(panel_content, title=panel_title, border_style="yellow"),
                    message_type="tool_call",
                )

            if image_data:
                self._write_message(
                    "\n[bold]AI Assistant (analyzing image):[/bold]",
                    message_type="info",
                )
                img = Image(
                    source=image_data.get("base64"),
                    mime_type=f"image/{image_data.get('format', 'png')}",
                )
                response_stream = await self.agent.run(
                    prompt, images=[img], stream=True
                )
                for response_chunk in response_stream:
                    if response_chunk.content:
                        self._write_message(
                            Markdown(response_chunk.content)
                            if not self.no_markdown
                            else response_chunk.content,
                            message_type="ai",
                        )
            elif initial_response and initial_response.content:
                self._write_message(
                    Markdown(initial_response.content)
                    if not self.no_markdown
                    else initial_response.content,
                    message_type="ai",
                )

            self._write_message("", message_type="info")

        except ResponseError as e:
            self._write_message(
                f"Ollama API error (status: {e.status_code}): {e.error}",
                message_type="error",
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._write_message(
                f"An unexpected error occurred: {e}", message_type="error"
            )
        finally:
            self.query_one("#loading_indicator", LoadingIndicator).add_class("-hidden")

    def _write_message(
        self,
        content: str | Panel | Markdown,
        message_type: str = "info",
    ) -> None:
        """Writes a styled message to the RichLog."""
        output_log = self.query_one("#output", RichLog)
        output_log.write("")

        message_formats = {
            "user": f"[dim]{content}[/dim]",
            "error": f"[bold red]ERROR:[/bold red] {content}",
        }

        if message_type in message_formats:
            output_log.write(message_formats[message_type])
        else:
            output_log.write(content)


def app(model_id: str, backend: str, system_prompt: str, no_markdown: bool) -> None:
    """Entry point for the Textual app."""
    RaiTUI(model_id, backend, system_prompt, no_markdown).run()


@click.command()
@click.option(
    "-s",
    "--system",
    default="You are a versatile and helpful AI assistant.",
    help="Defines the system prompt for the AI.",
)
@click.option(
    "-m",
    "--model",
    default="gemma3:1b",
    help="ID of the Ollama model to be used (e.g., gemma2:9b, llama3.2).",
)
@click.option(
    "-b",
    "--backend",
    default="ollama",
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"]),
    help="The LLM backend to use.",
)
@click.option(
    "--no-markdown",
    is_flag=True,
    help="Disable Markdown rendering for LLM responses.",
)
def run_tui_cli(system, model, backend, no_markdown) -> None:
    """Run the Rai TUI application."""
    app(model, backend, system, no_markdown)
