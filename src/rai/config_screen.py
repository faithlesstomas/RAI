# pylint: disable=no-name-in-module,import-error
"""Configuration screen for the TUI."""

from textual.app import ComposeResult  # pylint: disable=no-name-in-module
from textual.screen import Screen  # pylint: disable=import-error
from textual.widgets import Button, Footer, Header, Input, Label, Select
from textual.containers import Container, VerticalScroll  # pylint: disable=import-error

class ConfigScreen(Screen):
    """A screen to configure the AI agent."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back to Chat"),
        ("s", "save_config", "Save Config"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        with Container(id="config-grid"):
            with VerticalScroll(id="config-form"):
                yield Label("[b]Model Configuration[/b]", classes="config-section-title")
                yield Label("Model ID:")
                yield Input(id="model_id_input", placeholder="e.g., gemma3:1b")
                yield Label("Backend:")
                yield Select(
                    [
                        ("Ollama", "ollama"),
                        ("Gemini", "gemini"),
                        ("Anthropic", "anthropic"),
                        ("OpenAI", "openai"),
                        ("Groq", "groq"),
                    ],
                    id="backend_select",
                )
                yield Label("\n[b]System Prompt[/b]", classes="config-section-title")
                yield Input(id="system_prompt_input", placeholder="You are a helpful AI assistant.")
                yield Button("Save Configuration", id="save_config_button", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        # Populate current values from the app
        self.query_one("#model_id_input", Input).value = self.app.model_id
        self.query_one("#backend_select", Select).value = self.app.backend
        self.query_one("#system_prompt_input", Input).value = self.app.system_prompt

    def action_save_config(self) -> None:
        """Save the configuration and return to the chat screen."""
        new_model_id = self.query_one("#model_id_input", Input).value
        new_backend = self.query_one("#backend_select", Select).value
        new_system_prompt = self.query_one("#system_prompt_input", Input).value

        # Update the app's state
        self.app.model_id = new_model_id  # pylint: disable=attribute-defined-outside-init
        self.app.backend = new_backend  # pylint: disable=attribute-defined-outside-init
        self.app.system_prompt = new_system_prompt  # pylint: disable=attribute-defined-outside-init

        # Re-initialize the agent in the main app
        self.app.reinitialize_agent()

        self.app.pop_screen() # Go back to the chat screen
