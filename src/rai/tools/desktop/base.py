"""Abstract base class for desktop environment integrations."""
from abc import ABC, abstractmethod
from typing import Optional, Any

class DesktopAdapter(ABC):
    """Abstract base class representing Linux desktop tools interface."""

    def __deepcopy__(self, memo: Any) -> "DesktopAdapter":
        # Desktop adapters are service singletons/adapters and should not be deep-copied.
        # Returning self prevents recursive deep-copying of D-Bus / GLib objects (like MainLoop).
        return self

    @abstractmethod
    def send_notification(self, summary: str, body: str, app_name: str = "AI Assistant") -> str:
        """
        Sends a desktop notification.

        Args:
            summary (str): The bold title of the notification.
            body (str): The main text content of the notification.
            app_name (str): The name of the application sending the notification.

        Returns:
            str: A message indicating success or failure.
        """
        pass

    @abstractmethod
    def take_screenshot(self, delay: int = 0) -> str:
        """
        Takes a full-screen screenshot, encodes it in base64, and returns it.

        Args:
            delay (int): Delay in seconds before taking the screenshot.

        Returns:
            str: A JSON string containing either the base64-encoded image or an error.
        """
        pass

    @abstractmethod
    def weather(self, location: Optional[str] = "current_location") -> str:
        """
        Retrieves weather information from the desktop system or fallback API.

        Args:
            location (str | None): The name of the city to get weather for.

        Returns:
            str: A JSON string containing weather data or an error dict.
        """
        pass
