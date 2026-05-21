"""GNOME-specific desktop adapter implementing the DesktopAdapter interface."""
import base64
import datetime
import json
import os
import subprocess
from typing import Optional

try:
    import pydbus
    from gi.repository import GLib
    _HAS_DBUS = True
except ImportError:
    _HAS_DBUS = False

from rai.tools.desktop.base import DesktopAdapter

class GnomeDesktopAdapter(DesktopAdapter):
    """GNOME desktop adapter leveraging standard Freedesktop/GNOME D-Bus interfaces."""

    def __init__(self) -> None:
        if _HAS_DBUS:
            try:
                self.loop = GLib.MainLoop()
            except Exception:  # pylint: disable=broad-except
                self.loop = None
        else:
            self.loop = None

    def send_notification(self, summary: str, body: str, app_name: str = "AI Assistant") -> str:
        """Sends a desktop notification via standard Freedesktop D-Bus."""
        if not _HAS_DBUS:
            # Command fallback using notify-send if pydbus/GLib is missing
            try:
                subprocess.run(["notify-send", "-a", app_name, summary, body], check=True)
                return f"Notification sent via notify-send: Summary='{summary}', Body='{body}'"
            except Exception as e:  # pylint: disable=broad-except
                return f"Failed to send notification: {e}. No D-Bus or notify-send available."

        try:
            bus = pydbus.SessionBus()
            notifications = bus.get(
                "org.freedesktop.Notifications", "/org/freedesktop/Notifications"
            )
            notifications.Notify(
                app_name,
                0,  # replaces_id
                "",  # app_icon
                summary,
                body,
                [],  # actions
                {},  # hints
                -1,  # expire_timeout (-1 for default)
            )
            return f"Notification sent: Summary='{summary}', Body='{body}'"
        except Exception as e:  # pylint: disable=broad-except
            return (
                f"Failed to send notification: {e}. "
                "Ensure a notification daemon is running."
            )

    def take_screenshot(self, delay: int = 0) -> str:
        """Takes a full-screen screenshot using gnome-screenshot command utility."""
        filename = ""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("/tmp", f"screenshot_{timestamp}.png")
            command = ["/usr/bin/gnome-screenshot", "--file", filename]
            if delay > 0:
                command.extend(["-d", str(delay)])

            subprocess.run(command, capture_output=True, text=True, check=True)

            if not os.path.exists(filename):
                return json.dumps({
                    "status": "error",
                    "message": f"Screenshot file not found at: {filename} after capture."
                })

            with open(filename, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            os.remove(filename)

            return json.dumps({
                "type": "image_data",
                "format": "png",
                "base64": encoded_string
            })

        except Exception as e:  # pylint: disable=broad-except
            if filename and os.path.exists(filename):
                os.remove(filename)
            return json.dumps({
                "status": "error",
                "message": f"Failed to take screenshot: {e}. "
                           "Ensure gnome-screenshot is installed and available in PATH."
            })

    def weather(self, location: Optional[str] = "current_location") -> str:
        """Retrieves weather information from the GNOME Shell Weather DBus service."""
        if not _HAS_DBUS:
            return json.dumps({
                "error": "D-Bus libraries not available to query GNOME Weather."
            })

        if not location:
            location = "current_location"

        try:
            bus = pydbus.SessionBus()
            weather_service = bus.get("org.gnome.Shell.Weather")
            weather_info_variant = weather_service.GetWeatherInfo()

            if not weather_info_variant:
                return json.dumps(
                    {"error": "No weather information configured in the GNOME system."}
                )

            found_location_info = None
            if location == "current_location":
                if weather_info_variant:
                    first_location_id = list(weather_info_variant.keys())[0]
                    found_location_info = weather_info_variant[first_location_id]
            else:
                for _, info in weather_info_variant.items():
                    if info.get("name", "").lower() == location.lower():
                        found_location_info = info
                        break

            if not found_location_info:
                available_locations = [
                    info.get("name", "") for info in weather_info_variant.values()
                ]
                return json.dumps(
                    {
                        "error": f"Weather for {location} not found.",
                        "available_locations": available_locations,
                    }
                )

            info = found_location_info
            weather_data = {
                "location": info.get("name"),
                "temperature_c": info.get("main", {}).get("temp"),
                "feels_like_c": info.get("main", {}).get("temp_feels_like"),
                "conditions": info.get("description"),
                "wind_speed_kmh": info.get("wind", {}).get("speed"),
                "pressure_hpa": info.get("main", {}).get("pressure"),
                "humidity_percent": info.get("main", {}).get("humidity"),
            }
            return json.dumps(weather_data, indent=2)

        except Exception as e:  # pylint: disable=broad-except
            return json.dumps(
                {
                    "error": "An error occurred while retrieving weather from GNOME.",
                    "details": str(e),
                    "suggestion": (
                        "This tool requires a GNOME desktop environment with the "
                        "'org.gnome.Shell.Weather' service available via DBus. "
                        "If you are not in a GNOME environment, this tool will not work."
                    ),
                }
            )
