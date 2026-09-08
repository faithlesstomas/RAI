"""COSMIC desktop adapter implementing the DesktopAdapter interface."""
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.parse
from typing import Optional

try:
    import pydbus
    _HAS_DBUS = True
except ImportError:
    _HAS_DBUS = False

from rai.tools.desktop.base import DesktopAdapter

class CosmicDesktopAdapter(DesktopAdapter):
    """COSMIC desktop adapter leveraging Freedesktop D-Bus and Wayland utilities."""

    def send_notification(self, summary: str, body: str, app_name: str = "AI Assistant") -> str:
        """Sends a desktop notification via standard Freedesktop D-Bus."""
        if not _HAS_DBUS:
            try:
                command = shutil.which("notify-send")
                if command is None:
                    raise FileNotFoundError("notify-send is unavailable")
                # The executable is resolved to an absolute path; arguments are not shell-evaluated.
                subprocess.run(  # noqa: S603
                    [command, "-a", app_name, summary, body], check=True
                )
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
        """Takes a full-screen screenshot using cosmic-screenshot or grim on COSMIC Wayland."""
        filename = ""
        temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            temporary_dir = tempfile.TemporaryDirectory(prefix="rai-screenshot-")
            filename = os.path.join(temporary_dir.name, "screenshot.png")
            if delay > 0:
                time.sleep(delay)

            # 1. Attempt using cosmic-screenshot
            try:
                executable = shutil.which("cosmic-screenshot")
                if executable is None:
                    raise FileNotFoundError("cosmic-screenshot is unavailable")
                # The executable is resolved to an absolute path; arguments are not shell-evaluated.
                subprocess.run(  # noqa: S603
                    [executable, filename], capture_output=True, text=True, check=True
                )
            except Exception:  # pylint: disable=broad-except
                # 2. Attempt using grim (Wayland general)
                executable = shutil.which("grim")
                if executable is None:
                    raise FileNotFoundError("grim is unavailable")
                # The executable is resolved to an absolute path; arguments are not shell-evaluated.
                subprocess.run(  # noqa: S603
                    [executable, filename], capture_output=True, text=True, check=True
                )

            if not os.path.exists(filename):
                return json.dumps({
                    "status": "error",
                    "message": f"Screenshot file not found at: {filename} after capture."
                })

            with open(filename, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            return json.dumps({
                "type": "image_data",
                "format": "png",
                "base64": encoded_string
            })

        except Exception as e:  # pylint: disable=broad-except
            return json.dumps({
                "status": "error",
                "message": f"Failed to take screenshot: {e}. "
                           "Ensure cosmic-screenshot or grim is installed and available in PATH."
            })
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()

    def weather(self, location: Optional[str] = "current_location") -> str:
        """Retrieves weather info using online wttr.in fallback for COSMIC."""
        loc = location if location and location != "current_location" else "London"
        try:
            loc_encoded = urllib.parse.quote(loc)
            url = f"https://wttr.in/{loc_encoded}?format=j1"
            # The URL is constructed locally with a fixed HTTPS origin.
            req = urllib.request.Request(  # noqa: S310
                url, headers={'User-Agent': 'curl/7.81.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
                data = json.loads(response.read().decode('utf-8'))
                current = data['current_condition'][0]
                area = data['nearest_area'][0]
                weather_data = {
                    "location": f"{area['areaName'][0]['value']}, {area['country'][0]['value']}",
                    "temperature_c": float(current['temp_C']),
                    "feels_like_c": float(current['FeelsLikeC']),
                    "conditions": current['weatherDesc'][0]['value'],
                    "wind_speed_kmh": float(current['windspeedKmph']),
                    "pressure_hpa": float(current['pressure']),
                    "humidity_percent": float(current['humidity']),
                }
                return json.dumps(weather_data, indent=2)
        except Exception as e:  # pylint: disable=broad-except
            return json.dumps({
                "error": f"Failed to retrieve weather for '{loc}' from online fallback",
                "details": str(e)
            })
