"""COSMIC desktop adapter implementing the DesktopAdapter interface."""
import base64
import datetime
import json
import os
import subprocess
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
        """Takes a full-screen screenshot using cosmic-screenshot or grim on COSMIC Wayland."""
        filename = ""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("/tmp", f"screenshot_{timestamp}.png")
            
            # 1. Attempt using cosmic-screenshot
            try:
                command = ["cosmic-screenshot", filename]
                if delay > 0:
                    command = ["sleep", str(delay), "&&", "cosmic-screenshot", filename]
                    subprocess.run(" ".join(command), shell=True, check=True)
                else:
                    subprocess.run(command, capture_output=True, text=True, check=True)
            except Exception:  # pylint: disable=broad-except
                # 2. Attempt using grim (Wayland general)
                command = ["grim", filename]
                if delay > 0:
                    command = ["sleep", str(delay), "&&", "grim", filename]
                    subprocess.run(" ".join(command), shell=True, check=True)
                else:
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
                           "Ensure cosmic-screenshot or grim is installed and available in PATH."
            })

    def weather(self, location: Optional[str] = "current_location") -> str:
        """Retrieves weather info using online wttr.in fallback for COSMIC."""
        loc = location if location and location != "current_location" else "London"
        try:
            loc_encoded = urllib.parse.quote(loc)
            url = f"https://wttr.in/{loc_encoded}?format=j1"
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.81.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
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
