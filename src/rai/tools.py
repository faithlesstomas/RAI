import os
import json
import datetime
import subprocess
from agno.tools import tool
import pydbus
from gi.repository import GLib

@tool
def send_notification(summary: str, body: str, app_name: str = "AI Assistant") -> str:
    """
    Sends a desktop notification to the GNOME (or Freedesktop.org compatible) notification system.

    Args:
        summary (str): The bold title of the notification.
        body (str): The main text content of the notification.
        app_name (str): The name of the application sending the notification. Defaults to "AI Assistant".

    Returns:
        str: A message indicating success or failure.
    """
    try:
        bus = pydbus.SessionBus()
        notifications = bus.get("org.freedesktop.Notifications", "/org/freedesktop/Notifications")
        notifications.Notify(
            app_name,
            0,  # replaces_id
            "", # app_icon
            summary,
            body,
            [], # actions
            {}, # hints
            -1  # expire_timeout (-1 for default)
        )
        return f"Notification sent: Summary='{summary}', Body='{body}'"
    except Exception as e:
        return f"Failed to send notification: {e}. Ensure a notification daemon is running."

@tool
def take_screenshot(delay: int = 0) -> str:
    """
    Takes a full-screen screenshot using the `gnome-screenshot` command-line tool.
    The screenshot is saved to a temporary file.

    Args:
        delay (int): Delay in seconds before taking the screenshot. Defaults to 0.

    Returns:
        str: The absolute path to the saved screenshot file on success, or an error message on failure.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("/tmp", f"screenshot_{timestamp}.png")
        command = ["/usr/bin/gnome-screenshot"]
        if delay > 0:
            command.extend(["-d", str(delay)])
        command.append(filename)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Screenshot saved to: {filename}"
        else:
            return f"Failed to take screenshot: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed to take screenshot: {e}. Ensure gnome-screenshot is installed and available in PATH."

# Inicjalizacja pętli zdarzeń GLib
loop = GLib.MainLoop()

@tool
def weather(location: str | None = "current_location") -> str:
    """
    Retrieves current weather information directly from the GNOME desktop system.

    This tool is ideal for questions about the weather at the user's current location
    or other locations configured in their GNOME settings. It provides system-specific data.

    Args:
        location (str | None): The name of the city to get weather for. Defaults to "current_location"
                               to fetch weather for the primary location configured in GNOME.

    Returns:
        str: A JSON string containing a dictionary with weather data on success,
             or a dictionary with an "error" key on failure.
             Example success: '{"location": "Warsaw", "temperature_c": 15, "conditions": "Clear"}'
             Example error:   '{"error": "Location not found."}'
    """
    bus = pydbus.SessionBus()
    if not location:
        location = "current_location"
    try:
        weather_service = bus.get('org.gnome.Shell.Weather')
        weather_info_variant = weather_service.GetWeatherInfo()

        if not weather_info_variant:
            return json.dumps({"error": "No weather information configured in the GNOME system."})

        found_location_info = None
        if location == "current_location":
            if weather_info_variant:
                first_location_id = list(weather_info_variant.keys())[0]
                found_location_info = weather_info_variant[first_location_id]
        else:
            for loc_id, info in weather_info_variant.items():
                if info.get('name', '').lower() == location.lower():
                    found_location_info = info
                    break

        if not found_location_info:
            available_locations = [info.get('name', 'Unknown') for info in weather_info_variant.values()]
            return json.dumps({
                "error": f"Weather information not found for location: {location}.",
                "available_locations": available_locations
            })

        info = found_location_info
        weather_data = {
            "location": info.get('name'),
            "temperature_c": info.get('main', {}).get('temp'),
            "feels_like_c": info.get('main', {}).get('temp_feels_like'),
            "conditions": info.get('description'),
            "wind_speed_kmh": info.get('wind', {}).get('speed'),
            "pressure_hpa": info.get('main', {}).get('pressure'),
            "humidity_percent": info.get('main', {}).get('humidity'),
        }
        return json.dumps(weather_data, indent=2)

    except Exception as e:
        return json.dumps({
            "error": "An error occurred while retrieving weather from GNOME.",
            "details": str(e),
            "suggestion": "Ensure you are running a GNOME desktop environment and that the 'org.gnome.Shell.Weather' service is available via DBus."
        })
