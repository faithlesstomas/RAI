from unittest.mock import MagicMock, patch
import json
from subprocess import CalledProcessError

# Import the tools from the src.rai.tools module
from rai.tools import send_notification, take_screenshot, weather


class TestSendNotification:
    @patch('rai.tools.pydbus.SessionBus')
    def test_send_notification_success(self, mock_session_bus):
        # Setup mocks
        mock_notifications = MagicMock()
        mock_session_bus.return_value.get.return_value = mock_notifications

        summary = "Test Summary"
        body = "Test Body"
        result = send_notification.entrypoint(summary, body)

        # Assertions
        mock_session_bus.return_value.get.assert_called_once_with(
            "org.freedesktop.Notifications", "/org/freedesktop/Notifications"
        )
        mock_notifications.Notify.assert_called_once_with(
            "AI Assistant", 0, "", summary, body, [], {}, -1
        )
        assert "Notification sent" in result

    @patch('rai.tools.pydbus.SessionBus')
    def test_send_notification_failure(self, mock_session_bus):
        # Simulate an exception during DBus call
        mock_session_bus.return_value.get.side_effect = Exception("DBus error")

        summary = "Test Summary"
        body = "Test Body"
        result = send_notification.entrypoint(summary, body)

        # Assertions
        assert "Failed to send notification" in result
        assert "DBus error" in result


class TestTakeScreenshot:
    @patch('rai.tools.subprocess.run')
    @patch('rai.tools.os.path.join', return_value='/tmp/screenshot_test.png')
    @patch('rai.tools.datetime')
    def test_take_screenshot_success(self, mock_datetime, mock_join, mock_subprocess_run):
        # Setup mocks
        mock_datetime.datetime.now.strftime.return_value = "20250816_120000"
        mock_subprocess_run.return_value.returncode = 0
        mock_subprocess_run.return_value.stdout = "/tmp/screenshot_test.png\n"
        mock_subprocess_run.return_value.stderr = ""

        result = take_screenshot.entrypoint(delay=0)

        # Assertions
        mock_subprocess_run.assert_called_once_with(
            ["/usr/bin/gnome-screenshot", "/tmp/screenshot_test.png"],
            capture_output=True, text=True, check=True
        )
        assert "Screenshot saved to: /tmp/screenshot_test.png" in result

    @patch('rai.tools.subprocess.run')
    @patch('rai.tools.os.path.join', return_value='/tmp/screenshot_test.png')
    @patch('rai.tools.datetime')
    def test_take_screenshot_with_delay(self, mock_datetime, mock_join, mock_subprocess_run):
        # Setup mocks
        mock_datetime.datetime.now.strftime.return_value = "20250816_120000"
        mock_subprocess_run.return_value.returncode = 0
        mock_subprocess_run.return_value.stdout = "/tmp/screenshot_test.png\n"
        mock_subprocess_run.return_value.stderr = ""

        result = take_screenshot.entrypoint(delay=5)

        # Assertions
        mock_subprocess_run.assert_called_once_with(
            ["/usr/bin/gnome-screenshot", "-d", "5", "/tmp/screenshot_test.png"],
            capture_output=True, text=True, check=True
        )
        assert "Screenshot saved to: /tmp/screenshot_test.png" in result

    @patch('rai.tools.subprocess.run')
    @patch('rai.tools.os.path.join', return_value='/tmp/screenshot_test.png')
    @patch('rai.tools.datetime')
    def test_take_screenshot_failure(self, mock_datetime, mock_join, mock_subprocess_run):
        # Simulate a failed subprocess run
        mock_datetime.datetime.now.strftime.return_value = "20250816_120000"
        mock_subprocess_run.side_effect = CalledProcessError(1, "gnome-screenshot", stderr="Error taking screenshot")

        result = take_screenshot.entrypoint(delay=0)

        # Assertions
        assert "Failed to take screenshot" in result
        assert "Command 'gnome-screenshot' returned non-zero exit status 1" in result


class TestWeather:
    @patch('rai.tools.pydbus.SessionBus')
    def test_weather_success_current_location(self, mock_session_bus):
        # Setup mocks for successful weather retrieval
        mock_weather_service = MagicMock()
        mock_session_bus.return_value.get.return_value = mock_weather_service
        mock_weather_service.GetWeatherInfo.return_value = {
            'loc1': {
                'name': 'Warsaw',
                'main': {'temp': 15, 'temp_feels_like': 14},
                'description': 'Clear sky',
                'wind': {'speed': 10},
                'pressure': 1012,
                'humidity': 70
            }
        }

        result = weather.entrypoint(location="current_location")
        parsed_result = json.loads(result)

        # Assertions
        mock_session_bus.return_value.get.assert_called_once_with('org.gnome.Shell.Weather')
        mock_weather_service.GetWeatherInfo.assert_called_once()
        assert parsed_result['location'] == 'Warsaw'
        assert parsed_result['temperature_c'] == 15
        assert parsed_result['conditions'] == 'Clear sky'

    @patch('rai.tools.pydbus.SessionBus')
    def test_weather_success_specific_location(self, mock_session_bus):
        # Setup mocks for successful weather retrieval for a specific location
        mock_weather_service = MagicMock()
        mock_session_bus.return_value.get.return_value = mock_weather_service
        mock_weather_service.GetWeatherInfo.return_value = {
            'loc1': {'name': 'Warsaw', 'main': {}, 'description': '', 'wind': {}, 'pressure': 0, 'humidity': 0},
            'loc2': {
                'name': 'London',
                'main': {'temp': 10, 'temp_feels_like': 8},
                'description': 'Cloudy',
                'wind': {'speed': 5},
                'pressure': 1000,
                'humidity': 85
            }
        }

        result = weather.entrypoint(location="London")
        parsed_result = json.loads(result)

        # Assertions
        assert parsed_result['location'] == 'London'
        assert parsed_result['temperature_c'] == 10
        assert parsed_result['conditions'] == 'Cloudy'

    @patch('rai.tools.pydbus.SessionBus')
    def test_weather_no_info_configured(self, mock_session_bus):
        # Simulate no weather info configured in GNOME
        mock_weather_service = MagicMock()
        mock_session_bus.return_value.get.return_value = mock_weather_service
        mock_weather_service.GetWeatherInfo.return_value = None

        result = weather.entrypoint(location="current_location")
        parsed_result = json.loads(result)

        # Assertions
        assert "error" in parsed_result
        assert "No weather information configured" in parsed_result["error"]

    @patch('rai.tools.pydbus.SessionBus')
    def test_weather_location_not_found(self, mock_session_bus):
        # Simulate location not found
        mock_weather_service = MagicMock()
        mock_session_bus.return_value.get.return_value = mock_weather_service
        mock_weather_service.GetWeatherInfo.return_value = {
            'loc1': {'name': 'Warsaw', 'main': {}, 'description': '', 'wind': {}, 'pressure': 0, 'humidity': 0}
        }

        result = weather.entrypoint(location="NonExistentCity")
        parsed_result = json.loads(result)

        # Assertions
        assert "error" in parsed_result
        assert "Weather information not found" in parsed_result["error"]
        assert "available_locations" in parsed_result

    @patch('rai.tools.pydbus.SessionBus')
    def test_weather_exception_handling(self, mock_session_bus):
        # Simulate an exception during weather service call
        mock_session_bus.return_value.get.side_effect = Exception("DBus connection error")

        result = weather.entrypoint(location="current_location")
        parsed_result = json.loads(result)

        # Assertions
        assert "error" in parsed_result
        assert "An error occurred while retrieving weather from GNOME" in parsed_result["error"]
        assert "DBus connection error" in parsed_result["details"]
