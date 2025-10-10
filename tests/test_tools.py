"""Tests for the rai.tools module."""
import json
from subprocess import CalledProcessError
from unittest.mock import MagicMock, patch

import pytest
from rai.tools import send_notification, take_screenshot, weather


@pytest.fixture(autouse=True)
def mock_glib_mainloop():
    """Fixture to mock GLib.MainLoop."""
    with patch('rai.tools.GLib.MainLoop'):
        yield


class TestSendNotification:
    """Tests for the send_notification tool."""

    @patch('rai.tools.pydbus.SessionBus')
    def test_send_notification_success(self, mock_session_bus):
        """Test successful notification sending."""
        mock_bus = MagicMock()
        mock_session_bus.return_value = mock_bus
        mock_notifications = MagicMock()
        mock_bus.get.return_value = mock_notifications

        summary = "Test Summary"
        body = "Test Body"
        result = send_notification.entrypoint(summary=summary, body=body)

        mock_bus.get.assert_called_once_with(
            "org.freedesktop.Notifications", "/org/freedesktop/Notifications"
        )
        mock_notifications.Notify.assert_called_once()
        assert f"Notification sent: Summary='{summary}', Body='{body}'" in result

    @patch('rai.tools.pydbus.SessionBus', side_effect=Exception("DBus error"))
    def test_send_notification_failure(self, _mock_session_bus):
        """Test failure in sending notification due to DBus error."""
        result = send_notification.entrypoint(summary="Test", body="Test")
        assert "Failed to send notification: DBus error." in result


class TestTakeScreenshot:
    """Tests for the take_screenshot tool."""

    @patch('rai.tools.os.remove')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('rai.tools.base64.b64encode')
    @patch('rai.tools.os.path.exists', return_value=True)
    @patch('rai.tools.subprocess.run')
    @patch('rai.tools.datetime')
    def test_success(self, mock_dt, mock_run, _mock_exists,
                     mock_b64, mock_open, mock_remove):
        """Test successful screenshot capture and base64 encoding."""
        mock_dt.datetime.now.strftime.return_value = "20230101_120000"
        mock_b64.return_value.decode.return_value = "fake_base64_string"

        with patch('rai.tools.os.path.join', return_value='/tmp/s.png'):
            result = take_screenshot.entrypoint(delay=0)
            result_json = json.loads(result)

            mock_run.assert_called_once_with(
                ["/usr/bin/gnome-screenshot", "--file", "/tmp/s.png"],
                capture_output=True, text=True, check=True
            )
            mock_open.assert_called_once_with("/tmp/s.png", "rb")
            mock_b64.assert_called_once()
            mock_remove.assert_called_once_with("/tmp/s.png")
            assert result_json["type"] == "image_data"
            assert result_json["base64"] == "fake_base64_string"

    @patch('rai.tools.subprocess.run')
    def test_capture_fails(self, mock_run):
        """Test failure when the screenshot command itself fails."""
        mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")
        result = take_screenshot.entrypoint(delay=0)
        result_json = json.loads(result)
        assert result_json["status"] == "error"
        assert "Failed to take screenshot" in result_json["message"]

    @patch('rai.tools.os.path.exists', return_value=False)
    @patch('rai.tools.subprocess.run')
    def test_file_not_found(self, _mock_run, _mock_exists):
        """Test failure when the screenshot file is not found after capture."""
        result = take_screenshot.entrypoint(delay=0)
        result_json = json.loads(result)
        assert result_json["status"] == "error"
        assert "Screenshot file not found" in result_json["message"]


class TestWeather:
    """Tests for the weather tool."""

    @patch('rai.tools.pydbus.SessionBus')
    def test_dbus_error(self, mock_session_bus):
        """Test weather tool with a DBus connection error."""
        mock_session_bus.side_effect = Exception("DBus connection error")
        with pytest.raises(Exception, match="DBus connection error"):
            weather.entrypoint(location="Warsaw")