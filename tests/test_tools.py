"""Tests for the rai.tools.desktop module."""
import json
import os
from subprocess import CalledProcessError
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from rai.tools.desktop import get_desktop_adapter
from rai.tools.desktop.base import DesktopAdapter
from rai.tools.desktop.gnome import GnomeDesktopAdapter
from rai.tools.desktop.cosmic import CosmicDesktopAdapter


def test_factory_detection() -> None:
    """Test that the dynamic desktop adapter factory resolves correctly based on XDG_CURRENT_DESKTOP."""
    # Reset cached active adapter
    from rai.tools.desktop import _ACTIVE_ADAPTER
    with patch("rai.tools.desktop._ACTIVE_ADAPTER", None):
        with patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "COSMIC"}):
            adapter = get_desktop_adapter()
            assert isinstance(adapter, CosmicDesktopAdapter)

    with patch("rai.tools.desktop._ACTIVE_ADAPTER", None):
        with patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "GNOME"}):
            adapter = get_desktop_adapter()
            assert isinstance(adapter, GnomeDesktopAdapter)

    with patch("rai.tools.desktop._ACTIVE_ADAPTER", None):
        with patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "ubuntu:GNOME"}):
            adapter = get_desktop_adapter()
            assert isinstance(adapter, GnomeDesktopAdapter)

    with patch("rai.tools.desktop._ACTIVE_ADAPTER", None):
        with patch.dict(os.environ, {"XDG_CURRENT_DESKTOP": "unknown"}):
            adapter = get_desktop_adapter()
            assert isinstance(adapter, GnomeDesktopAdapter)  # standard fallback


class TestGnomeDesktopAdapter:
    """Tests for GnomeDesktopAdapter."""

    @patch("rai.tools.desktop.gnome._HAS_DBUS", True)
    @patch("rai.tools.desktop.gnome.pydbus.SessionBus")
    def test_send_notification_dbus_success(self, mock_session_bus: MagicMock) -> None:
        """Test sending a notification using DBus."""
        mock_bus = MagicMock()
        mock_session_bus.return_value = mock_bus
        mock_notifications = MagicMock()
        mock_bus.get.return_value = mock_notifications

        adapter = GnomeDesktopAdapter()
        summary = "Test Summary"
        body = "Test Body"
        result = adapter.send_notification(summary=summary, body=body)

        mock_bus.get.assert_called_once_with(
            "org.freedesktop.Notifications", "/org/freedesktop/Notifications"
        )
        mock_notifications.Notify.assert_called_once()
        assert f"Notification sent: Summary='{summary}', Body='{body}'" in result

    @patch("rai.tools.desktop.gnome._HAS_DBUS", False)
    @patch("rai.tools.desktop.gnome.subprocess.run")
    def test_send_notification_cli_fallback(self, mock_run: MagicMock) -> None:
        """Test notify-send CLI fallback when DBus is not available."""
        adapter = GnomeDesktopAdapter()
        summary = "Test Summary"
        body = "Test Body"
        result = adapter.send_notification(summary=summary, body=body)

        mock_run.assert_called_once_with(
            ["notify-send", "-a", "AI Assistant", summary, body], check=True
        )
        assert "Notification sent via notify-send" in result

    @patch("rai.tools.desktop.gnome.os.remove")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("rai.tools.desktop.gnome.base64.b64encode")
    @patch("rai.tools.desktop.gnome.os.path.exists", return_value=True)
    @patch("rai.tools.desktop.gnome.subprocess.run")
    @patch("rai.tools.desktop.gnome.datetime")
    def test_take_screenshot_success(
        self, mock_dt: MagicMock, mock_run: MagicMock, _mock_exists: MagicMock,
        mock_b64: MagicMock, mock_open: MagicMock, mock_remove: MagicMock
    ) -> None:
        """Test taking a screenshot on GNOME."""
        mock_dt.datetime.now.strftime.return_value = "20230101_120000"
        mock_b64.return_value.decode.return_value = "fake_base64_string"

        adapter = GnomeDesktopAdapter()
        with patch("rai.tools.desktop.gnome.os.path.join", return_value="/tmp/s.png"):
            result = adapter.take_screenshot(delay=0)
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

    @patch("rai.tools.desktop.gnome.subprocess.run")
    def test_take_screenshot_fails(self, mock_run: MagicMock) -> None:
        """Test failure during gnome-screenshot capture."""
        mock_run.side_effect = CalledProcessError(1, "cmd", stderr="Error")
        adapter = GnomeDesktopAdapter()
        result = adapter.take_screenshot(delay=0)
        result_json = json.loads(result)
        assert result_json["status"] == "error"
        assert "Failed to take screenshot" in result_json["message"]


class TestCosmicDesktopAdapter:
    """Tests for CosmicDesktopAdapter."""

    @patch("rai.tools.desktop.cosmic.subprocess.run")
    def test_take_screenshot_cosmic_success(self, mock_run: MagicMock) -> None:
        """Test taking a screenshot using cosmic-screenshot on COSMIC."""
        adapter = CosmicDesktopAdapter()
        
        # Mock file reading and encoding
        with patch("rai.tools.desktop.cosmic.os.path.exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("rai.tools.desktop.cosmic.base64.b64encode") as mock_b64, \
             patch("rai.tools.desktop.cosmic.os.remove") as mock_remove:
            
            mock_b64.return_value.decode.return_value = "cosmic_b64"
            result = adapter.take_screenshot(delay=0)
            result_json = json.loads(result)

            assert mock_run.call_args[0][0][0] == "cosmic-screenshot"
            assert result_json["type"] == "image_data"
            assert result_json["base64"] == "cosmic_b64"

    @patch("rai.tools.desktop.cosmic.urllib.request.urlopen")
    def test_weather_wttr_fallback(self, mock_urlopen: MagicMock) -> None:
        """Test wttr.in weather fallback retrieval on COSMIC."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "current_condition": [{
                "temp_C": "22",
                "FeelsLikeC": "21",
                "weatherDesc": [{"value": "Sunny"}],
                "windspeedKmph": "10",
                "pressure": "1015",
                "humidity": "50"
            }],
            "nearest_area": [{
                "areaName": [{"value": "Warsaw"}],
                "country": [{"value": "Poland"}]
            }]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        adapter = CosmicDesktopAdapter()
        result = adapter.weather(location="Warsaw")
        result_json = json.loads(result)

        assert "Warsaw" in result_json["location"]
        assert result_json["temperature_c"] == 22.0
        assert result_json["conditions"] == "Sunny"


def test_desktop_adapter_deepcopy() -> None:
    """Test that DesktopAdapter instances return themselves when deepcopied."""
    import copy
    from rai.tools.desktop.gnome import GnomeDesktopAdapter
    from rai.tools.desktop.cosmic import CosmicDesktopAdapter

    gnome_adapter = GnomeDesktopAdapter()
    assert copy.deepcopy(gnome_adapter) is gnome_adapter

    cosmic_adapter = CosmicDesktopAdapter()
    assert copy.deepcopy(cosmic_adapter) is cosmic_adapter


def test_gitlab_tools_deepcopy() -> None:
    """Test that GitlabTools instances return themselves when deepcopied."""
    import copy
    from rai.tools.gitlab import GitlabTools

    # Mock environment variable for GitlabTools initialization to prevent validation error
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"GITLAB_ACCESS_TOKEN": "mock-token"}):
        gitlab_tools = GitlabTools()
        assert copy.deepcopy(gitlab_tools) is gitlab_tools