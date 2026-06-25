"""Desktop adapter package exposing the dynamic adapter factory."""
import os
import logging
from rai.tools.desktop.base import DesktopAdapter
from rai.tools.desktop.gnome import GnomeDesktopAdapter
from rai.tools.desktop.cosmic import CosmicDesktopAdapter

logger = logging.getLogger(__name__)

# Cache active adapter instance
_ACTIVE_ADAPTER = None

def get_desktop_adapter() -> DesktopAdapter:
    """
    Detects the current desktop environment and returns the appropriate adapter.
    Caches the adapter instance for performance.
    """
    global _ACTIVE_ADAPTER
    if _ACTIVE_ADAPTER is not None:
        return _ACTIVE_ADAPTER

    desktop = os.getenv("XDG_CURRENT_DESKTOP", "").upper()
    if "COSMIC" in desktop:
        logger.debug("COSMIC desktop environment detected. Initializing CosmicDesktopAdapter.")
        _ACTIVE_ADAPTER = CosmicDesktopAdapter()
    elif "GNOME" in desktop or "UBUNTU" in desktop:
        logger.debug("GNOME desktop environment detected. Initializing GnomeDesktopAdapter.")
        _ACTIVE_ADAPTER = GnomeDesktopAdapter()
    else:
        logger.debug(
            f"No specific desktop environment matched '{desktop}'. "
            "Falling back to standard GnomeDesktopAdapter for Freedesktop D-Bus compatibility."
        )
        _ACTIVE_ADAPTER = GnomeDesktopAdapter()

    return _ACTIVE_ADAPTER
