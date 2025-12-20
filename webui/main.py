"""
Main entry point for RAI WebUI.
"""
import logging
from nicegui import ui
from pages import chat, agents, chains, settings, dashboard # pylint: disable=unused-import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("webui.log")
    ]
)
# Suppress noisy watchfiles logs
logging.getLogger("watchfiles").setLevel(logging.WARNING)

logger = logging.getLogger("webui")
logger.info("Starting RAI Web UI...")

# Redirect root to dashboard
@ui.page('/')
def index() -> None:
    """Redirect root to dashboard."""
    ui.navigate.to('/dashboard')

# Initialize pages (they register themselves via decorators)
# This import is enough because the modules execute their @ui.page calls

def main() -> None:
    """Run the NiceGUI application."""
    ui.run(
        title='RAI Assistant',
        port=5001,
        dark=True,
        show=False,
        reload=True,
        storage_secret='rai-secret-key'
    )

if __name__ in {"__main__", "__mp_main__"}:
    main()
