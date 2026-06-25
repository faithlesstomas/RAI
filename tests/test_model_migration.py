import os
import tempfile
import yaml
from unittest.mock import patch
from rai.config_manager import load_agents

def test_load_agents_migrates_model() -> None:
    """Test that load_agents automatically migrates gemini-1.5-flash to gemini-2.5-flash."""
    # Create a temporary yaml agents file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        initial_agents = {
            "legacy-agent": {
                "name": "legacy-agent",
                "model": "gemini-1.5-flash",
                "system": "legacy system instructions",
                "tools": ["CalculatorTools"]
            },
            "modern-agent": {
                "name": "modern-agent",
                "model": "gemini-2.5-flash",
                "system": "modern system instructions",
                "tools": ["CalculatorTools"]
            }
        }

        with open(tmp_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(initial_agents, f)

        # Load agents using the temporary file path
        loaded = load_agents(path=tmp_name)

        # Assertions: Legacy agent should be migrated
        assert loaded["legacy-agent"]["model"] == "gemini-2.5-flash"
        # Modern agent should remain unchanged
        assert loaded["modern-agent"]["model"] == "gemini-2.5-flash"

        # Verify that changes were saved back to the file
        with open(tmp_name, "r", encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        assert saved["legacy-agent"]["model"] == "gemini-2.5-flash"

    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
