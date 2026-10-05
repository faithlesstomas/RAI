"""Reading a profile must not change models or rewrite user files."""
import yaml
from rai.config_manager import load_agents


def test_load_agents_preserves_model_and_source(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump({"work": {"model": "gemini-1.5-flash"}}))
    before = path.read_bytes()
    assert load_agents(str(path))["work"]["model"] == "gemini-1.5-flash"
    assert path.read_bytes() == before
