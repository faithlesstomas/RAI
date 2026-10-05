"""Document permissions survive the runtime configuration load/save boundary."""
import json

from rai import config_manager


def test_actions_config_survives_load_and_save(tmp_path, monkeypatch):
    monkeypatch.setattr(config_manager, 'load_agents', lambda: {})
    path = tmp_path / 'config.json'
    actions = {'allowed_file_roots': ['/tmp/rai-documents'], 'browser_endpoint': 'http://127.0.0.1:9222'}
    path.write_text(json.dumps({'actions': actions}))
    loaded = config_manager.load_config(str(path))
    assert loaded['actions'] == actions
    config_manager.save_config(loaded, str(path))
    assert json.loads(path.read_text())['actions'] == actions
    assert config_manager.load_config(str(path))['actions'] == actions
