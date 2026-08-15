"""Tests for XDG-compliant runtime paths."""

from pathlib import Path

import pytest

from rai import paths


def test_explicit_rai_path_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("RAI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAI_RUNTIME_DIR", str(tmp_path / "run"))

    assert paths.config_dir() == tmp_path / "config"
    assert paths.cache_dir() == tmp_path / "cache"
    assert paths.data_dir() == tmp_path / "data"
    assert paths.runtime_dir() == tmp_path / "run"


def test_xdg_runtime_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAI_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert paths.runtime_dir() == tmp_path / "rai"
