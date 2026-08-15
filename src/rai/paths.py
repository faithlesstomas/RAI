"""XDG-compliant filesystem locations used by the RAI runtime."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_dir(env_name: str, fallback: str) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser()
    return Path(fallback).expanduser()


def config_dir() -> Path:
    """Return RAI's configuration directory."""
    override = os.environ.get("RAI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg_dir("XDG_CONFIG_HOME", "~/.config") / "rai"


def cache_dir() -> Path:
    """Return RAI's disposable cache directory."""
    override = os.environ.get("RAI_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg_dir("XDG_CACHE_HOME", "~/.cache") / "rai"


def data_dir() -> Path:
    """Return RAI's persistent data directory."""
    override = os.environ.get("RAI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg_dir("XDG_DATA_HOME", "~/.local/share") / "rai"


def runtime_dir() -> Path:
    """Return the directory for sockets and ephemeral credentials."""
    override = os.environ.get("RAI_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return Path(xdg_runtime) / "rai"
    return cache_dir() / "run"

