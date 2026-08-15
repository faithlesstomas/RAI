"""Regression tests for package import invariants."""

import importlib
import os
import subprocess

import pytest
import rai


def test_import_does_not_patch_process_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_popen = subprocess.Popen
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    importlib.reload(rai)

    assert subprocess.Popen is original_popen
    assert "GEMINI_API_KEY" not in os.environ
