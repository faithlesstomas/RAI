"""Document activation reports only independently observed open files."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

import pytest
from returns.result import Failure, Success

import rai.actions.documents as documents
from rai.actions.documents import LinuxDocumentBackend, document_reader
from rai.actions.files import Document
from rai.kernel.ports import CancellationToken

EVIDENCE = {"pid": 1, "start_time": "2", "document_fingerprint": "f", "open_file_descriptor": True}


def document(tmp_path: Path) -> Document:
    path = tmp_path / "note.txt"
    path.write_text("content")
    return Document(path, "fingerprint")


def test_reader_finds_own_open_file_descriptor(tmp_path: Path) -> None:
    target = document(tmp_path)
    with target.path.open("rb"):
        evidence = document_reader(target)
    assert evidence is not None
    assert evidence["pid"] == os.getpid() and evidence["open_file_descriptor"] is True
    assert evidence["document_fingerprint"] == "fingerprint"


def test_reader_reports_nothing_for_unopened_missing_or_unbounded_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = document(tmp_path)
    assert document_reader(target) is None
    assert document_reader(Document(tmp_path / "missing.txt", "x")) is None
    with target.path.open("rb"):
        monkeypatch.setattr(documents, "MAX_PROCESSES", 0)
        assert document_reader(target) is None
        monkeypatch.setattr(documents, "MAX_PROCESSES", 4096)
        monkeypatch.setattr(documents, "MAX_FDS", 0)
        assert document_reader(target) is None


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("RAI_SECRET_TOKEN", "must-not-leak")
    monkeypatch.setattr(documents, "POLL_INTERVAL", 0.01)
    monkeypatch.setattr(documents, "OPEN_TIMEOUT", 0.3)


def fake_gio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> Path:
    gio = tmp_path / "gio"
    gio.write_text(f"#!/bin/sh\n{script}\n")
    gio.chmod(gio.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(documents.shutil, "which", lambda _name: str(gio))
    return gio


async def test_open_succeeds_when_reader_observes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    target = document(tmp_path)
    marker = tmp_path / "environment"
    fake_gio(tmp_path, monkeypatch, f'env > "{marker}"')
    monkeypatch.setattr(documents, "document_reader",
                        lambda _document: EVIDENCE if marker.exists() and marker.stat().st_size else None)
    assert await LinuxDocumentBackend().open(target, CancellationToken()) == Success(EVIDENCE)
    assert "RAI_SECRET_TOKEN" not in marker.read_text()



async def test_open_without_gio_or_session_is_desktop_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = document(tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(documents.shutil, "which", lambda _name: "/usr/bin/gio")
    assert await LinuxDocumentBackend().open(target, CancellationToken()) == Failure("DESKTOP_UNAVAILABLE")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(documents.shutil, "which", lambda _name: None)
    assert await LinuxDocumentBackend().open(target, CancellationToken()) == Failure("DESKTOP_UNAVAILABLE")


async def test_cancelled_before_spawn_never_starts_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    target = document(tmp_path)
    marker = tmp_path / "started"
    fake_gio(tmp_path, monkeypatch, f'touch "{marker}"')
    token = CancellationToken()
    token.cancel()
    assert await LinuxDocumentBackend().open(target, token) == Failure("CANCELLED")
    assert not marker.exists()


async def test_cancellation_after_spawn_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    target = document(tmp_path)
    fake_gio(tmp_path, monkeypatch, "exit 0")
    token = CancellationToken()

    def reader(_document: Document) -> Any:
        token.cancel()
        return None

    monkeypatch.setattr(documents, "document_reader", reader)
    assert await LinuxDocumentBackend().open(target, token) == Failure("UNKNOWN")


async def test_viewer_failure_without_evidence_is_open_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    target = document(tmp_path)
    fake_gio(tmp_path, monkeypatch, "exit 3")
    monkeypatch.setattr(documents, "document_reader", lambda _document: None)
    assert await LinuxDocumentBackend().open(target, CancellationToken()) == Failure("DOCUMENT_OPEN_FAILED")


async def test_timeout_without_evidence_is_unknown_and_reaps_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    target = document(tmp_path)
    monkeypatch.setattr(documents, "OPEN_TIMEOUT", 0.1)
    fake_gio(tmp_path, monkeypatch, "sleep 30")
    monkeypatch.setattr(documents, "document_reader", lambda _document: None)
    assert await LinuxDocumentBackend().open(target, CancellationToken()) == Failure("UNKNOWN")


async def test_directory_verification_requires_exact_application(tmp_path, monkeypatch):
    import json
    import rai.actions.commands as commands
    from rai.actions.applications import Application
    from rai.actions.documents import directory_reader
    target = Document(tmp_path, "directory-fingerprint")
    async def receipt(*args):
        return Success(json.dumps({"pid": os.getpid()}))
    monkeypatch.setattr(commands, "run_command", receipt)
    evidence = await directory_reader(target, None, CancellationToken())
    assert evidence["directory_location_observed"]
    wrong = Application("wrong.desktop", "Wrong", tmp_path / "wrong.desktop", "fp", "/not-this-process")
    assert await directory_reader(target, wrong, CancellationToken()) is None
    async def unavailable(*args):
        return Failure("COMMAND_FAILED")
    monkeypatch.setattr(commands, "run_command", unavailable)
    assert await directory_reader(target, None, CancellationToken()) is None
