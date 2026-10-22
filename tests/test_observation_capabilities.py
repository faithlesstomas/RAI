"""Unit and integration tests for observation processing capabilities."""
# ruff: noqa: PLR2004

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
from returns.result import Success

from rai.actions.observations import (
    register_observation_capabilities,
)
from rai.assistant.records import ObservationArtifact, summarize_observation_schema
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.records import CapabilityRequest, DataClass, ProducerIdentity
from rai.kernel.transport import normalize_request

ACTOR = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    return SQLiteMemoryGraphStore(tmp_path / "memory.sqlite3")


@pytest.fixture
def registry(store: SQLiteMemoryGraphStore) -> CapabilityRegistry:
    reg = CapabilityRegistry()
    register_observation_capabilities(reg, store)
    return reg


async def _invoke(registry: CapabilityRegistry, req: CapabilityRequest) -> Any:
    cap_res = registry.resolve(req)
    assert isinstance(cap_res, Success)
    res = await cap_res.unwrap().invoke(req, CancellationToken())
    assert isinstance(res, Success)
    return res.unwrap()


@pytest.mark.asyncio
async def test_observation_search_records(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "applications": [
            {"name": "VLC Media Player", "categories": ["AudioVideo", "Player"], "comment": "Play movies"},
            {"name": "Firefox", "categories": ["Network", "WebBrowser"], "comment": "Browse web"},
            {"name": "MPV", "categories": ["AudioVideo", "Player"], "comment": "Lightweight video player"},
            {"name": "LibreOffice Writer", "categories": ["Office", "WordProcessor"], "comment": "Edit documents"},
        ]
    }
    art = ObservationArtifact(
        artifact_id="obs_apps",
        session_id="session-1",
        turn_id="turn-1",
        capability="application.list",
        request_id="req-1",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=200,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.search")
    assert desc is not None
    req = normalize_request(
        desc,
        {"ref": "obs_apps", "pattern": "player"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )

    res = await _invoke(registry, req)
    output = res.output
    assert output["total_matches"] == 2
    matched_names = {m["name"] for m in output["matches"]}
    assert matched_names == {"VLC Media Player", "MPV"}
    assert "artifact_id" in output


@pytest.mark.asyncio
async def test_observation_search_text_lines(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    text = (
        "Heading: Introduction\n"
        "This is line 1 with general info.\n"
        "Secret fact: the treasure is under the oak tree.\n"
        "Concluding remarks.\n"
    )
    content = {"title": "Page", "text": text}
    art = ObservationArtifact(
        artifact_id="obs_page",
        session_id="session-1",
        turn_id="turn-1",
        capability="browser.read_page",
        request_id="req-page",
        data_class=DataClass.PUBLIC,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=len(text),
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.search")
    assert desc is not None
    req = normalize_request(
        desc,
        {"ref": "obs_page", "pattern": "treasure"},
        actor=ACTOR,
        data_class=DataClass.PUBLIC,
    )

    res = await _invoke(registry, req)
    output = res.output
    assert output["total_matches"] == 1
    match = output["matches"][0]
    assert match["line_number"] == 3
    assert "treasure is under the oak tree" in match["line"]
    assert "Introduction" in match["context"] or "line 1" in match["context"]


@pytest.mark.asyncio
async def test_observation_filter(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "entries": [
            {"name": "report.pdf", "size": 1500, "kind": "file"},
            {"name": "notes.txt", "size": 300, "kind": "file"},
            {"name": "images", "size": 0, "kind": "directory"},
            {"name": "summary.odt", "size": 4200, "kind": "file"},
        ]
    }
    art = ObservationArtifact(
        artifact_id="obs_files",
        session_id="session-1",
        turn_id="turn-1",
        capability="file.list",
        request_id="req-files",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=200,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.filter")
    assert desc is not None

    # 1. contains
    req1 = normalize_request(
        desc,
        {"ref": "obs_files", "field": "name", "op": "contains", "value": "re"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out1 = (await _invoke(registry, req1)).output
    assert out1["total_matches"] == 1  # report.pdf (images has im, not re)

    # 2. numeric gt
    req2 = normalize_request(
        desc,
        {"ref": "obs_files", "field": "size", "op": "gt", "value": 1000},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out2 = (await _invoke(registry, req2)).output
    assert out2["total_matches"] == 2
    assert {e["name"] for e in out2["results"]} == {"report.pdf", "summary.odt"}


@pytest.mark.asyncio
async def test_observation_slice(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {"items": [f"item_{i}" for i in range(100)]}
    art = ObservationArtifact(
        artifact_id="obs_seq",
        session_id="session-1",
        turn_id="turn-1",
        capability="test",
        request_id="req-seq",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=500,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.slice")
    assert desc is not None
    req = normalize_request(
        desc,
        {"ref": "obs_seq", "offset": 10, "limit": 5},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res = (await _invoke(registry, req)).output
    assert res["total_count"] == 100
    assert list(res["items"]) == ["item_10", "item_11", "item_12", "item_13", "item_14"]


@pytest.mark.asyncio
async def test_observation_count(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "processes": [
            {"pid": 1, "state": "S"},
            {"pid": 2, "state": "R"},
            {"pid": 3, "state": "S"},
            {"pid": 4, "state": "Z"},
            {"pid": 5, "state": "S"},
        ]
    }
    art = ObservationArtifact(
        artifact_id="obs_procs",
        session_id="session-1",
        turn_id="turn-1",
        capability="process.inspect",
        request_id="req-procs",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=150,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.count")
    assert desc is not None

    # Total count
    req1 = normalize_request(
        desc,
        {"ref": "obs_procs"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out1 = (await _invoke(registry, req1)).output
    assert out1["total_count"] == 5

    # Group by field
    req2 = normalize_request(
        desc,
        {"ref": "obs_procs", "field": "state"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out2 = (await _invoke(registry, req2)).output
    assert out2["counts"] == {"S": 3, "R": 1, "Z": 1}


@pytest.mark.asyncio
async def test_observation_get_json_pointer(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "applications": [
            {"name": "App0", "desktop_id": "app0.desktop"},
            {"name": "App1", "desktop_id": "app1.desktop"},
        ],
        "metadata": {"version": "2.0"},
    }
    art = ObservationArtifact(
        artifact_id="obs_ptr",
        session_id="session-1",
        turn_id="turn-1",
        capability="test",
        request_id="req-ptr",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=200,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.get")
    assert desc is not None
    req = normalize_request(
        desc,
        {"ref": "obs_ptr", "path": "/applications/1/name"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out = (await _invoke(registry, req)).output
    assert out["value"] == "App1"


def test_register_observation_capabilities(store: SQLiteMemoryGraphStore) -> None:
    registry = CapabilityRegistry()
    register_observation_capabilities(registry, store)
    assert registry.descriptor("observation.search") is not None
    assert registry.descriptor("observation.filter") is not None
    assert registry.descriptor("observation.slice") is not None
    assert registry.descriptor("observation.count") is not None
    assert registry.descriptor("observation.get") is not None
