"""Unit and integration tests for observation processing capabilities."""
# ruff: noqa: PLR2004

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
from returns.result import Failure, Success

from conftest import ASGITestClient
from rai.actions.observations import (
    register_observation_capabilities,
)
from rai.assistant.records import ObservationArtifact, summarize_observation_schema
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionResult, CapabilityRequest, DataClass, ProducerIdentity
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.routers.mcp import call_tool
from rai.server import create_app
from rai.tools.security.auth import authorization_headers, create_session_token

ACTOR = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    return SQLiteMemoryGraphStore(tmp_path / "memory.sqlite3")


@pytest.fixture
def registry(store: SQLiteMemoryGraphStore) -> CapabilityRegistry:
    reg = CapabilityRegistry()
    register_observation_capabilities(reg, store)
    return reg


async def _invoke(registry: CapabilityRegistry, req: CapabilityRequest) -> ActionResult:
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
        {"session_id": "session-1", "ref": "obs_apps", "pattern": "player"},
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
        {"session_id": "session-1", "ref": "obs_page", "pattern": "treasure"},
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
        {"session_id": "session-1", "ref": "obs_files", "field": "name", "op": "contains", "value": "re"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out1 = (await _invoke(registry, req1)).output
    assert out1["total_matches"] == 1  # report.pdf (images has im, not re)

    # 2. numeric gt
    req2 = normalize_request(
        desc,
        {"session_id": "session-1", "ref": "obs_files", "field": "size", "op": "gt", "value": 1000},
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
        {"session_id": "session-1", "ref": "obs_seq", "offset": 10, "limit": 5},
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
        {"session_id": "session-1", "ref": "obs_procs"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out1 = (await _invoke(registry, req1)).output
    assert out1["total_count"] == 5
    assert "artifact_id" in out1

    # Group by field
    req2 = normalize_request(
        desc,
        {"session_id": "session-1", "ref": "obs_procs", "field": "state"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out2 = (await _invoke(registry, req2)).output
    assert out2["counts"] == {"S": 3, "R": 1, "Z": 1}
    assert "artifact_id" in out2


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
        {"session_id": "session-1", "ref": "obs_ptr", "path": "/applications/1/name"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    out = (await _invoke(registry, req)).output
    assert out["value"] == "App1"


@pytest.mark.asyncio
async def test_observation_session_isolation_refuses_cross_session_access(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {"secret_notes": ["Private user data in session A"]}
    art = ObservationArtifact(
        artifact_id="obs_private",
        session_id="session-A",
        turn_id="turn-A",
        capability="test",
        request_id="req-A",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=100,
        producer=ACTOR,
    )
    await store.put(art)

    # Attempt to read session-A artifact from session-B
    desc = registry.descriptor("observation.search")
    assert desc is not None
    req = normalize_request(
        desc,
        {"session_id": "session-B", "ref": "obs_private", "pattern": "Private"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap_res = registry.resolve(req)
    assert isinstance(cap_res, Success)
    res = await cap_res.unwrap().invoke(req, CancellationToken())
    assert isinstance(res, Failure)
    assert res.failure().code == "ARTIFACT_NOT_FOUND"


@pytest.mark.asyncio
async def test_observation_requires_session_id(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    desc = registry.descriptor("observation.search")
    assert desc is not None

    # Missing session_id -> schema rejection
    req1 = normalize_request(
        desc,
        {"ref": "obs_any", "pattern": "test"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap_res1 = registry.resolve(req1)
    assert isinstance(cap_res1, Success)
    res1 = await cap_res1.unwrap().invoke(req1, CancellationToken())
    assert isinstance(res1, Failure)
    assert res1.failure().code == "UNAUTHORIZED"
    assert "session_id" in res1.failure().message

    # Empty session_id -> capability rejection
    req2 = normalize_request(
        desc,
        {"session_id": "   ", "ref": "obs_any", "pattern": "test"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap_res2 = registry.resolve(req2)
    assert isinstance(cap_res2, Success)
    res2 = await cap_res2.unwrap().invoke(req2, CancellationToken())
    assert isinstance(res2, Failure)
    assert res2.failure().code == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_observation_search_redos_safe(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    # Evil pattern is escaped and treated as safe literal substring, finishing instantly
    evil_text = "a" * 35 + "!"
    art = ObservationArtifact(
        artifact_id="obs_evil",
        session_id="session-timeout",
        turn_id="turn-timeout",
        capability="test",
        request_id="req-evil",
        data_class=DataClass.LOCAL,
        content={"text": evil_text},
        schema_summary=summarize_observation_schema({"text": evil_text}),
        size_chars=len(evil_text),
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.search")
    assert desc is not None
    req = normalize_request(
        desc,
        {
            "session_id": "session-timeout",
            "ref": "obs_evil",
            "pattern": "^(a+)+$",
        },
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res = await _invoke(registry, req)
    assert res.output["total_matches"] == 0


@pytest.mark.asyncio
async def test_observation_search_cancellation(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    lines = "\n".join(f"Line {i}: test data content" for i in range(500))
    art = ObservationArtifact(
        artifact_id="obs_cancel",
        session_id="session-cancel",
        turn_id="turn-cancel",
        capability="test",
        request_id="req-cancel",
        data_class=DataClass.LOCAL,
        content={"text": lines},
        schema_summary=summarize_observation_schema({"text": lines}),
        size_chars=len(lines),
        producer=ACTOR,
    )
    await store.put(art)

    token = CancellationToken()
    token.cancel()

    desc = registry.descriptor("observation.search")
    assert desc is not None
    req = normalize_request(
        desc,
        {"session_id": "session-cancel", "ref": "obs_cancel", "pattern": "Line"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap_res = registry.resolve(req)
    assert isinstance(cap_res, Success)
    res = await cap_res.unwrap().invoke(req, token)
    assert isinstance(res, Failure)
    assert res.failure().code == "CANCELLED"


@pytest.mark.asyncio
async def test_observation_get_rfc6901_unescaping(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "special/key": "slash_value",
        "tilde~key": "tilde_value",
    }
    art = ObservationArtifact(
        artifact_id="obs_rfc",
        session_id="session-rfc",
        turn_id="turn-rfc",
        capability="test",
        request_id="req-rfc",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=100,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.get")
    assert desc is not None

    # Test ~1 decoding to /
    req1 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc", "path": "/special~1key"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res1 = (await _invoke(registry, req1)).output
    assert res1["value"] == "slash_value"

    # Test ~0 decoding to ~
    req2 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc", "path": "/tilde~0key"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res2 = (await _invoke(registry, req2)).output
    assert res2["value"] == "tilde_value"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_observation_get_rfc6901_empty_tokens_and_invalid_pointers(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    content = {
        "": "root_empty_value",
        "nested": {"": "nested_empty_value", "key": "val"},
        "items": ["first", "second"],
    }
    art = ObservationArtifact(
        artifact_id="obs_rfc_spec",
        session_id="session-rfc",
        turn_id="turn-rfc",
        capability="test",
        request_id="req-rfc",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=100,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.get")
    assert desc is not None

    # 1. "/" accesses empty key "" in root object
    req1 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res1 = (await _invoke(registry, req1)).output
    assert res1["value"] == "root_empty_value"

    # 2. "/nested/" accesses empty key "" in object "nested"
    req2 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/nested/"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res2 = (await _invoke(registry, req2)).output
    assert res2["value"] == "nested_empty_value"

    # 3. "/items/0" accesses first array element
    req3 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/items/0"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res3 = (await _invoke(registry, req3)).output
    assert res3["value"] == "first"

    # 4. Missing empty key under nested
    req4 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/items/"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap4 = registry.resolve(req4).unwrap()
    res4 = await cap4.invoke(req4, CancellationToken())
    assert isinstance(res4, Failure)
    assert res4.failure().code == "INVALID_POINTER"

    # 5. Missing key
    req5 = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/missing"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    cap5 = registry.resolve(req5).unwrap()
    res5 = await cap5.invoke(req5, CancellationToken())
    assert isinstance(res5, Failure)
    assert res5.failure().code == "PATH_NOT_FOUND"

    # 6. Invalid escapes (~2, trailing ~)
    for bad_path in ("/bad/~2", "/bad~"):
        req_bad = normalize_request(
            desc,
            {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": bad_path},
            actor=ACTOR,
            data_class=DataClass.LOCAL,
        )
        res_bad = await registry.resolve(req_bad).unwrap().invoke(req_bad, CancellationToken())
        assert isinstance(res_bad, Failure)
        assert res_bad.failure().code == "INVALID_POINTER"

    # 7. Array invalid index (leading zero, negative, out of bounds)
    for bad_idx in ("/items/01", "/items/-1"):
        req_idx = normalize_request(
            desc,
            {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": bad_idx},
            actor=ACTOR,
            data_class=DataClass.LOCAL,
        )
        res_idx = await registry.resolve(req_idx).unwrap().invoke(req_idx, CancellationToken())
        assert isinstance(res_idx, Failure)
        assert res_idx.failure().code == "INVALID_POINTER"

    req_oob = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": "/items/5"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res_oob = await registry.resolve(req_oob).unwrap().invoke(req_oob, CancellationToken())
    assert isinstance(res_oob, Failure)
    assert res_oob.failure().code == "PATH_NOT_FOUND"

    # 8. Empty pointer "" returns the root document
    req_root = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec", "path": ""},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res_root = (await _invoke(registry, req_root)).output
    assert res_root["value"] == (await store.get("session-rfc", "obs_rfc_spec")).unwrap().content
    assert res_root["path"] == ""

    # 9. Missing path argument fails with INVALID_ARGUMENT
    req_no_path = normalize_request(
        desc,
        {"session_id": "session-rfc", "ref": "obs_rfc_spec"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res_no_path = await registry.resolve(req_no_path).unwrap().invoke(req_no_path, CancellationToken())
    assert isinstance(res_no_path, Failure)
    assert res_no_path.failure().code == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_observation_search_pagination_and_bounds(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    items = [{"id": i, "name": f"item_{i}"} for i in range(10)]
    content = {"items": items}
    art = ObservationArtifact(
        artifact_id="obs_search_page",
        session_id="session-page",
        turn_id="turn-page",
        capability="test",
        request_id="req-page",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=200,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.search")
    assert desc is not None

    # Page 1: limit 3, offset 0
    req1 = normalize_request(
        desc,
        {"session_id": "session-page", "ref": "obs_search_page", "pattern": "item_", "limit": 3, "offset": 0},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res1 = (await _invoke(registry, req1)).output
    assert res1["total_matches"] == 10
    assert len(res1["matches"]) == 3
    assert res1["offset"] == 0
    assert res1["limit"] == 3
    assert res1["complete"] is False
    assert "paginated" in (res1["reason"] or "")

    # Page 2: limit 3, offset 3
    req2 = normalize_request(
        desc,
        {"session_id": "session-page", "ref": "obs_search_page", "pattern": "item_", "limit": 3, "offset": 3},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res2 = (await _invoke(registry, req2)).output
    assert len(res2["matches"]) == 3
    assert res2["matches"][0]["id"] == 3
    assert res2["complete"] is False

    # Full page: limit 10, offset 0
    req3 = normalize_request(
        desc,
        {"session_id": "session-page", "ref": "obs_search_page", "pattern": "item_", "limit": 10, "offset": 0},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res3 = (await _invoke(registry, req3)).output
    assert len(res3["matches"]) == 10
    assert res3["complete"] is True
    assert res3["reason"] is None


@pytest.mark.asyncio
async def test_observation_filter_pagination_and_cancellation(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    items = [{"id": i, "score": i * 10} for i in range(10)]
    content = {"items": items}
    art = ObservationArtifact(
        artifact_id="obs_filter_page",
        session_id="session-filter",
        turn_id="turn-filter",
        capability="test",
        request_id="req-filter",
        data_class=DataClass.LOCAL,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=200,
        producer=ACTOR,
    )
    await store.put(art)

    desc = registry.descriptor("observation.filter")
    assert desc is not None

    # Paginated filter
    req1 = normalize_request(
        desc,
        {"session_id": "session-filter", "ref": "obs_filter_page", "field": "score", "op": "gte", "value": 0, "limit": 4, "offset": 0},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res1 = (await _invoke(registry, req1)).output
    assert res1["total_matches"] == 10
    assert len(res1["results"]) == 4
    assert res1["complete"] is False
    assert "paginated" in (res1["reason"] or "")

    # Cooperative cancellation
    token = CancellationToken()
    token.cancel()
    res_cancel = await registry.resolve(req1).unwrap().invoke(req1, token)
    assert isinstance(res_cancel, Failure)
    assert res_cancel.failure().code == "CANCELLED"


@pytest.mark.asyncio
async def test_observation_transport_session_isolation_mcp(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    art = ObservationArtifact(
        artifact_id="obs_victim",
        session_id="session-victim",
        turn_id="turn-victim",
        capability="test",
        request_id="req-victim",
        data_class=DataClass.LOCAL,
        content={"secret": "victim-private-data"},
        schema_summary=summarize_observation_schema({"secret": "victim-private-data"}),
        size_chars=50,
        producer=ACTOR,
    )
    await store.put(art)

    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger())

    # 1. Attacker calls tool specifying victim's session_id in arguments, but transport session is session-attacker
    mcp_result = await call_tool(
        service,
        "observation.get",
        {"session_id": "session-victim", "ref": "obs_victim", "path": "/secret"},
        session_id="session-attacker",
    )
    assert mcp_result.isError is True
    assert "ARTIFACT_NOT_FOUND" in mcp_result.content[0].text

    # 2. Client calls without transport session_id
    mcp_unauth = await call_tool(
        service,
        "observation.get",
        {"session_id": "session-victim", "ref": "obs_victim", "path": "/secret"},
        session_id=None,
    )
    assert mcp_unauth.isError is True
    assert "UNAUTHORIZED" in mcp_unauth.content[0].text


def test_observation_transport_session_isolation_rest(tmp_path: Path) -> None:
    container = ApplicationContainer(
        config={"db_path": str(tmp_path / "memory.sqlite3")},
        audit_ledger=InMemoryAuditLedger(),
        testing=True,
    )

    art = ObservationArtifact(
        artifact_id="obs_victim_rest",
        session_id="session-victim",
        turn_id="turn-victim",
        capability="test",
        request_id="req-victim-rest",
        data_class=DataClass.LOCAL,
        content={"secret": "victim-rest-secret"},
        schema_summary=summarize_observation_schema({"secret": "victim-rest-secret"}),
        size_chars=50,
        producer=ACTOR,
    )
    asyncio.run(container.memory_graph_store.put(art))

    client = ASGITestClient(create_app(container))
    desc = container.capability_registry.descriptor("observation.get")
    assert desc is not None
    req_body = normalize_request(
        desc,
        {"session_id": "session-victim", "ref": "obs_victim_rest", "path": "/secret"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    ).model_dump(mode="json")

    # 1. REST call with victim session in body, but NO session token
    resp_unauth = client.post("/api/v1/capabilities/invoke", json=req_body)
    assert resp_unauth.status_code == 200
    res_data = resp_unauth.json()
    assert res_data["ok"] is False
    assert res_data["result"]["code"] == "UNAUTHORIZED"

    # 2. REST call with client trying X-Session-ID header (ignored/unauthenticated)
    resp_fake_header = client.post(
        "/api/v1/capabilities/invoke",
        headers={"x-session-id": "session-victim"},
        json=req_body,
    )
    assert resp_fake_header.status_code == 200
    res_fake_data = resp_fake_header.json()
    assert res_fake_data["ok"] is False
    assert res_fake_data["result"]["code"] == "UNAUTHORIZED"

    # 3. REST call with forged X-Session-Token
    resp_forged = client.post(
        "/api/v1/capabilities/invoke",
        headers={"x-session-token": "session-victim.invalid_sig"},
        json=req_body,
    )
    assert resp_forged.status_code == 200
    res_forged_data = resp_forged.json()
    assert res_forged_data["ok"] is False
    assert res_forged_data["result"]["code"] == "UNAUTHORIZED"

    # 4. REST call with valid X-Session-Token for session-attacker (attempting to read victim's artifact)
    attacker_token = create_session_token("session-attacker")
    resp_attacker = client.post(
        "/api/v1/capabilities/invoke",
        headers={"x-session-token": attacker_token},
        json=req_body,
    )
    assert resp_attacker.status_code == 200
    res_att_data = resp_attacker.json()
    assert res_att_data["ok"] is False
    assert res_att_data["result"]["code"] == "ARTIFACT_NOT_FOUND"

    # 5. Legitimate REST call with valid X-Session-Token: session-victim
    victim_token = create_session_token("session-victim")
    resp_legit = client.post(
        "/api/v1/capabilities/invoke",
        headers={"x-session-token": victim_token},
        json=req_body,
    )
    assert resp_legit.status_code == 200
    res_legit_data = resp_legit.json()
    assert res_legit_data["ok"] is True
    assert res_legit_data["result"]["output"]["value"] == "victim-rest-secret"


@pytest.mark.asyncio
async def test_observation_visited_limits_and_cooperative_yielding(
    registry: CapabilityRegistry, store: SQLiteMemoryGraphStore
) -> None:
    large_items = [{"id": i, "val": f"val_{i}"} for i in range(12000)]
    art = ObservationArtifact(
        artifact_id="obs_large",
        session_id="session-large",
        turn_id="turn-large",
        capability="test",
        request_id="req-large",
        data_class=DataClass.LOCAL,
        content={"items": large_items},
        schema_summary=summarize_observation_schema({"items": large_items[:10]}),
        size_chars=50000,
        producer=ACTOR,
    )
    await store.put(art)

    # 1. Search over 12,000 items with 0 matches: must hit MAX_VISITED_LIMIT (10,000)
    desc_search = registry.descriptor("observation.search")
    assert desc_search is not None
    req_search = normalize_request(
        desc_search,
        {"session_id": "session-large", "ref": "obs_large", "pattern": "nonexistent_pattern"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res_search = (await _invoke(registry, req_search)).output
    assert res_search["total_matches"] == 0
    assert res_search["complete"] is False
    assert "Hit maximum search scan limit" in (res_search["reason"] or "")

    # 2. Filter over 12,000 items with 0 matches: must hit MAX_VISITED_LIMIT (10,000)
    desc_filter = registry.descriptor("observation.filter")
    assert desc_filter is not None
    req_filter = normalize_request(
        desc_filter,
        {"session_id": "session-large", "ref": "obs_large", "field": "val", "op": "equals", "value": "nonexistent"},
        actor=ACTOR,
        data_class=DataClass.LOCAL,
    )
    res_filter = (await _invoke(registry, req_filter)).output
    assert res_filter["total_matches"] == 0
    assert res_filter["complete"] is False
    assert "Hit maximum filter scan limit" in (res_filter["reason"] or "")


@pytest.mark.asyncio
async def test_observation_mcp_sse_end_to_end(tmp_path: Path) -> None:
    container = ApplicationContainer(
        config={"db_path": str(tmp_path / "memory_sse.sqlite3")},
        audit_ledger=InMemoryAuditLedger(),
        testing=True,
    )
    art = ObservationArtifact(
        artifact_id="obs_victim_sse",
        session_id="session-victim",
        turn_id="turn-victim",
        capability="test",
        request_id="req-victim-sse",
        data_class=DataClass.LOCAL,
        content={"secret": "victim-sse-secret"},
        schema_summary=summarize_observation_schema({"secret": "victim-sse-secret"}),
        size_chars=50,
        producer=ACTOR,
    )
    await container.memory_graph_store.put(art)

    app = create_app(container)
    config = uvicorn.Config(app, host="127.0.0.1", port=8991, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)

    try:
        # 1. Attacker connects via SSE without session token, attempts to access victim's artifact
        headers = dict(authorization_headers())
        async with sse_client("http://127.0.0.1:8991/api/v1/mcp/sse", headers=headers) as (r1, w1):
            async with ClientSession(r1, w1) as session1:
                await session1.initialize()
                # Attacker passes session_id="session-victim" in arguments
                res_att = await session1.call_tool(
                    "observation.get",
                    {"ref": "obs_victim_sse", "path": "/secret", "session_id": "session-victim"},
                )
                assert res_att.isError is True
                assert "ARTIFACT_NOT_FOUND" in res_att.content[0].text

        # 2. Legitimate client connects via SSE presenting a valid session token for session-victim
        headers["x-session-token"] = create_session_token("session-victim")
        async with sse_client("http://127.0.0.1:8991/api/v1/mcp/sse", headers=headers) as (r2, w2):
            async with ClientSession(r2, w2) as session2:
                await session2.initialize()
                res_legit = await session2.call_tool(
                    "observation.get",
                    {"ref": "obs_victim_sse", "path": "/secret"},
                )
                assert res_legit.isError is False
                assert "victim-sse-secret" in res_legit.content[0].text
    finally:
        server.should_exit = True
        await server_task


def test_register_observation_capabilities(store: SQLiteMemoryGraphStore) -> None:
    registry = CapabilityRegistry()
    register_observation_capabilities(registry, store)
    assert registry.descriptor("observation.search") is not None
    assert registry.descriptor("observation.filter") is not None
    assert registry.descriptor("observation.slice") is not None
    assert registry.descriptor("observation.count") is not None
    assert registry.descriptor("observation.get") is not None
