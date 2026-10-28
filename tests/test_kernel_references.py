"""Tests for kernel-resolved observation references (Phase 4 of #56)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from returns.result import Failure, Result, Success

from rai.assistant.records import (
    ObservationArtifact,
    summarize_observation_schema,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    RegisteredCapability,
)
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionResult,
    CapabilityRequest,
    DataClass,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
    _new_id,
)
from rai.kernel.references import (
    ArtifactReference,
    extract_references,
    get_artifact_for_reference,
    is_artifact_reference,
    parse_json_pointer,
    preflight_scan_references,
    resolve_arguments,
    traverse_json_pointer,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request

PRODUCER = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")


@pytest.fixture
def temp_store() -> Any:
    tmp_dir = Path(tempfile.mkdtemp(prefix="rai-test-refs-"))
    store = SQLiteMemoryGraphStore(path=tmp_dir / "store.sqlite3")
    yield store


def _make_artifact(
    session_id: str,
    artifact_id: str,
    content: dict[str, Any],
    data_class: DataClass = DataClass.LOCAL,
    expires_at: datetime | None = None,
) -> ObservationArtifact:
    return ObservationArtifact(
        record_id=_new_id(),
        artifact_id=artifact_id,
        session_id=session_id,
        turn_id="turn_1",
        capability="test.produce",
        request_id=_new_id(),
        data_class=data_class,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=100,
        expires_at=expires_at,
        producer=PRODUCER,
    )


def test_is_artifact_reference_and_extraction() -> None:
    assert not is_artifact_reference("obs_1")
    assert not is_artifact_reference(123)
    assert not is_artifact_reference({})
    assert not is_artifact_reference({"ref": "obs_1"})
    assert is_artifact_reference({"$ref": "obs_1"})
    assert is_artifact_reference({"$ref": "obs_1", "path": "/data/0"})

    complex_payload = {
        "action": "filter",
        "input": {"$ref": "obs_1", "path": "/files"},
        "nested": [
            1,
            {"$ref": "obs_2"},
            {"inner": {"$ref": "obs_3", "path": "/meta/tag"}},
        ],
        "plain": "value",
    }
    extracted = extract_references(complex_payload)
    assert len(extracted) == 3
    assert extracted[0] == ArtifactReference(ref="obs_1", path="/files")
    assert extracted[1] == ArtifactReference(ref="obs_2", path="")
    assert extracted[2] == ArtifactReference(ref="obs_3", path="/meta/tag")


def test_json_pointer_parsing_and_traversal() -> None:
    # Parsing
    assert parse_json_pointer("").unwrap() == []
    assert parse_json_pointer("/a/b/c").unwrap() == ["a", "b", "c"]
    assert parse_json_pointer("/~1path/~0tilde").unwrap() == ["/path", "~tilde"]
    assert isinstance(parse_json_pointer("invalid"), Failure)
    assert isinstance(parse_json_pointer("/invalid~escape"), Failure)

    # Traversal
    data = {
        "items": ["apple", "banana", "cherry"],
        "nested": {"key": "secret_val", "/slash": 42},
    }
    assert traverse_json_pointer(data, ["items", "1"]).unwrap() == "banana"
    assert traverse_json_pointer(data, ["nested", "key"]).unwrap() == "secret_val"
    assert traverse_json_pointer(data, ["nested", "/slash"]).unwrap() == 42

    # Errors
    err_key = traverse_json_pointer(data, ["missing"]).failure()
    assert err_key[0] == "PATH_NOT_FOUND"

    err_idx = traverse_json_pointer(data, ["items", "10"]).failure()
    assert err_idx[0] == "PATH_NOT_FOUND"

    err_type = traverse_json_pointer(data, ["nested", "key", "invalid"]).failure()
    assert err_type[0] == "INVALID_POINTER"


@pytest.mark.asyncio
async def test_get_artifact_for_reference_isolation_and_expiry(temp_store: Any) -> None:
    session_a = "session_A"
    session_b = "session_B"

    now = datetime.now(timezone.utc)
    art_valid = _make_artifact(session_a, "obs_1", {"key": "val1"}, DataClass.LOCAL)
    art_expired = _make_artifact(
        session_a, "obs_2", {"key": "val2"}, DataClass.LOCAL, expires_at=now - timedelta(seconds=10)
    )
    assert isinstance(await temp_store.put_observation_artifact(art_valid), Success)
    assert isinstance(await temp_store.put_observation_artifact(art_expired), Success)

    # 1. Missing session_id -> UNAUTHORIZED
    res_no_sess = await get_artifact_for_reference(temp_store, "", "obs_1", now=now)
    assert isinstance(res_no_sess, Failure)
    assert res_no_sess.failure()[0] == "UNAUTHORIZED"

    # 2. Foreign session -> ARTIFACT_NOT_FOUND
    res_foreign = await get_artifact_for_reference(temp_store, session_b, "obs_1", now=now)
    assert isinstance(res_foreign, Failure)
    assert res_foreign.failure()[0] == "ARTIFACT_NOT_FOUND"

    # 3. Expired artifact -> REFERENCE_EXPIRED
    res_exp = await get_artifact_for_reference(temp_store, session_a, "obs_2", now=now)
    assert isinstance(res_exp, Failure)
    assert res_exp.failure()[0] == "REFERENCE_EXPIRED"

    # 4. Valid artifact -> Success
    res_ok = await get_artifact_for_reference(temp_store, session_a, "obs_1", now=now)
    assert isinstance(res_ok, Success)
    assert res_ok.unwrap().artifact_id == "obs_1"


@pytest.mark.asyncio
async def test_preflight_scan_and_data_class_elevation(temp_store: Any) -> None:
    session_id = "session_scan"
    art_pub = _make_artifact(session_id, "obs_pub", {"x": 1}, DataClass.PUBLIC)
    art_local = _make_artifact(session_id, "obs_loc", {"y": 2}, DataClass.LOCAL)
    art_priv = _make_artifact(session_id, "obs_priv", {"z": 3}, DataClass.PRIVATE)

    await temp_store.put_observation_artifact(art_pub)
    await temp_store.put_observation_artifact(art_local)
    await temp_store.put_observation_artifact(art_priv)

    # Base PUBLIC + ref LOCAL -> LOCAL
    args_local = {"data": {"$ref": "obs_loc"}}
    scan1 = (await preflight_scan_references(args_local, temp_store, session_id, DataClass.PUBLIC)).unwrap()
    assert scan1.has_references
    assert scan1.effective_data_class == DataClass.LOCAL
    assert scan1.referenced_ids == ("obs_loc",)

    # Base LOCAL + ref PRIVATE -> PRIVATE
    args_priv = {"a": {"$ref": "obs_pub"}, "b": {"$ref": "obs_priv"}}
    scan2 = (await preflight_scan_references(args_priv, temp_store, session_id, DataClass.LOCAL)).unwrap()
    assert scan2.has_references
    assert scan2.effective_data_class == DataClass.PRIVATE
    assert scan2.referenced_ids == ("obs_priv", "obs_pub")

    # No references -> retains base data class
    args_none = {"plain": "value"}
    scan3 = (await preflight_scan_references(args_none, temp_store, session_id, DataClass.PUBLIC)).unwrap()
    assert not scan3.has_references
    assert scan3.effective_data_class == DataClass.PUBLIC
    assert scan3.referenced_ids == ()


def test_resolve_arguments_deep_substitution() -> None:
    art1 = _make_artifact("s1", "obs_1", {"users": [{"name": "Alice"}, {"name": "Bob"}]})
    art2 = _make_artifact("s1", "obs_2", {"config": {"retries": 3}})
    artifact_map = {"obs_1": art1, "obs_2": art2}

    arguments = {
        "primary_user": {"$ref": "obs_1", "path": "/users/0/name"},
        "all_users": {"$ref": "obs_1", "path": "/users"},
        "full_obj": {"$ref": "obs_2"},
        "constant": 100,
    }

    resolved = resolve_arguments(arguments, artifact_map).unwrap()
    assert resolved["primary_user"] == "Alice"
    assert list(resolved["all_users"]) == [{"name": "Alice"}, {"name": "Bob"}]
    assert resolved["full_obj"] == {"config": {"retries": 3}}
    assert resolved["constant"] == 100


@pytest.mark.asyncio
async def test_capability_service_end_to_end_pipeline(temp_store: Any) -> None:
    session_id = "session_pipeline"
    source_art = _make_artifact(
        session_id,
        "obs_1",
        {"files": ["/var/log/syslog", "/var/log/auth.log"], "owner": "sysadmin"},
        DataClass.LOCAL,
    )
    await temp_store.put_observation_artifact(source_art)

    registry = CapabilityRegistry()

    received_args: dict[str, Any] = {}

    async def echo_handler(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal received_args
        received_args = dict(args)
        return {"status": "SUCCESS", "count": len(args.get("items", []))}

    desc = CapabilityDescriptor(
        name="test.pipeline",
        description="Processes pipeline items",
        input_schema={
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "owner": {"type": "string"},
            },
            "required": ["items", "owner"],
        },
        risk_class=RiskClass.LOW,
        side_effects=(),
        isolation="host-api",
        verification_plan=("observed",),
    )
    registry.register(RegisteredCapability(desc, handler=echo_handler))

    audit = InMemoryAuditLedger()
    policy = PolicyEngine()
    service = CapabilityService(registry, policy, audit, observation_store=temp_store)

    request = normalize_request(
        desc,
        {
            "items": {"$ref": "obs_1", "path": "/files"},
            "owner": {"$ref": "obs_1", "path": "/owner"},
        },
        actor=PRODUCER,
        session_id=session_id,
    )

    decision, result = await service.invoke(request, CancellationToken())
    assert decision is not None
    assert decision.outcome == PolicyOutcome.ALLOW
    assert isinstance(result, Success)

    act_result: ActionResult = result.unwrap()
    assert act_result.output["status"] == "SUCCESS"
    assert act_result.output["count"] == 2

    # Verify handler received resolved concrete values
    assert list(received_args["items"]) == ["/var/log/syslog", "/var/log/auth.log"]
    assert received_args["owner"] == "sysadmin"

    # Verify provenance chain includes obs_1
    assert any(p.source_id == "obs_1" and p.source_type == "observation_artifact" for p in act_result.provenance)

    # Verify audit ledger recorded referenced_artifact_ids
    assert len(audit.entries) == 2
    assert audit.entries[0].stage == "DECISION"
    assert audit.entries[0].referenced_artifact_ids == ("obs_1",)
    assert audit.entries[1].stage == "TERMINAL"
    assert audit.entries[1].referenced_artifact_ids == ("obs_1",)


@pytest.mark.asyncio
async def test_egress_policy_enforcement_on_referenced_data(temp_store: Any) -> None:
    session_id = "session_egress"
    art_local = _make_artifact(session_id, "obs_local", {"data": "internal"}, DataClass.LOCAL)
    art_priv = _make_artifact(session_id, "obs_private", {"secret_note": "personal"}, DataClass.PRIVATE)
    await temp_store.put_observation_artifact(art_local)
    await temp_store.put_observation_artifact(art_priv)

    registry = CapabilityRegistry()
    network_desc = CapabilityDescriptor(
        name="network.export",
        description="Exports data to external URL",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "payload": {"type": "string"},
            },
            "required": ["url", "payload"],
        },
        risk_class=RiskClass.LOW,
        side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )
    registry.register(RegisteredCapability(network_desc, handler=lambda args: {"sent": True}))

    audit = InMemoryAuditLedger()
    policy = PolicyEngine()
    service = CapabilityService(registry, policy, audit, observation_store=temp_store)

    # 1. Base request is PUBLIC, but references LOCAL artifact -> Policy DENY (LOCAL_DATA_EGRESS_FORBIDDEN)
    req_local = normalize_request(
        network_desc,
        {"url": "https://api.example.com", "payload": {"$ref": "obs_local", "path": "/data"}},
        data_class=DataClass.PUBLIC,
        session_id=session_id,
    )
    decision1, res1 = await service.invoke(req_local, CancellationToken())
    assert decision1 is not None
    assert decision1.outcome == PolicyOutcome.DENY
    assert "LOCAL_DATA_EGRESS_FORBIDDEN" in decision1.reason_codes
    assert isinstance(res1, Failure)
    assert res1.failure().code == "POLICY_DENIED"

    # 2. Base request is PUBLIC, but references PRIVATE artifact -> Policy ESCALATE (PRIVATE_DATA_EGRESS)
    req_priv = normalize_request(
        network_desc,
        {"url": "https://api.example.com", "payload": {"$ref": "obs_private", "path": "/secret_note"}},
        data_class=DataClass.PUBLIC,
        session_id=session_id,
    )
    decision2, res2 = await service.invoke(req_priv, CancellationToken())
    assert decision2 is not None
    assert decision2.outcome == PolicyOutcome.ESCALATE
    assert "PRIVATE_DATA_EGRESS" in decision2.reason_codes
    assert isinstance(res2, Failure)
    assert res2.failure().code == "ESCALATION_REQUIRED"
