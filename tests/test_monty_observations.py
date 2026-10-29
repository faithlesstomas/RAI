"""Tests for Monty REPL observation bindings, publish helper, and data-class propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
from returns.result import Success

from rai.assistant.records import ObservationArtifact, summarize_observation_schema
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.records import DataClass, ProducerIdentity, _new_id
from rai.kernel.references import get_artifact_for_reference
from rai.routers.assistant import REPLExecuteRequest, execute_session_repl
from rai.tools.python import (
    prepare_monty_observation_context,
    run_secure_python_code,
)

_TEST_PROD = ProducerIdentity(producer_id="test.suite", kind="test", version="1.0.0")


def _make_art(
    session_id: str,
    artifact_id: str,
    capability: str,
    data_class: DataClass,
    content: dict[str, Any],
    turn_id: str = "t1",
) -> ObservationArtifact:
    return ObservationArtifact(
        record_id=_new_id(),
        artifact_id=artifact_id,
        session_id=session_id,
        turn_id=turn_id,
        capability=capability,
        request_id=_new_id(),
        data_class=data_class,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=len(str(content)),
        producer=_TEST_PROD,
    )


@pytest.fixture
async def observation_store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    db_file = tmp_path / "obs_test.sqlite3"
    store = SQLiteMemoryGraphStore(db_file)
    init_res = await store.start()
    assert isinstance(init_res, Success)
    return store


@pytest.mark.asyncio
async def test_monty_binds_session_observations(observation_store: SQLiteMemoryGraphStore) -> None:
    """Test that session observations (obs_1, obs mapping) are bound into Monty REPL."""
    session_id = "test-session-obs-bind"

    art1 = _make_art(
        session_id=session_id,
        artifact_id="obs_1",
        capability="application.list",
        data_class=DataClass.PUBLIC,
        content={"results": ["item1", "item2", "item3"], "status": "ok"},
    )
    art2 = _make_art(
        session_id=session_id,
        artifact_id="obs_2",
        capability="system.info",
        data_class=DataClass.LOCAL,
        content={"count": 42, "category": "system"},
    )
    await observation_store.put(art1)
    await observation_store.put(art2)

    code = """
r1 = obs_1["results"]
r2 = obs["obs_1"]["results"]
r3 = obs[1]["results"]
c1 = obs_2["count"]
c2 = obs[2]["count"]

assert r1 == r2 == r3
assert c1 == c2 == 42
ans = len(r1) + c1
"""
    output = await run_secure_python_code(
        code,
        session_id=session_id,
        store=observation_store,
    )

    assert "[Active REPL session state:" in output
    # User variables should be tracked
    assert "ans (int)" in output
    # Observation aliases must NOT be in user variables
    assert "obs_1" not in output.split("[Active REPL session state:")[1].split("]")[0]
    assert "obs " not in output.split("[Active REPL session state:")[1].split("]")[0]

    # Observation schema summary feedback must be present
    assert "[Available session observations:" in output
    assert "obs_1: capability=application.list, data_class=PUBLIC" in output
    assert "obs_2: capability=system.info, data_class=LOCAL" in output


@pytest.mark.asyncio
async def test_monty_publish_sequential_alias(observation_store: SQLiteMemoryGraphStore) -> None:
    """Test that scripts can call publish() to save derived artifacts with sequential obs_N aliases."""
    session_id = "test-session-publish-seq"

    code = """
p1 = publish([10, 20, 30])
p2 = publish({"total": 60, "status": "processed"})
res = f"{p1},{p2}"
res
"""
    output = await run_secure_python_code(
        code,
        session_id=session_id,
        store=observation_store,
    )
    assert "'obs_1,obs_2'" in output or "obs_1,obs_2" in output

    # Check store has persisted artifacts
    saved_arts = (await observation_store.list_for_session(session_id)).unwrap()
    assert len(saved_arts) == 2
    assert saved_arts[0].artifact_id == "obs_1"
    assert list(saved_arts[0].content["items"]) == [10, 20, 30]
    assert saved_arts[0].content["total_count"] == 3
    assert saved_arts[1].artifact_id == "obs_2"
    assert saved_arts[1].content == {"total": 60, "status": "processed"}


@pytest.mark.asyncio
async def test_monty_publish_custom_name_and_ref_pipeline(
    observation_store: SQLiteMemoryGraphStore,
) -> None:
    """Test that published artifact with custom name is retrievable via get_artifact_for_reference."""
    session_id = "test-session-publish-custom"

    code = """
items = [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]
name = publish({"items": items, "count": len(items)}, name="my_filtered_data")
name
"""
    output = await run_secure_python_code(
        code,
        session_id=session_id,
        store=observation_store,
    )
    assert "my_filtered_data" in output

    ref_res = await get_artifact_for_reference(
        observation_store,
        session_id=session_id,
        ref="my_filtered_data",
    )
    assert isinstance(ref_res, Success)
    art = ref_res.unwrap()
    assert art.artifact_id == "my_filtered_data"
    assert art.content["count"] == 2
    assert len(art.content["items"]) == 2


@pytest.mark.asyncio
async def test_monty_data_class_propagation(observation_store: SQLiteMemoryGraphStore) -> None:
    """Test that consuming PRIVATE data taints session and published artifacts."""
    session_id = "test-session-taint"

    art = _make_art(
        session_id=session_id,
        artifact_id="obs_1",
        capability="secure.store",
        data_class=DataClass.PRIVATE,
        content={"key": "highly_sensitive"},
    )
    await observation_store.put(art)

    ctx = await prepare_monty_observation_context(
        session_id=session_id,
        store=observation_store,
        default_data_class=DataClass.PUBLIC,
    )
    assert ctx.session_data_class == DataClass.PRIVATE

    code = """
p1 = publish({"summary": "derived_info"})
p2 = publish({"public_claim": "safe"}, data_class="PUBLIC")
"""
    await run_secure_python_code(
        code,
        session_id=session_id,
        store=observation_store,
    )

    saved_arts = {
        a.artifact_id: a
        for a in (await observation_store.list_for_session(session_id)).unwrap()
    }
    # p1 inherits PRIVATE from session context
    assert saved_arts["obs_2"].data_class == DataClass.PRIVATE
    # p2 tries to claim PUBLIC, but taint prevents degradation
    assert saved_arts["obs_3"].data_class == DataClass.PRIVATE


@pytest.mark.asyncio
async def test_monty_tier2_egress_blocking(observation_store: SQLiteMemoryGraphStore) -> None:
    """Test that Tier-2 Bubblewrap guards against egress for LOCAL and PRIVATE data classes."""
    session_id_local = "test-egress-local"
    await observation_store.put(
        _make_art(
            session_id=session_id_local,
            artifact_id="obs_1",
            capability="local.data",
            data_class=DataClass.LOCAL,
            content={"internal": "data"},
        )
    )

    # LOCAL + allow_network must be rejected
    out_local = await run_secure_python_code(
        "import requests",
        allow_network=True,
        session_id=session_id_local,
        store=observation_store,
    )
    assert "Network access is prohibited for LOCAL data class" in out_local

    session_id_priv = "test-egress-private"
    await observation_store.put(
        _make_art(
            session_id=session_id_priv,
            artifact_id="obs_1",
            capability="private.data",
            data_class=DataClass.PRIVATE,
            content={"secret": "user_id"},
        )
    )

    # PRIVATE + allow_network without require_hitl must be rejected
    out_priv = await run_secure_python_code(
        "import requests",
        allow_network=True,
        require_hitl=False,
        session_id=session_id_priv,
        store=observation_store,
    )
    assert "Egress of PRIVATE data requires explicit user approval" in out_priv


@pytest.mark.asyncio
async def test_assistant_router_repl_with_observations_and_publish(
    observation_store: SQLiteMemoryGraphStore,
) -> None:
    """Test that execute_session_repl handles observation bindings and publish correctly."""
    from unittest.mock import MagicMock
    from rai.assistant.service import AssistantService

    session_id = "test-router-repl-obs"
    await observation_store.put(
        _make_art(
            session_id=session_id,
            artifact_id="obs_1",
            capability="app.list",
            data_class=DataClass.PUBLIC,
            content={"items": [1, 2, 3]},
        )
    )

    mock_service = MagicMock(spec=AssistantService)
    mock_service.store = observation_store
    mock_service.profile_scope = "default"

    res = await execute_session_repl(
        session_id,
        REPLExecuteRequest(code="total = sum(obs_1['items'])\npublish({'total': total}, name='total_art')\ntotal"),
        service=mock_service,
    )

    assert res.status == "ok"
    assert res.output == "6"
    assert res.variables.get("total") == "int"

    # Verify published artifact
    pub_art = (await observation_store.get(session_id, "total_art")).unwrap()
    assert pub_art is not None
    assert pub_art.content["total"] == 6
