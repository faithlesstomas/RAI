"""Tests for semantic map-reduce and llm_query in Monty REPL."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from returns.result import Success

from rai.assistant.records import (
    ObservationArtifact,
    ProducerIdentity,
    summarize_observation_schema,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, InferenceBudget
from rai.tools.python import (
    prepare_monty_observation_context,
    run_secure_python_code,
)


@pytest.fixture
def repl_producer() -> ProducerIdentity:
    return ProducerIdentity(
        producer_id="rai.tools.python.repl", kind="repl", version="1.0.0"
    )


@pytest.fixture
async def observation_store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    db_file = tmp_path / "obs_test.sqlite3"
    store = SQLiteMemoryGraphStore(db_file)
    init_res = await store.start()
    assert isinstance(init_res, Success)
    return store


@pytest.mark.asyncio
async def test_chunks_text_and_records(
    repl_producer: ProducerIdentity,
    observation_store: SQLiteMemoryGraphStore,
) -> None:
    """chunks splits text with overlap and lists into sub-lists."""
    content = {"text": "A" * 5000}
    art = ObservationArtifact(
        record_id="rec_1",
        artifact_id="obs_1",
        session_id="session_test",
        turn_id="turn_1",
        capability="browser.read_page",
        request_id="req_1",
        data_class=DataClass.PUBLIC,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=5000,
        producer=repl_producer,
    )
    await observation_store.put(art)

    ctx = await prepare_monty_observation_context(
        session_id="session_test",
        store=observation_store,
    )

    chunks_fn = ctx.external_lookup["chunks"]

    # 1. Target as alias obs_1
    c1 = chunks_fn("obs_1", size=2000, overlap=200)
    assert len(c1) == 3
    assert len(c1[0]) == 2000
    assert len(c1[1]) == 2000
    assert len(c1[2]) == 1400

    # 2. Target as raw list of items
    items = list(range(100))
    c2 = chunks_fn(items, size=25, overlap=5)
    assert len(c2) == 5
    assert len(c2[0]) == 25
    assert c2[0][:5] == [0, 1, 2, 3, 4]
    assert c2[1][:5] == [20, 21, 22, 23, 24]  # 5-item overlap


@pytest.mark.asyncio
async def test_llm_query_with_handler() -> None:
    """llm_query executes bounded_task_handler cleanly and synchronously in REPL."""
    called_queries = []

    def mock_handler(chunk_text: str, question: str) -> str:
        called_queries.append((chunk_text, question))
        return f"Answer for: {question}"

    ctx = await prepare_monty_observation_context(
        session_id="session_test",
        bounded_task_handler=mock_handler,
    )

    llm_query = ctx.external_lookup["llm_query"]
    ans = llm_query("Some chunk text", "What is the meaning?")
    assert ans == "Answer for: What is the meaning?"
    assert len(called_queries) == 1
    assert called_queries[0] == ("Some chunk text", "What is the meaning?")


@pytest.mark.asyncio
async def test_llm_query_with_supervisor() -> None:
    """llm_query invokes processor supervisor process_bounded via worker thread."""
    claim_mock = MagicMock()
    claim_mock.statement = "Extracted claim result"

    supervisor_mock = MagicMock()
    supervisor_mock.process_bounded = AsyncMock(return_value=Success(claim_mock))

    ctx = await prepare_monty_observation_context(
        session_id="session_test",
        supervisor=supervisor_mock,
    )

    llm_query = ctx.external_lookup["llm_query"]
    ans = await asyncio.to_thread(llm_query, "Chunk payload", "Find the claim")
    assert ans == "Extracted claim result"
    supervisor_mock.process_bounded.assert_called_once()


@pytest.mark.asyncio
async def test_map_reduce_needle_in_200kb(
    repl_producer: ProducerIdentity,
    observation_store: SQLiteMemoryGraphStore,
) -> None:
    """map_reduce extracts needle located near the end of a 200 kB synthetic document."""
    # Build 200 kB document: mostly filler lines, with a distinct secret near the end
    filler = "This is boilerplate content line that describes background data.\n" * 3000
    secret_line = "CRITICAL FINDING: The decryption token is ALPHA-OMEGA-999.\n"
    more_filler = "Trailing boilerplate lines following the secret data.\n" * 500
    doc_text = filler + secret_line + more_filler
    assert len(doc_text) > 180_000

    content = {"text": doc_text}
    art = ObservationArtifact(
        record_id="rec_big",
        artifact_id="obs_1",
        session_id="session_needle",
        turn_id="turn_1",
        capability="browser.read_page",
        request_id="req_big",
        data_class=DataClass.PUBLIC,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=len(doc_text),
        producer=repl_producer,
    )
    await observation_store.put(art)

    ctx = await prepare_monty_observation_context(
        session_id="session_needle",
        store=observation_store,
    )

    map_reduce_fn = ctx.external_lookup["map_reduce"]

    # Using default fallback heuristic in llm_query (no mock handler needed)
    result = map_reduce_fn("obs_1", "decryption token", size=4000, overlap=400)
    assert "ALPHA-OMEGA-999" in result


@pytest.mark.asyncio
async def test_inference_budget_guard_limits(repl_producer: ProducerIdentity) -> None:
    """InferenceBudget sub-call and token limits are strictly enforced."""
    budget = InferenceBudget(
        record_id="budget_1",
        producer=repl_producer,
        max_input_tokens=100,  # very small token budget
        max_output_tokens=50,
        max_agent_turns=1,
        max_tool_calls=2,  # max 2 sub-calls
        max_images=0,
        max_audio_seconds=0.0,
        max_latency_seconds=10.0,
        max_provider_cost=0.0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    # 1. Test token exhaustion
    ctx_token = await prepare_monty_observation_context(
        session_id="session_b",
        budget=budget,
    )
    llm_query = ctx_token.external_lookup["llm_query"]
    with pytest.raises(RuntimeError, match="token budget .* exhausted"):
        # est_tokens will be ~500 > 100
        llm_query("X" * 2000, "query")

    # 2. Test max sub-calls
    budget_calls = InferenceBudget(
        record_id="budget_2",
        producer=repl_producer,
        max_input_tokens=10000,
        max_output_tokens=50,
        max_agent_turns=1,
        max_tool_calls=2,  # allow only 2 calls
        max_images=0,
        max_audio_seconds=0.0,
        max_latency_seconds=10.0,
        max_provider_cost=0.0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    ctx_calls = await prepare_monty_observation_context(
        session_id="session_c",
        budget=budget_calls,
    )
    llm_query_calls = ctx_calls.external_lookup["llm_query"]
    llm_query_calls("short", "q1")
    llm_query_calls("short", "q2")
    with pytest.raises(RuntimeError, match="maximum sub-calls .* exceeded"):
        llm_query_calls("short", "q3")


@pytest.mark.asyncio
async def test_cancellation_propagation(repl_producer: ProducerIdentity) -> None:
    """Cancelled CancellationToken immediately aborts llm_query."""
    cancellation = CancellationToken()
    cancellation.cancel()

    ctx = await prepare_monty_observation_context(
        session_id="session_canc",
        cancellation=cancellation,
    )
    llm_query = ctx.external_lookup["llm_query"]
    with pytest.raises(RuntimeError, match="Operation cancelled"):
        llm_query("some text", "some query")


@pytest.mark.asyncio
async def test_deadline_expiration(repl_producer: ProducerIdentity) -> None:
    """Expired cancellation_deadline raises InferenceBudget error."""
    budget = InferenceBudget(
        record_id="budget_exp",
        producer=repl_producer,
        max_input_tokens=1000,
        max_output_tokens=50,
        max_agent_turns=1,
        max_tool_calls=5,
        max_images=0,
        max_audio_seconds=0.0,
        max_latency_seconds=10.0,
        max_provider_cost=0.0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    ctx = await prepare_monty_observation_context(
        session_id="session_dead",
        budget=budget,
    )
    llm_query = ctx.external_lookup["llm_query"]
    with pytest.raises(RuntimeError, match="cancellation deadline expired"):
        llm_query("chunk text", "question")


@pytest.mark.asyncio
async def test_run_secure_python_code_with_map_reduce(
    repl_producer: ProducerIdentity,
    observation_store: SQLiteMemoryGraphStore,
) -> None:
    """run_secure_python_code executes code using chunks and llm_query in Monty."""
    content = {
        "text": "Line 1: system start.\nLine 2: Target ID is SEC-42.\nLine 3: done."
    }
    art = ObservationArtifact(
        record_id="rec_e2e",
        artifact_id="obs_1",
        session_id="session_e2e",
        turn_id="turn_1",
        capability="browser.read_page",
        request_id="req_e2e",
        data_class=DataClass.PUBLIC,
        content=content,
        schema_summary=summarize_observation_schema(content),
        size_chars=80,
        producer=repl_producer,
    )
    await observation_store.put(art)

    code = """
sub_chunks = chunks(obs_1, size=50, overlap=10)
answer = llm_query(sub_chunks[0], "Target ID")
published_id = publish({"result": answer}, name="extracted_target")
f"Extracted: {answer}"
"""
    output = await run_secure_python_code(
        code,
        session_id="session_e2e",
        store=observation_store,
    )
    assert "Extracted: Line 2: Target ID is SEC-42." in output
