"""Policy, privacy and resource-bound tests for bounded inference caching."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, List, Optional

import pytest
from returns.result import Failure, Result, Success

from rai.inference import (
    BoundedResultCacheKey,
    BoundedTaskKind,
    CacheLookupStatus,
    CacheMissReason,
    InferenceResult,
    ProcessorSupervisor,
    SQLiteBoundedResultCache,
)
from rai.container import ApplicationContainer
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    Task,
)

TEST_PRODUCER = ProducerIdentity(
    producer_id="test.inference-cache", kind="test", version="1.0.0"
)
VALID_OUTPUT = (
    '{"summary":"Worked on RAI.","key_points":["Reviewed cache"],'
    '"confidence":0.9}'
)
CACHE_CAPACITY = 8
EXPECTED_THREE_GENERATIONS = 3
EXPECTED_TWO_GENERATIONS = 2


def _key(**updates: str) -> BoundedResultCacheKey:
    values = {
        "model_name": "model-a",
        "model_artifact_version": "sha256:artifact-a",
        "task_kind": "episode_summarization",
        "contract_version": "1.0.0",
        "prompt_version": "1.0.0",
        "policy_version": "1.0.0",
    }
    values.update(updates)
    return BoundedResultCacheKey.build(
        **values,
        normalized_input={"content": {"b": 2, "a": 1}},
    )


def _budget() -> InferenceBudget:
    return InferenceBudget(
        producer=TEST_PRODUCER,
        max_input_tokens=10_000,
        max_output_tokens=1_000,
        max_agent_turns=1,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=8 * 1024**3,
        max_vram_bytes=4 * 1024**3,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=60),
    )


def _context(data_class: DataClass = DataClass.LOCAL) -> ContextPackage:
    return ContextPackage(
        task_id="task:cache",
        producer=TEST_PRODUCER,
        manifest=ContextManifest(
            destination="local-processor",
            approved=True,
            producer=TEST_PRODUCER,
            items=(
                ContextManifestItem(
                    source_id="episode",
                    source_type="episode",
                    data_class=data_class,
                    fields=("applications",),
                ),
            ),
        ),
        content={"episode": {"applications": ["code.desktop"]}},
    )


class ScriptedEngine:
    """Async deterministic engine with observable generation calls."""

    model_name = "cache-test-model"

    def __init__(self, response: str = VALID_OUTPUT) -> None:
        self.response = response
        self.is_loaded = False
        self.generate_calls = 0

    async def load(self) -> Result[None, Exception]:
        self.is_loaded = True
        return Success(None)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        del prompt, stop, max_tokens, temperature
        self.generate_calls += 1
        return Success(InferenceResult(text=self.response))

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[Result[str, Exception]]:
        del prompt, stop, max_tokens, temperature
        if False:
            yield Success("")

    async def unload(self) -> Result[None, Exception]:
        self.is_loaded = False
        return Success(None)


def test_cache_key_is_stable_and_versions_are_isolated() -> None:
    first = _key()
    equivalent = BoundedResultCacheKey.build(
        model_name="model-a",
        model_artifact_version="sha256:artifact-a",
        task_kind="episode_summarization",
        contract_version="1.0.0",
        prompt_version="1.0.0",
        policy_version="1.0.0",
        normalized_input={"content": {"a": 1, "b": 2}},
    )

    assert first == equivalent
    assert first.digest != _key(model_artifact_version="sha256:artifact-b").digest
    assert first.digest != _key(contract_version="2.0.0").digest
    assert first.digest != _key(policy_version="2.0.0").digest
    assert "content" not in repr(first)


@pytest.mark.parametrize(
    ("ttl_seconds", "max_entries"),
    ((0, 1), (1, 0)),
)
def test_cache_rejects_unbounded_configuration(
    tmp_path: Path, ttl_seconds: int, max_entries: int
) -> None:
    with pytest.raises(ValueError):
        SQLiteBoundedResultCache(
            tmp_path / "cache.sqlite3",
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
        )


@pytest.mark.asyncio
async def test_storage_failure_is_a_typed_cache_miss(tmp_path: Path) -> None:
    cache = SQLiteBoundedResultCache(tmp_path)

    lookup = await cache.lookup(_key())

    assert lookup.metadata.status == CacheLookupStatus.MISS
    assert lookup.metadata.miss_reason == CacheMissReason.ERROR


@pytest.mark.asyncio
async def test_sqlite_cache_enforces_ttl_and_lru_capacity(tmp_path: Path) -> None:
    now = [1_000.0]
    cache = SQLiteBoundedResultCache(
        tmp_path / "cache.sqlite3",
        ttl_seconds=10,
        max_entries=2,
        clock=lambda: now[0],
    )
    first = _key(model_name="a")
    second = _key(model_name="b")
    third = _key(model_name="c")

    await cache.store(
        first,
        payload_json=VALID_OUTPUT,
        input_data_class=DataClass.LOCAL,
        result_data_class=DataClass.LOCAL,
    )
    now[0] += 1
    await cache.store(
        second,
        payload_json=VALID_OUTPUT,
        input_data_class=DataClass.LOCAL,
        result_data_class=DataClass.LOCAL,
    )
    now[0] += 1
    assert (await cache.lookup(first)).metadata.status == CacheLookupStatus.HIT
    now[0] += 1
    await cache.store(
        third,
        payload_json=VALID_OUTPUT,
        input_data_class=DataClass.LOCAL,
        result_data_class=DataClass.LOCAL,
    )

    assert (await cache.lookup(first)).metadata.status == CacheLookupStatus.HIT
    assert (await cache.lookup(second)).metadata.miss_reason == CacheMissReason.NOT_FOUND
    assert (await cache.lookup(third)).metadata.status == CacheLookupStatus.HIT

    now[0] += 11
    expired = await cache.lookup(first)
    assert expired.metadata.status == CacheLookupStatus.MISS
    assert expired.metadata.miss_reason == CacheMissReason.EXPIRED
    assert oct((tmp_path / "cache.sqlite3").stat().st_mode & 0o777) == "0o600"


@pytest.mark.asyncio
async def test_sqlite_cache_supports_concurrent_bounded_access(tmp_path: Path) -> None:
    cache = SQLiteBoundedResultCache(
        tmp_path / "cache.sqlite3", ttl_seconds=60, max_entries=CACHE_CAPACITY
    )
    keys = tuple(_key(model_name=f"model-{index}") for index in range(24))

    await asyncio.gather(
        *(
            cache.store(
                key,
                payload_json=VALID_OUTPUT,
                input_data_class=DataClass.LOCAL,
                result_data_class=DataClass.LOCAL,
            )
            for key in keys
        )
    )
    lookups = await asyncio.gather(
        *(cache.lookup(key) for key in keys)
    )

    assert sum(
        lookup.metadata.status == CacheLookupStatus.HIT for lookup in lookups
    ) == CACHE_CAPACITY


@pytest.mark.asyncio
async def test_supervisor_reuses_validated_payload_and_rebinds_provenance(
    tmp_path: Path,
) -> None:
    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    engine = ScriptedEngine()
    supervisor = ProcessorSupervisor(
        engine=engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)
    first_context = _context()
    second_context = _context()

    first = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        first_context,
        _budget(),
        CancellationToken(),
    )
    second = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        second_context,
        _budget(),
        CancellationToken(),
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert engine.generate_calls == 1
    assert second.unwrap().provenance[0].source_id == second_context.record_id
    assert second.unwrap().provenance[0].source_id != first_context.record_id
    assert supervisor.last_cache_lookup is not None
    assert supervisor.last_cache_lookup.status == CacheLookupStatus.HIT
    assert supervisor.health().cache_hits == 1
    assert supervisor.health().cache_misses == 1
    await supervisor.stop()


@pytest.mark.asyncio
async def test_privacy_classification_and_policy_versions_do_not_cross_reuse(
    tmp_path: Path,
) -> None:
    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    engine = ScriptedEngine()
    supervisor = ProcessorSupervisor(
        engine=engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
        policy_version="1.0.0",
    )
    await supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)

    for data_class in (DataClass.LOCAL, DataClass.PRIVATE):
        result = await supervisor.process_bounded(
            BoundedTaskKind.EPISODE_SUMMARIZATION,
            task,
            _context(data_class),
            _budget(),
            CancellationToken(),
        )
        assert isinstance(result, Success)
        assert result.unwrap().data_class == data_class
    await supervisor.stop()

    changed_policy = ProcessorSupervisor(
        engine=engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
        policy_version="2.0.0",
    )
    await changed_policy.start()
    result = await changed_policy.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        _context(DataClass.LOCAL),
        _budget(),
        CancellationToken(),
    )
    assert isinstance(result, Success)
    assert engine.generate_calls == EXPECTED_THREE_GENERATIONS
    await changed_policy.stop()


@pytest.mark.asyncio
async def test_cache_hit_cannot_bypass_provider_allow_list(tmp_path: Path) -> None:
    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    engine = ScriptedEngine()
    supervisor = ProcessorSupervisor(
        engine=engine,
        backend="test-backend",
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)
    context = _context()
    primed = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(primed, Success)

    disallowed_budget = _budget().model_copy(
        update={"allowed_providers": ("different-model",)}
    )
    denied = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        disallowed_budget,
        CancellationToken(),
    )
    assert isinstance(denied, Failure)
    assert denied.failure().code == "PROVIDER_DISALLOWED"
    assert engine.generate_calls == 1
    await supervisor.stop()


@pytest.mark.asyncio
async def test_invalid_generated_or_cached_payload_is_never_reused(
    tmp_path: Path,
) -> None:
    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    engine = ScriptedEngine("not-json")
    supervisor = ProcessorSupervisor(
        engine=engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)
    context = _context()

    invalid = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(invalid, Failure)
    engine.response = VALID_OUTPUT
    valid = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(valid, Success)
    assert engine.generate_calls == EXPECTED_TWO_GENERATIONS

    assert supervisor.last_cache_lookup is not None
    key = BoundedResultCacheKey(
        digest=supervisor.last_cache_lookup.key_digest,
        model_name=supervisor.last_cache_lookup.model_name,
        model_artifact_version=(
            supervisor.last_cache_lookup.model_artifact_version
        ),
        task_kind=supervisor.last_cache_lookup.task_kind,
        contract_version=supervisor.last_cache_lookup.contract_version,
        prompt_version=supervisor.last_cache_lookup.prompt_version,
        policy_version=supervisor.last_cache_lookup.policy_version,
        version=supervisor.last_cache_lookup.key_version,
    )
    await cache.store(
        key,
        payload_json="not-json",
        input_data_class=DataClass.LOCAL,
        result_data_class=DataClass.LOCAL,
    )
    recovered = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(recovered, Success)
    assert engine.generate_calls == EXPECTED_THREE_GENERATIONS
    await supervisor.stop()


@pytest.mark.asyncio
async def test_result_cancelled_after_generation_is_not_cached(tmp_path: Path) -> None:
    token = CancellationToken()

    class CancellingEngine(ScriptedEngine):
        async def generate(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            max_tokens: int = 1024,
            temperature: float = 0.7,
        ) -> Result[InferenceResult, Exception]:
            result = await super().generate(prompt, stop, max_tokens, temperature)
            token.cancel()
            return result

    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    cancelling_engine = CancellingEngine()
    supervisor = ProcessorSupervisor(
        engine=cancelling_engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)
    context = _context()
    cancelled = await supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        token,
    )
    assert isinstance(cancelled, Failure)
    assert cancelled.failure().code == "CANCELLED"
    await supervisor.stop()

    replacement = ScriptedEngine()
    retry = ProcessorSupervisor(
        engine=replacement,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await retry.start()
    result = await retry.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(result, Success)
    assert replacement.generate_calls == 1
    await retry.stop()


@pytest.mark.asyncio
async def test_valid_but_partial_generation_is_not_cached(tmp_path: Path) -> None:
    class PartialEngine(ScriptedEngine):
        async def generate(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            max_tokens: int = 1024,
            temperature: float = 0.7,
        ) -> Result[InferenceResult, Exception]:
            await super().generate(prompt, stop, max_tokens, temperature)
            return Success(InferenceResult(text=self.response, finish_reason="length"))

    cache = SQLiteBoundedResultCache(tmp_path / "cache.sqlite3")
    partial_engine = PartialEngine()
    first_supervisor = ProcessorSupervisor(
        engine=partial_engine,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await first_supervisor.start()
    task = Task(objective="Summarize", producer=TEST_PRODUCER)
    context = _context()
    partial = await first_supervisor.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(partial, Success)
    await first_supervisor.stop()

    replacement = ScriptedEngine()
    retry = ProcessorSupervisor(
        engine=replacement,
        idle_unload_seconds=0,
        result_cache=cache,
        model_artifact_version="sha256:test-artifact",
    )
    await retry.start()
    result = await retry.process_bounded(
        BoundedTaskKind.EPISODE_SUMMARIZATION,
        task,
        context,
        _budget(),
        CancellationToken(),
    )
    assert isinstance(result, Success)
    assert replacement.generate_calls == 1
    await retry.stop()


@pytest.mark.asyncio
async def test_container_enables_cache_only_with_artifact_version(
    tmp_path: Path,
) -> None:
    without_version = ApplicationContainer(config={}, testing=True)
    assert without_version.processor_supervisor.result_cache is None
    await without_version.close()

    cache_path = tmp_path / "configured-cache.sqlite3"
    configured = ApplicationContainer(
        config={
            "local_ai": {
                "model_artifact_version": "sha256:test-artifact",
                "result_cache": {"path": str(cache_path)},
            }
        },
        testing=True,
    )
    assert isinstance(
        configured.processor_supervisor.result_cache, SQLiteBoundedResultCache
    )
    assert configured.processor_supervisor.model_artifact_version == (
        "sha256:test-artifact"
    )
    await configured.close()
