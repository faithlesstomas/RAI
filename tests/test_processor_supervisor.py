"""
Unit and integration test suite for Stage 4 Processor Supervisor.

Covers:
- GitLab Issue #8: Lazy loading and guards for heavy inference dependencies.
- GitLab Issue #9: Offloading blocking local inference to worker threads.
- ProcessorSupervisor lifecycle, concurrency limits, and idle unloading.
- Stage 4 Acceptance Slice 1: Offline episode to structured Claim with provenance.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import time
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Failure, Result, Success

from rai.inference import (
    GenerationStats,
    InferenceResult,
    LocalTextEngine,
    ProcessorSupervisor,
    get_available_backends,
    is_backend_available,
    load_local_model,
)
from rai.inference.engines.iree import is_iree_available
from rai.inference.engines.llama import LlamaCppEngine, is_llama_cpp_available
from rai.inference.engines.ollama import OllamaEngine
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    Claim,
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    Episode,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    Task,
)

TEST_PRODUCER = ProducerIdentity(
    producer_id="test.stage4", kind="test", version="1.0.0"
)


def _make_budget(max_tokens: int = 100) -> InferenceBudget:
    return InferenceBudget(
        producer=TEST_PRODUCER,
        max_input_tokens=max_tokens,
        max_output_tokens=max_tokens,
        max_agent_turns=1,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0.0,
        max_latency_seconds=30.0,
        max_provider_cost=0.0,
        max_ram_bytes=1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=60),
    )



class MockEngine(LocalTextEngine):
    """Deterministic mock text engine for non-blocking local inference tests."""

    def __init__(
        self,
        model_name: str = "mock-model",
        delay: float = 0.0,
        fail_load: bool = False,
        fail_generate: bool = False,
    ) -> None:
        self._model_name = model_name
        self.delay = delay
        self.fail_load = fail_load
        self.fail_generate = fail_generate
        self._is_loaded = False
        self.generate_calls: list[str] = []
        self.unload_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    async def load(self) -> Result[None, Exception]:
        if self.fail_load:
            return Failure(RuntimeError("Simulated model loading error"))
        self._is_loaded = True
        return Success(None)

    async def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Result[InferenceResult, Exception]:
        self.generate_calls.append(prompt)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail_generate:
            return Failure(RuntimeError("Simulated generation failure"))
        stats = GenerationStats(
            input_tokens=len(prompt.split()),
            output_tokens=10,
            total_time_sec=self.delay,
            tokens_per_sec=10.0 / self.delay if self.delay > 0 else 100.0,
        )
        return Success(
            InferenceResult(
                text="Generated activity summary for episode.",
                stats=stats,
                finish_reason="stop",
            )
        )

    async def unload(self) -> Result[None, Exception]:
        self._is_loaded = False
        self.unload_calls += 1
        return Success(None)


def _make_episode() -> Episode:
    prov = ProvenanceReference(
        source_id="obs:1",
        source_type="observation",
        source_version="1.0.0",
        relation="derived-from",
        producer=TEST_PRODUCER,
    )
    return Episode(
        producer=TEST_PRODUCER,
        started_at=datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 9, 7, 10, 30, tzinfo=timezone.utc),
        observation_ids=("obs:1", "obs:2"),
        provenance=(prov,),
        applications=("firefox.desktop", "code.desktop"),
        resources=("https://example.com", "src/rai/core.py"),
        projects=("rai",),
        activity_types=("coding", "research"),
        confidence=1.0,
        data_class=DataClass.LOCAL,
    )


def _make_context_package(episode: Episode) -> ContextPackage:
    manifest_item = ContextManifestItem(
        source_id=episode.record_id,
        source_type="episode",
        data_class=DataClass.LOCAL,
        fields=("applications", "resources", "projects"),
        redactions=(),
    )
    manifest = ContextManifest(
        destination="local-processor",
        items=(manifest_item,),
        approved=True,
        producer=TEST_PRODUCER,
    )
    return ContextPackage(
        task_id="task:1",
        manifest=manifest,
        content={"episode": episode.model_dump()},
        producer=TEST_PRODUCER,
    )


# --- Tests for GitLab Issue #8 (Lazy loading and guards) ---


def test_issue_8_lazy_imports_and_guards() -> None:
    """Importing engines must be side-effect free and not crash on missing extras."""
    # is_iree_available is False (frozen per roadmap)
    assert not is_iree_available()
    assert not is_backend_available("iree")
    assert not is_backend_available("onnx")

    # llama engine can be initialized even if llama_cpp is absent
    engine = LlamaCppEngine(model_path="/tmp/nonexistent.gguf")
    assert not engine.is_loaded
    assert engine.model_name == "nonexistent.gguf"

    # If llama_cpp is not installed, calling load fails gracefully
    with patch("rai.inference.engines.llama.is_llama_cpp_available", return_value=False):
        assert not is_backend_available("llama")
        res = engine.generate_sync("hello")
        assert isinstance(res, Failure)
        assert isinstance(res.failure(), ImportError)

    # factory load_local_model guards unsupported/frozen backends
    load_onnx = load_local_model("/tmp/test.onnx")
    assert isinstance(load_onnx, Failure)

    backends = get_available_backends()
    assert isinstance(backends, tuple)


# --- Tests for GitLab Issue #9 (Offloading blocking inference) ---


@pytest.mark.asyncio
async def test_issue_9_event_loop_not_blocked() -> None:
    """Inference execution must run offloaded without starving concurrent asyncio tasks."""
    mock_engine = MockEngine(delay=0.1)
    supervisor = ProcessorSupervisor(
        engine=mock_engine,
        idle_unload_seconds=0,
    )
    await supervisor.start()

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.01)
            ticks += 1

    episode = _make_episode()
    context = _make_context_package(episode)
    task = Task(objective="Summarize recent activity", producer=TEST_PRODUCER)
    budget = _make_budget(100)
    cancellation = CancellationToken()

    # Run inference and concurrent ticker task simultaneously
    ticker_task = asyncio.create_task(ticker())
    res = await supervisor.process(task, context, budget, cancellation)
    await ticker_task

    await supervisor.stop()

    assert isinstance(res, Success)
    # The event loop ticked multiple times during inference, proving it wasn't blocked
    assert ticks >= 5


# --- Tests for Idle Unloading (ROADMAP.md WP 4.1) ---


@pytest.mark.asyncio
async def test_idle_unload_triggers_after_threshold() -> None:
    """Model weights must be automatically unloaded after configurable idle duration."""
    mock_engine = MockEngine()
    # Configure tiny idle timeout for testing
    supervisor = ProcessorSupervisor(
        engine=mock_engine,
        idle_unload_seconds=0.2,
    )
    await supervisor.start()

    episode = _make_episode()
    context = _make_context_package(episode)
    task = Task(objective="Summarize recent activity", producer=TEST_PRODUCER)
    budget = _make_budget(100)
    cancellation = CancellationToken()

    # Execute one request to load the model
    res = await supervisor.process(task, context, budget, cancellation)
    assert isinstance(res, Success)
    assert mock_engine.is_loaded

    # Wait for the idle reaper to fire
    await asyncio.sleep(0.4)

    assert not mock_engine.is_loaded
    assert mock_engine.unload_calls >= 1

    await supervisor.stop()


# --- Tests for Concurrency & Capacity (ROADMAP.md WP 4.1) ---


@pytest.mark.asyncio
async def test_concurrency_capacity_limit_rejected() -> None:
    """Requests exceeding concurrency limits must fail cleanly with CAPACITY_EXCEEDED."""
    mock_engine = MockEngine(delay=0.3)
    supervisor = ProcessorSupervisor(
        engine=mock_engine,
        max_concurrency=1,
        idle_unload_seconds=0,
    )
    await supervisor.start()

    episode = _make_episode()
    context = _make_context_package(episode)
    task1 = Task(objective="Task 1", producer=TEST_PRODUCER)
    task2 = Task(objective="Task 2", producer=TEST_PRODUCER)
    budget = _make_budget(50)

    t1 = asyncio.create_task(
        supervisor.process(task1, context, budget, CancellationToken())
    )
    # Brief pause to ensure task1 acquired the semaphore
    await asyncio.sleep(0.02)
    # task2 should exceed capacity quickly
    t2 = asyncio.create_task(
        supervisor.process(task2, context, budget, CancellationToken())
    )

    res1, res2 = await asyncio.gather(t1, t2)
    await supervisor.stop()

    assert isinstance(res1, Success)
    assert isinstance(res2, Failure)
    failure = res2.failure()
    assert isinstance(failure, ActionFailure)
    assert failure.code == "CAPACITY_EXCEEDED"
    assert failure.retryable is True


# --- Tests for Cancellation ---


@pytest.mark.asyncio
async def test_cancellation_returns_single_terminal_failure() -> None:
    """Pre-cancelled token must immediately return CANCELLED failure."""
    mock_engine = MockEngine()
    supervisor = ProcessorSupervisor(engine=mock_engine, idle_unload_seconds=0)
    await supervisor.start()

    cancellation = CancellationToken()
    cancellation.cancel()

    episode = _make_episode()
    context = _make_context_package(episode)
    task = Task(objective="Cancelled task", producer=TEST_PRODUCER)
    budget = _make_budget(100)

    res = await supervisor.process(task, context, budget, cancellation)
    await supervisor.stop()

    assert isinstance(res, Failure)
    assert res.failure().code == "CANCELLED"
    assert res.failure().terminal is True


# --- Stage 4 Acceptance Slice 1 ---


@pytest.mark.asyncio
async def test_stage4_acceptance_slice_1_offline_episode_to_claim() -> None:
    """
    Acceptance slice 1: A local processor converts an episode into validated
    structured output while offline; processor failure does not affect the original episode.
    """
    mock_engine = MockEngine(model_name="qwen2.5:1.5b")
    supervisor = ProcessorSupervisor(
        engine=mock_engine,
        name="test-supervisor",
        idle_unload_seconds=0,
    )
    await supervisor.start()

    episode = _make_episode()
    context = _make_context_package(episode)
    task = Task(
        objective="Summarize the active applications and goals for this episode",
        correlation_id="corr:abc",
        producer=TEST_PRODUCER,
    )
    budget = _make_budget(150)
    cancellation = CancellationToken()

    # Step 1: Successful offline processing to Claim
    result = await supervisor.process(task, context, budget, cancellation)
    assert isinstance(result, Success)

    claim: Claim = result.unwrap()
    assert claim.record_type == "claim"
    assert claim.correlation_id == "corr:abc"
    assert claim.epistemic_status == "inferred"
    assert claim.data_class == DataClass.LOCAL
    assert claim.confidence == 0.9
    assert len(claim.provenance) == 1
    assert claim.provenance[0].source_id == context.record_id
    assert claim.provenance[0].relation == "derived-from"
    assert "Generated activity summary for episode." in claim.statement

    # Step 2: Processor failure does not affect original episode
    mock_engine.fail_generate = True
    fail_result = await supervisor.process(task, context, budget, cancellation)
    assert isinstance(fail_result, Failure)
    assert fail_result.failure().code == "INFERENCE_FAILED"

    # Verify original episode remains immutable and identical
    assert episode.applications == ("firefox.desktop", "code.desktop")
    assert episode.projects == ("rai",)
    assert episode.data_class == DataClass.LOCAL

    await supervisor.stop()


# --- OllamaEngine Integration Unit Test ---


@pytest.mark.asyncio
async def test_ollama_engine_generate_and_unload() -> None:
    """Test OllamaEngine client interaction, options, stats, and keep_alive=0 unload."""
    engine = OllamaEngine(model_name="llama3.2:1b")
    mock_client = AsyncMock()

    mock_client.show.return_value = {"model_info": {}}
    mock_client.generate.return_value = {
        "response": "Ollama generated text.",
        "eval_count": 15,
        "prompt_eval_count": 5,
        "done": True,
    }

    with patch.object(engine, "_get_client", return_value=mock_client):
        # Load
        load_res = await engine.load()
        assert isinstance(load_res, Success)
        assert engine.is_loaded

        # Generate
        gen_res = await engine.generate("Test prompt", stop=["\n"], max_tokens=100)
        assert isinstance(gen_res, Success)
        inference_result = gen_res.unwrap()
        assert inference_result.text == "Ollama generated text."
        assert inference_result.stats is not None
        assert inference_result.stats.output_tokens == 15
        assert inference_result.stats.input_tokens == 5

        # Unload (keep_alive=0)
        unload_res = await engine.unload()
        assert isinstance(unload_res, Success)
        assert not engine.is_loaded
        mock_client.generate.assert_called_with(
            model="llama3.2:1b", prompt="", keep_alive=0
        )


# --- Supervisor Health Reporting Test ---


@pytest.mark.asyncio
async def test_supervisor_health_reporting() -> None:
    mock_engine = MockEngine(model_name="test-engine")
    supervisor = ProcessorSupervisor(
        engine=mock_engine,
        name="supervisor-health",
        idle_unload_seconds=0,
    )
    health_before = supervisor.health()
    assert health_before.state == LifecycleState.CREATED
    assert health_before.loaded_model is None

    await supervisor.start()
    health_running = supervisor.health()
    assert health_running.state == LifecycleState.RUNNING

    await supervisor.stop()
    health_stopped = supervisor.health()
    assert health_stopped.state == LifecycleState.STOPPED


# --- Factory Function Tests ---


def test_factory_backend_availability() -> None:
    assert not is_backend_available("onnx")
    assert not is_backend_available("nonexistent_backend")
    assert not is_iree_available()
    backends = get_available_backends()
    assert isinstance(backends, tuple)


def test_factory_load_local_model_validation(tmp_path: Any) -> None:
    # Non-existent file
    missing = tmp_path / "missing.gguf"
    res = load_local_model(str(missing))
    assert isinstance(res, Failure)
    assert isinstance(res.failure(), FileNotFoundError)

    # Directory instead of file
    res_dir = load_local_model(str(tmp_path))
    assert isinstance(res_dir, Failure)
    assert isinstance(res_dir.failure(), IsADirectoryError)

    # Unsupported extension
    dummy_txt = tmp_path / "model.txt"
    dummy_txt.write_text("not a model")
    res_ext = load_local_model(str(dummy_txt))
    assert isinstance(res_ext, Failure)
    assert isinstance(res_ext.failure(), ValueError)

    # ONNX extension (frozen)
    dummy_onnx = tmp_path / "model.onnx"
    dummy_onnx.write_text("onnx")
    res_onnx = load_local_model(str(dummy_onnx))
    assert isinstance(res_onnx, Failure)
    assert isinstance(res_onnx.failure(), NotImplementedError)

    # IREE extension
    dummy_vmfb = tmp_path / "model.vmfb"
    dummy_vmfb.write_text("vmfb")
    res_iree = load_local_model(str(dummy_vmfb))
    assert isinstance(res_iree, Success)

    # Ollama by explicit backend (no file check needed)
    res_ollama = load_local_model("llama3.2", backend="ollama")
    assert isinstance(res_ollama, Success)

