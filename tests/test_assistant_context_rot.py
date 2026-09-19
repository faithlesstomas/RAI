"""Unit tests for protocol-v2 context-rot evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.context_rot_evaluation import (
    ContextRotCorpus,
    estimate_haystack_layout,
    estimate_tokens,
    evaluate_context_rot,
    format_context_rot_summary_table,
    generate_haystack,
    load_context_rot_corpus,
    save_context_rot_report,
)
from rai.assistant.records import AssistantCandidate, InferenceRequest
from rai.kernel.ports import CancellationToken

_FIXTURE_PATH = Path("tests/fixtures/assistant/v2/context-rot.corpus.json")
_MIN_SCENARIOS = 6
_MIN_DISTRACTORS = 10
_LAYOUT_TARGET_TOKENS = 2000
_START_MAX_DEPTH = 0.25
_MIDDLE_MIN_DEPTH = 0.35
_MIDDLE_MAX_DEPTH = 0.65
_END_MIN_DEPTH = 0.75
_CONFIGURED_CONTEXT_WINDOW = 8192
_EXPECTED_SUCCESS_CASES = 10
_EXPECTED_GRAPH_CASES = 2
_EXPECTED_FAILURE_CASES = 4


class _GroundingOverrideBackend(DeterministicAssistantBackend):
    """Return disagreeing model and delivered outputs for isolation testing."""

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Success[AssistantCandidate]:
        del request, cancellation
        return Success(
            AssistantCandidate(
                text="Według zapisanej informacji wartość to 5433.",
                tokens_in=111,
                tokens_out=7,
                metadata={
                    "raw_model_output": "Nie wiem.",
                    "grounding_override": True,
                },
            )
        )


class _SlowBackend(DeterministicAssistantBackend):
    """Sleep long enough to exercise the evaluator's own timeout."""

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Success[AssistantCandidate]:
        await asyncio.sleep(0.02)
        return await super().generate(request, cancellation)


def _single_scenario_corpus() -> ContextRotCorpus:
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    return ContextRotCorpus(
        corpus_version=corpus.corpus_version,
        description=corpus.description,
        scenarios=corpus.scenarios[:1],
        distractors=corpus.distractors[:3],
    )


def test_load_context_rot_corpus() -> None:
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    assert isinstance(corpus, ContextRotCorpus)
    assert corpus.corpus_version == "rai-assistant-context-rot-corpus-v2"
    assert len(corpus.scenarios) >= _MIN_SCENARIOS
    assert len(corpus.distractors) >= _MIN_DISTRACTORS
    assert {scenario.task_family for scenario in corpus.scenarios} == {
        "literal_retrieval",
        "semantic_retrieval",
    }


def test_generate_haystack_records_estimated_needle_depths() -> None:
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    needle_user = "TEST_NEEDLE_USER_FACT_42"
    needle_assistant = "TEST_NEEDLE_ASSISTANT_ACK"

    for depth in ("start", "middle", "end"):
        turns = generate_haystack(
            distractors=corpus.distractors,
            needle_user=needle_user,
            needle_assistant=needle_assistant,
            target_tokens=_LAYOUT_TARGET_TOKENS,
            depth=depth,  # type: ignore[arg-type]
        )
        estimated_tokens, relative_depth = estimate_haystack_layout(
            turns, needle_user, needle_assistant
        )
        assert estimated_tokens >= _LAYOUT_TARGET_TOKENS
        assert relative_depth is not None
        if depth == "start":
            assert relative_depth <= _START_MAX_DEPTH
        elif depth == "middle":
            assert _MIDDLE_MIN_DEPTH <= relative_depth <= _MIDDLE_MAX_DEPTH
        else:
            assert relative_depth >= _END_MIN_DEPTH


def test_estimate_tokens_handles_empty_and_scales_monotonically() -> None:
    short_text = "Cześć, jak się masz?"
    assert estimate_tokens("") == 0
    assert estimate_tokens(short_text) >= 1
    assert estimate_tokens(short_text) < estimate_tokens(short_text * 50)


@pytest.mark.asyncio
async def test_evaluate_context_rot_v2_uses_one_graph_control() -> None:
    result = await evaluate_context_rot(
        DeterministicAssistantBackend(),
        corpus=_single_scenario_corpus(),
        token_steps=(200, 500),
        depths=("start", "end"),
        trials=1,
        max_output_tokens=64,
        max_latency_seconds=10.0,
        configured_context_window=_CONFIGURED_CONTEXT_WINDOW,
    )

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.artifact_version == "rai-assistant-context-rot-v2"
    assert report.configured_context_window == _CONFIGURED_CONTEXT_WINDOW
    assert report.total_cases == _EXPECTED_SUCCESS_CASES
    assert report.successful_cases == _EXPECTED_SUCCESS_CASES
    assert report.failed_cases == 0

    graph_measurements = [
        measurement
        for measurement in report.measurements
        if measurement.strategy == "graph_memory"
    ]
    assert len(graph_measurements) == _EXPECTED_GRAPH_CASES
    assert all(measurement.target_context_tokens == 0 for measurement in graph_measurements)
    assert all(measurement.depth is None for measurement in graph_measurements)

    table = format_context_rot_summary_table(report)
    assert "CONTEXT ROT V2" in table
    assert "Model Recall" in table
    assert "E2E ms" in table
    assert "TTFT" not in table

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.json"
        save_context_rot_report(report, output_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["artifact_version"] == "rai-assistant-context-rot-v2"
        assert len(data["measurements"]) == report.total_cases


@pytest.mark.asyncio
async def test_grounded_delivery_never_replaces_raw_model_score() -> None:
    result = await evaluate_context_rot(
        _GroundingOverrideBackend(),
        corpus=_single_scenario_corpus(),
        token_steps=(200,),
        depths=("middle",),
        trials=1,
        max_output_tokens=64,
        max_latency_seconds=10.0,
    )

    assert isinstance(result, Success)
    needle = next(
        measurement
        for measurement in result.unwrap().measurements
        if measurement.strategy == "raw_context" and measurement.query_type == "needle"
    )
    assert needle.raw_model_output == "Nie wiem."
    assert needle.delivered_text.endswith("5433.")
    assert needle.model_answer_correct is False
    assert needle.model_answer_abstained is True
    assert needle.assistant_answer_correct is True
    assert needle.grounding_override is True


@pytest.mark.asyncio
async def test_backend_failures_are_not_counted_as_hallucinations() -> None:
    result = await evaluate_context_rot(
        DeterministicAssistantBackend(fail_mode="error"),
        corpus=_single_scenario_corpus(),
        token_steps=(200,),
        depths=("middle",),
        trials=1,
        max_output_tokens=64,
        max_latency_seconds=10.0,
    )

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.successful_cases == 0
    assert report.failed_cases == _EXPECTED_FAILURE_CASES
    assert all(case.status == "backend_failure" for case in report.measurements)
    assert all(case.failure_code == "BACKEND_ERROR" for case in report.measurements)
    for aggregate in report.aggregates:
        assert aggregate.failure_rate == 1.0
        assert aggregate.model_needle_recall_accuracy is None
        assert aggregate.model_absent_hallucination_rate is None
        assert aggregate.assistant_absent_hallucination_rate is None


@pytest.mark.asyncio
async def test_latency_budget_is_enforced_and_reported_as_failure() -> None:
    result = await evaluate_context_rot(
        _SlowBackend(),
        corpus=_single_scenario_corpus(),
        token_steps=(200,),
        depths=("middle",),
        trials=1,
        max_output_tokens=64,
        max_latency_seconds=0.001,
    )

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.failed_cases == _EXPECTED_FAILURE_CASES
    assert {
        case.failure_code for case in report.measurements
    } == {"INFERENCE_TIMEOUT"}


@pytest.mark.asyncio
async def test_declared_context_window_rejects_an_obviously_oversized_run() -> None:
    with pytest.raises(ValueError, match="exceeds the configured server context window"):
        await evaluate_context_rot(
            DeterministicAssistantBackend(),
            corpus=_single_scenario_corpus(),
            token_steps=(4096,),
            depths=("middle",),
            trials=1,
            max_output_tokens=256,
            configured_context_window=4096,
        )
