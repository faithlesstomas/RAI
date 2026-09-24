"""Versioned long-context and context-rot evaluation for assistant backends.

The benchmark keeps two questions separate:

* ``raw_context`` measures the model's raw use of a long conversation;
* ``graph_memory`` is a constant compact-context control for the RAI memory path.

Protocol v2 never substitutes a grounded assistant response for raw model output,
and backend failures are reported separately from behavioral error rates.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from .judging import JUDGE_VERSION, judge_answer
from .ports import AssistantModelBackend
from .records import (
    AssistantContextManifest,
    AssistantContextPackage,
    InferenceRequest,
)

NeedleDepth = Literal["start", "middle", "end"]
EvaluationStrategy = Literal["raw_context", "graph_memory"]
InferenceStatus = Literal["success", "backend_failure"]
ModelOutputSource = Literal["model_metadata", "delivered_unmodified", "unavailable"]

_ESTIMATOR_VERSION = "rai-heuristic-words-chars-v1"
_EFFECTIVE_CONTEXT_RETENTION = 0.85
_MAX_ACCEPTABLE_FAILURE_RATE = 0.20
_MIN_SCORED_CASES_PER_CELL = 3
_CONTEXT_WINDOW_HEADROOM = 512
_ROT_EVALUATION_PRODUCER = ProducerIdentity(
    producer_id="assistant-context-rot-evaluator",
    kind="evaluation",
    version="2.0.0",
)


class ContextRotScenario(BaseModel):
    """A planted fact scenario with one present and one absent query."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    description: str
    task_family: str = "literal_retrieval"
    language: str = "pl"
    needle_user: str
    needle_assistant: str
    query_text: str
    expected_phrases: tuple[str, ...]
    forbidden_phrases: tuple[str, ...] = ()
    claim: dict[str, Any]
    absent_query_text: str | None = None
    absent_expected_phrases: tuple[str, ...] = ()
    absent_forbidden_phrases: tuple[str, ...] = ()


class DistractorItem(BaseModel):
    """A realistic desktop/system dialog exchange used as irrelevant context."""

    model_config = ConfigDict(frozen=True)

    distractor_id: str
    user_text: str
    assistant_text: str


class ContextRotCorpus(BaseModel):
    """Complete versioned corpus for context-rot testing."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    description: str
    scenarios: tuple[ContextRotScenario, ...]
    distractors: tuple[DistractorItem, ...]


class ContextRotInferenceOutcome(BaseModel):
    """One backend call before it is attached to an evaluation cell."""

    model_config = ConfigDict(frozen=True)

    status: InferenceStatus
    failure_code: str | None = None
    failure_message: str | None = None
    delivered_text: str = ""
    raw_model_output: str = ""
    model_output_source: ModelOutputSource = "unavailable"
    model_answer_correct: bool | None = None
    model_answer_abstained: bool | None = None
    assistant_answer_correct: bool | None = None
    assistant_answer_abstained: bool | None = None
    grounding_override: bool = False
    end_to_end_latency_ms: float = Field(ge=0.0)
    backend_reported_prompt_tokens: int | None = Field(default=None, ge=0)
    backend_reported_output_tokens: int | None = Field(default=None, ge=0)
    estimated_prompt_tokens: int = Field(ge=0)


class ContextRotCaseMeasurement(BaseModel):
    """Result of one inference attempt under one explicit evaluation condition."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    task_family: str
    language: str
    strategy: EvaluationStrategy
    target_context_tokens: int = Field(ge=0)
    depth: NeedleDepth | None
    estimated_context_tokens: int = Field(ge=0)
    estimated_needle_depth: float | None = Field(default=None, ge=0.0, le=1.0)
    trial_index: int = Field(ge=1)
    query_type: Literal["needle", "absent"]
    query_text: str
    status: InferenceStatus
    failure_code: str | None = None
    failure_message: str | None = None
    delivered_text: str
    raw_model_output: str
    model_output_source: ModelOutputSource
    model_answer_correct: bool | None = None
    model_answer_abstained: bool | None = None
    assistant_answer_correct: bool | None = None
    assistant_answer_abstained: bool | None = None
    grounding_override: bool
    end_to_end_latency_ms: float = Field(ge=0.0)
    backend_reported_prompt_tokens: int | None = Field(default=None, ge=0)
    backend_reported_output_tokens: int | None = Field(default=None, ge=0)
    estimated_prompt_tokens: int = Field(ge=0)
    context_turns_count: int = Field(ge=0)


class ContextRotAggregate(BaseModel):
    """Metrics for one strategy, length and optional needle position."""

    model_config = ConfigDict(frozen=True)

    strategy: EvaluationStrategy
    target_context_tokens: int = Field(ge=0)
    depth: NeedleDepth | None = None
    attempted_case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    model_scored_case_count: int = Field(ge=0)
    assistant_scored_case_count: int = Field(ge=0)
    model_needle_case_count: int = Field(ge=0)
    model_needle_recall_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    model_needle_recall_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    model_needle_recall_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    model_absent_case_count: int = Field(ge=0)
    model_absent_hallucination_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    model_absent_hallucination_ci_low: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    model_absent_hallucination_ci_high: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    assistant_needle_recall_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    assistant_absent_hallucination_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    mean_end_to_end_latency_ms: float = Field(ge=0.0)
    mean_backend_reported_prompt_tokens: float | None = Field(default=None, ge=0.0)
    mean_backend_reported_output_tokens: float | None = Field(default=None, ge=0.0)
    mean_estimated_prompt_tokens: float = Field(ge=0.0)


class ContextRotEffectiveContext(BaseModel):
    """Task-local effective-context estimate relative to the shortest baseline."""

    model_config = ConfigDict(frozen=True)

    depth: NeedleDepth
    retention_threshold: float = Field(ge=0.0, le=1.0)
    baseline_target_tokens: int = Field(ge=0)
    baseline_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    first_below_threshold_tokens: int | None = Field(default=None, ge=0)
    longest_tested_above_threshold_tokens: int | None = Field(default=None, ge=0)
    valid: bool
    invalid_reason: str | None = None


class ContextRotEvaluationReport(BaseModel):
    """Complete protocol-v2 context-rot artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_version: str = "rai-assistant-context-rot-v2"
    generated_at: str
    corpus_version: str
    judge_version: str = JUDGE_VERSION
    backend_name: str
    model_name: str
    model_artifact_version: str | None = None
    prompt_template_version: str | None = None
    configured_context_window: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    max_output_tokens: int = Field(ge=1)
    max_latency_seconds: float = Field(gt=0.0)
    token_estimator: str = _ESTIMATOR_VERSION
    token_steps: tuple[int, ...]
    depths: tuple[NeedleDepth, ...]
    trials: int = Field(ge=1)
    total_cases: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    measurements: tuple[ContextRotCaseMeasurement, ...]
    aggregates: tuple[ContextRotAggregate, ...]
    effective_context: tuple[ContextRotEffectiveContext, ...]


def load_context_rot_corpus(path: Path) -> ContextRotCorpus:
    """Load and validate a versioned context-rot corpus."""
    with open(path, encoding="utf-8") as file_handle:
        return ContextRotCorpus.model_validate(json.load(file_handle))


def estimate_tokens(text: str) -> int:
    """Estimate tokens when a backend tokenizer is unavailable."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int(words * 1.35 + chars / 15.0))


def generate_haystack(
    distractors: Sequence[DistractorItem],
    needle_user: str,
    needle_assistant: str,
    target_tokens: int,
    depth: NeedleDepth,
) -> list[dict[str, str]]:
    """Synthesize a deterministic conversation with one planted exchange."""
    needle_pair = [
        {"role": "user", "text": needle_user},
        {"role": "assistant", "text": needle_assistant},
    ]
    needle_tokens = estimate_tokens(needle_user) + estimate_tokens(needle_assistant)
    remaining_tokens = max(0, target_tokens - needle_tokens)
    distractor_pairs = tuple(
        (
            [
                {"role": "user", "text": item.user_text},
                {"role": "assistant", "text": item.assistant_text},
            ],
            estimate_tokens(item.user_text) + estimate_tokens(item.assistant_text),
        )
        for item in distractors
    )
    if remaining_tokens > 0 and not distractor_pairs:
        raise ValueError("distractors are required to reach the target token count")
    if remaining_tokens > 0 and not any(
        tokens > 0 for _pair, tokens in distractor_pairs
    ):
        raise ValueError("distractors must contain non-empty text")

    pairs: list[list[dict[str, str]]] = []
    accumulated_tokens = 0
    distractor_idx = 0
    while accumulated_tokens < remaining_tokens:
        pair, pair_tokens = distractor_pairs[distractor_idx % len(distractor_pairs)]
        pairs.append(pair)
        accumulated_tokens += pair_tokens
        distractor_idx += 1

    if not pairs:
        insert_pos = 0
    elif depth == "start":
        insert_pos = max(0, int(len(pairs) * 0.10))
    elif depth == "middle":
        insert_pos = int(len(pairs) * 0.50)
    else:
        insert_pos = min(len(pairs), max(0, int(len(pairs) * 0.90)))
    pairs.insert(insert_pos, needle_pair)
    return [turn for pair in pairs for turn in pair]


def estimate_haystack_layout(
    turns: Sequence[dict[str, str]], needle_user: str, needle_assistant: str
) -> tuple[int, float | None]:
    """Return estimated context length and the needle midpoint as a ratio."""
    turn_tokens = [estimate_tokens(str(turn.get("text", ""))) for turn in turns]
    total = sum(turn_tokens)
    for index, turn in enumerate(turns):
        if turn.get("role") != "user" or turn.get("text") != needle_user:
            continue
        needle_tokens = turn_tokens[index]
        if index + 1 < len(turns) and turns[index + 1].get("text") == needle_assistant:
            needle_tokens += turn_tokens[index + 1]
        midpoint = sum(turn_tokens[:index]) + needle_tokens / 2.0
        return total, midpoint / total if total else None
    return total, None


def _estimate_prompt_tokens(
    query_text: str,
    recent_turns: Sequence[dict[str, str]],
    durable_memories: Sequence[dict[str, Any]],
) -> int:
    context_tokens = sum(
        estimate_tokens(str(turn.get("text", ""))) for turn in recent_turns
    )
    memory_tokens = estimate_tokens(json.dumps(durable_memories, ensure_ascii=False))
    return context_tokens + memory_tokens + estimate_tokens(query_text) + 64


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return max(0.0, (centre - spread) / denominator), min(
        1.0, (centre + spread) / denominator
    )


async def _run_single_inference(  # noqa: PLR0913
    backend: AssistantModelBackend,
    session_id: str,
    turn_id: str,
    query_text: str,
    recent_turns: list[dict[str, str]],
    durable_memories: list[dict[str, Any]],
    expected_phrases: tuple[str, ...],
    forbidden_phrases: tuple[str, ...],
    expected_abstention: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    max_latency_seconds: float,
) -> ContextRotInferenceOutcome:
    """Execute one inference and independently judge raw and delivered output."""
    estimated_prompt_tokens = _estimate_prompt_tokens(
        query_text, recent_turns, durable_memories
    )
    now = _utc_now()
    budget = InferenceBudget(
        producer=_ROT_EVALUATION_PRODUCER,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_agent_turns=1,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=max_latency_seconds,
        max_provider_cost=0,
        max_ram_bytes=4 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=max_latency_seconds),
    )

    total_chars = (
        sum(len(str(turn.get("text", ""))) for turn in recent_turns)
        + len(json.dumps(durable_memories, ensure_ascii=False))
        + len(query_text)
    )
    manifest = AssistantContextManifest(
        producer=_ROT_EVALUATION_PRODUCER,
        session_id=session_id,
        turn_id=turn_id,
        routing_decision="recent_conversation" if recent_turns else "compact_memory",
        evidence_required=True,
        actual_characters=total_chars,
        policy_version="1.0.0",
        backend_name=str(getattr(backend, "backend_name", "unknown")),
        model_name=str(getattr(backend, "model_name", "unknown")),
        model_artifact_version=getattr(backend, "model_artifact_version", None),
        prompt_template_version=str(
            getattr(backend, "prompt_template_version", "unknown")
        ),
    )
    package = AssistantContextPackage(
        producer=_ROT_EVALUATION_PRODUCER,
        session_id=session_id,
        turn_id=turn_id,
        manifest=manifest,
        content={
            "recent_turns": recent_turns,
            "durable_memories": durable_memories,
            "current_turn": {"role": "user", "text": query_text},
        },
    )
    request = InferenceRequest(
        producer=_ROT_EVALUATION_PRODUCER,
        session_id=session_id,
        turn_id=turn_id,
        request_id=f"rot-inf-{_new_id()}",
        context=package,
        budget=budget,
        model_name=str(getattr(backend, "model_name", "unknown")),
        system_instruction=(
            "You are RAI desktop assistant. Answer the question using only facts "
            "and preferences given in the context. If the fact is not mentioned in "
            "the context, state clearly that you do not know."
        ),
    )

    start_time = time.monotonic()
    try:
        result = await asyncio.wait_for(
            backend.generate(request, CancellationToken()),
            timeout=max_latency_seconds,
        )
    except asyncio.TimeoutError:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        return ContextRotInferenceOutcome(
            status="backend_failure",
            failure_code="INFERENCE_TIMEOUT",
            failure_message=(
                f"backend exceeded the {max_latency_seconds:.3f}s latency budget"
            ),
            end_to_end_latency_ms=latency_ms,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start_time) * 1000.0
        return ContextRotInferenceOutcome(
            status="backend_failure",
            failure_code="BACKEND_EXCEPTION",
            failure_message=str(exc),
            end_to_end_latency_ms=latency_ms,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
    latency_ms = (time.monotonic() - start_time) * 1000.0
    if isinstance(result, Failure):
        failure = result.failure()
        return ContextRotInferenceOutcome(
            status="backend_failure",
            failure_code=str(getattr(failure, "code", "BACKEND_FAILURE")),
            failure_message=str(getattr(failure, "message", failure)),
            end_to_end_latency_ms=latency_ms,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )

    candidate = result.unwrap()
    delivered = candidate.text.strip()
    grounding_override = candidate.metadata.get("grounding_override") is True
    metadata_output = candidate.metadata.get("raw_model_output")
    if isinstance(metadata_output, str) and metadata_output.strip():
        raw_output = metadata_output.strip()
        output_source: ModelOutputSource = "model_metadata"
    elif not grounding_override:
        raw_output = delivered
        output_source = "delivered_unmodified"
    else:
        raw_output = ""
        output_source = "unavailable"

    model_correct: bool | None = None
    model_abstained: bool | None = None
    if raw_output:
        model_correct, model_abstained = judge_answer(
            raw_output,
            expected_phrases=expected_phrases,
            forbidden_phrases=forbidden_phrases,
            expected_abstention=expected_abstention,
        )
    assistant_correct, assistant_abstained = judge_answer(
        delivered,
        expected_phrases=expected_phrases,
        forbidden_phrases=forbidden_phrases,
        expected_abstention=expected_abstention,
    )
    return ContextRotInferenceOutcome(
        status="success",
        delivered_text=delivered,
        raw_model_output=raw_output,
        model_output_source=output_source,
        model_answer_correct=model_correct,
        model_answer_abstained=model_abstained,
        assistant_answer_correct=assistant_correct,
        assistant_answer_abstained=assistant_abstained,
        grounding_override=grounding_override,
        end_to_end_latency_ms=latency_ms,
        backend_reported_prompt_tokens=candidate.tokens_in,
        backend_reported_output_tokens=candidate.tokens_out,
        estimated_prompt_tokens=estimated_prompt_tokens,
    )


def _measurement(  # noqa: PLR0913
    *,
    scenario: ContextRotScenario,
    strategy: EvaluationStrategy,
    target_context_tokens: int,
    depth: NeedleDepth | None,
    estimated_context_tokens: int,
    estimated_needle_depth: float | None,
    trial_index: int,
    query_type: Literal["needle", "absent"],
    query_text: str,
    context_turns_count: int,
    outcome: ContextRotInferenceOutcome,
) -> ContextRotCaseMeasurement:
    return ContextRotCaseMeasurement(
        scenario_id=scenario.scenario_id,
        task_family=scenario.task_family,
        language=scenario.language,
        strategy=strategy,
        target_context_tokens=target_context_tokens,
        depth=depth,
        estimated_context_tokens=estimated_context_tokens,
        estimated_needle_depth=estimated_needle_depth,
        trial_index=trial_index,
        query_type=query_type,
        query_text=query_text,
        context_turns_count=context_turns_count,
        **outcome.model_dump(),
    )


async def _evaluate_query(  # noqa: PLR0913
    backend: AssistantModelBackend,
    scenario: ContextRotScenario,
    strategy: EvaluationStrategy,
    target_context_tokens: int,
    depth: NeedleDepth | None,
    trial_index: int,
    query_type: Literal["needle", "absent"],
    query_text: str,
    recent_turns: list[dict[str, str]],
    durable_memories: list[dict[str, Any]],
    expected_phrases: tuple[str, ...],
    forbidden_phrases: tuple[str, ...],
    expected_abstention: bool,
    estimated_context_tokens: int,
    estimated_needle_depth: float | None,
    max_input_tokens: int,
    max_output_tokens: int,
    max_latency_seconds: float,
) -> ContextRotCaseMeasurement:
    session_id = (
        f"sess-rot-{strategy}-{scenario.scenario_id}-{target_context_tokens}-"
        f"{depth or 'control'}-{trial_index}"
    )
    outcome = await _run_single_inference(
        backend=backend,
        session_id=session_id,
        turn_id=f"turn-{query_type}-{_new_id()}",
        query_text=query_text,
        recent_turns=recent_turns,
        durable_memories=durable_memories,
        expected_phrases=expected_phrases,
        forbidden_phrases=forbidden_phrases,
        expected_abstention=expected_abstention,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_latency_seconds=max_latency_seconds,
    )
    return _measurement(
        scenario=scenario,
        strategy=strategy,
        target_context_tokens=target_context_tokens,
        depth=depth,
        estimated_context_tokens=estimated_context_tokens,
        estimated_needle_depth=estimated_needle_depth,
        trial_index=trial_index,
        query_type=query_type,
        query_text=query_text,
        context_turns_count=len(recent_turns),
        outcome=outcome,
    )


def _optional_mean(values: Sequence[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _aggregate_group(
    strategy: EvaluationStrategy,
    target_context_tokens: int,
    depth: NeedleDepth | None,
    group: Sequence[ContextRotCaseMeasurement],
) -> ContextRotAggregate:
    successful = [case for case in group if case.status == "success"]
    model_scored = [
        case for case in successful if case.model_answer_correct is not None
    ]
    assistant_scored = [
        case for case in successful if case.assistant_answer_correct is not None
    ]
    model_needles = [
        bool(case.model_answer_correct)
        for case in model_scored
        if case.query_type == "needle"
    ]
    model_absent_hallucinations = [
        not bool(case.model_answer_abstained)
        for case in model_scored
        if case.query_type == "absent"
    ]
    assistant_needles = [
        bool(case.assistant_answer_correct)
        for case in assistant_scored
        if case.query_type == "needle"
    ]
    assistant_absent_hallucinations = [
        not bool(case.assistant_answer_abstained)
        for case in assistant_scored
        if case.query_type == "absent"
    ]
    recall_successes = sum(model_needles)
    hallucination_successes = sum(model_absent_hallucinations)
    recall_ci = _wilson_interval(recall_successes, len(model_needles))
    hallucination_ci = _wilson_interval(
        hallucination_successes, len(model_absent_hallucinations)
    )
    return ContextRotAggregate(
        strategy=strategy,
        target_context_tokens=target_context_tokens,
        depth=depth,
        attempted_case_count=len(group),
        successful_case_count=len(successful),
        failed_case_count=len(group) - len(successful),
        failure_rate=(len(group) - len(successful)) / len(group) if group else 0.0,
        model_scored_case_count=len(model_scored),
        assistant_scored_case_count=len(assistant_scored),
        model_needle_case_count=len(model_needles),
        model_needle_recall_accuracy=(
            recall_successes / len(model_needles) if model_needles else None
        ),
        model_needle_recall_ci_low=recall_ci[0],
        model_needle_recall_ci_high=recall_ci[1],
        model_absent_case_count=len(model_absent_hallucinations),
        model_absent_hallucination_rate=(
            hallucination_successes / len(model_absent_hallucinations)
            if model_absent_hallucinations
            else None
        ),
        model_absent_hallucination_ci_low=hallucination_ci[0],
        model_absent_hallucination_ci_high=hallucination_ci[1],
        assistant_needle_recall_accuracy=(
            sum(assistant_needles) / len(assistant_needles)
            if assistant_needles
            else None
        ),
        assistant_absent_hallucination_rate=(
            sum(assistant_absent_hallucinations) / len(assistant_absent_hallucinations)
            if assistant_absent_hallucinations
            else None
        ),
        mean_end_to_end_latency_ms=(
            sum(case.end_to_end_latency_ms for case in group) / len(group)
            if group
            else 0.0
        ),
        mean_backend_reported_prompt_tokens=_optional_mean(
            [case.backend_reported_prompt_tokens for case in successful]
        ),
        mean_backend_reported_output_tokens=_optional_mean(
            [case.backend_reported_output_tokens for case in successful]
        ),
        mean_estimated_prompt_tokens=(
            sum(case.estimated_prompt_tokens for case in group) / len(group)
            if group
            else 0.0
        ),
    )


def _aggregate_measurements(
    measurements: Sequence[ContextRotCaseMeasurement],
) -> list[ContextRotAggregate]:
    aggregates: list[ContextRotAggregate] = []
    for strategy in ("raw_context", "graph_memory"):
        targets = sorted(
            {
                case.target_context_tokens
                for case in measurements
                if case.strategy == strategy
            }
        )
        for target in targets:
            target_cases = [
                case
                for case in measurements
                if case.strategy == strategy and case.target_context_tokens == target
            ]
            depths = [
                depth
                for depth in ("start", "middle", "end")
                if any(case.depth == depth for case in target_cases)
            ]
            for depth in depths:
                depth_cases = [case for case in target_cases if case.depth == depth]
                aggregates.append(
                    _aggregate_group(strategy, target, depth, depth_cases)  # type: ignore[arg-type]
                )
            if not depths or len(depths) > 1:
                aggregates.append(
                    _aggregate_group(strategy, target, None, target_cases)
                )
    return aggregates


def _effective_context_summaries(
    aggregates: Sequence[ContextRotAggregate],
) -> tuple[ContextRotEffectiveContext, ...]:
    summaries: list[ContextRotEffectiveContext] = []
    for depth in ("start", "middle", "end"):
        points = sorted(
            (
                aggregate
                for aggregate in aggregates
                if aggregate.strategy == "raw_context" and aggregate.depth == depth
            ),
            key=lambda aggregate: aggregate.target_context_tokens,
        )
        if not points:
            continue
        baseline = points[0]
        invalid_reasons: list[str] = []
        if baseline.model_needle_recall_accuracy in (None, 0.0):
            invalid_reasons.append("short-context baseline is missing or zero")
        if any(point.failure_rate > _MAX_ACCEPTABLE_FAILURE_RATE for point in points):
            invalid_reasons.append("backend failure rate exceeds 20%")
        if any(
            point.model_needle_case_count < _MIN_SCORED_CASES_PER_CELL
            for point in points
        ):
            invalid_reasons.append(
                "fewer than three scored needle cases in at least one cell"
            )

        baseline_accuracy = baseline.model_needle_recall_accuracy
        threshold = (
            baseline_accuracy * _EFFECTIVE_CONTEXT_RETENTION
            if baseline_accuracy is not None
            else None
        )
        above = [
            point.target_context_tokens
            for point in points
            if threshold is not None
            and point.model_needle_recall_accuracy is not None
            and point.model_needle_recall_accuracy >= threshold
        ]
        below = [
            point.target_context_tokens
            for point in points
            if threshold is not None
            and point.model_needle_recall_accuracy is not None
            and point.model_needle_recall_accuracy < threshold
        ]
        summaries.append(
            ContextRotEffectiveContext(
                depth=depth,  # type: ignore[arg-type]
                retention_threshold=_EFFECTIVE_CONTEXT_RETENTION,
                baseline_target_tokens=baseline.target_context_tokens,
                baseline_accuracy=baseline_accuracy,
                first_below_threshold_tokens=min(below) if below else None,
                longest_tested_above_threshold_tokens=max(above) if above else None,
                valid=not invalid_reasons,
                invalid_reason="; ".join(invalid_reasons) or None,
            )
        )
    return tuple(summaries)


async def evaluate_context_rot(  # noqa: PLR0912,PLR0913
    backend: AssistantModelBackend,
    *,
    corpus: ContextRotCorpus,
    token_steps: Sequence[int] = (512, 2048, 8192, 32768),
    depths: Sequence[NeedleDepth] = ("start", "middle", "end"),
    trials: int = 3,
    max_output_tokens: int = 256,
    max_latency_seconds: float = 90.0,
    configured_context_window: int | None = None,
) -> Result[ContextRotEvaluationReport, ActionFailure]:
    """Execute protocol v2 and return raw-model plus delivered-system metrics."""
    if not token_steps or any(step <= 0 for step in token_steps):
        raise ValueError("token_steps must contain positive values")
    if trials < 1:
        raise ValueError("trials must be at least one")
    if (
        configured_context_window is not None
        and max(token_steps) + max_output_tokens + _CONTEXT_WINDOW_HEADROOM
        > configured_context_window
    ):
        raise ValueError(
            "largest target context plus output/headroom exceeds the configured "
            "server context window; lower --token-steps or increase the server limit"
        )

    backend_result = await backend.start()
    if isinstance(backend_result, Failure):
        return Failure(backend_result.failure())

    measurements: list[ContextRotCaseMeasurement] = []
    try:
        for token_step in token_steps:
            max_input_tokens = max(4096, int(token_step * 2.0) + 2048)
            for depth in depths:
                for scenario in corpus.scenarios:
                    turns = generate_haystack(
                        distractors=corpus.distractors,
                        needle_user=scenario.needle_user,
                        needle_assistant=scenario.needle_assistant,
                        target_tokens=token_step,
                        depth=depth,
                    )
                    estimated_context, estimated_depth = estimate_haystack_layout(
                        turns, scenario.needle_user, scenario.needle_assistant
                    )
                    for trial in range(1, trials + 1):
                        measurements.append(
                            await _evaluate_query(
                                backend,
                                scenario,
                                "raw_context",
                                token_step,
                                depth,
                                trial,
                                "needle",
                                scenario.query_text,
                                turns,
                                [],
                                scenario.expected_phrases,
                                scenario.forbidden_phrases,
                                False,
                                estimated_context,
                                estimated_depth,
                                max_input_tokens,
                                max_output_tokens,
                                max_latency_seconds,
                            )
                        )
                        if scenario.absent_query_text:
                            measurements.append(
                                await _evaluate_query(
                                    backend,
                                    scenario,
                                    "raw_context",
                                    token_step,
                                    depth,
                                    trial,
                                    "absent",
                                    scenario.absent_query_text,
                                    turns,
                                    [],
                                    scenario.absent_expected_phrases,
                                    scenario.absent_forbidden_phrases,
                                    True,
                                    estimated_context,
                                    estimated_depth,
                                    max_input_tokens,
                                    max_output_tokens,
                                    max_latency_seconds,
                                )
                            )

        # Compact graph memory is a constant control. It is intentionally run once
        # per scenario/trial instead of being relabelled for every length and depth.
        for scenario in corpus.scenarios:
            claim = {
                "topic": scenario.claim.get("topic", "needle.fact"),
                "content": scenario.claim.get("content", {}),
            }
            estimated_context = estimate_tokens(json.dumps([claim], ensure_ascii=False))
            for trial in range(1, trials + 1):
                measurements.append(
                    await _evaluate_query(
                        backend,
                        scenario,
                        "graph_memory",
                        0,
                        None,
                        trial,
                        "needle",
                        scenario.query_text,
                        [],
                        [claim],
                        scenario.expected_phrases,
                        scenario.forbidden_phrases,
                        False,
                        estimated_context,
                        None,
                        4096,
                        max_output_tokens,
                        max_latency_seconds,
                    )
                )
                if scenario.absent_query_text:
                    measurements.append(
                        await _evaluate_query(
                            backend,
                            scenario,
                            "graph_memory",
                            0,
                            None,
                            trial,
                            "absent",
                            scenario.absent_query_text,
                            [],
                            [claim],
                            scenario.absent_expected_phrases,
                            scenario.absent_forbidden_phrases,
                            True,
                            estimated_context,
                            None,
                            4096,
                            max_output_tokens,
                            max_latency_seconds,
                        )
                    )
    finally:
        await backend.stop()

    aggregates = tuple(_aggregate_measurements(measurements))
    successful = sum(case.status == "success" for case in measurements)
    return Success(
        ContextRotEvaluationReport(
            generated_at=_utc_now().isoformat(),
            corpus_version=corpus.corpus_version,
            backend_name=str(getattr(backend, "backend_name", "unknown")),
            model_name=str(getattr(backend, "model_name", "unknown")),
            model_artifact_version=getattr(backend, "model_artifact_version", None),
            prompt_template_version=getattr(backend, "prompt_template_version", None),
            configured_context_window=configured_context_window,
            temperature=getattr(backend, "temperature", None),
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
            token_steps=tuple(token_steps),
            depths=tuple(depths),
            trials=trials,
            total_cases=len(measurements),
            successful_cases=successful,
            failed_cases=len(measurements) - successful,
            measurements=tuple(measurements),
            aggregates=aggregates,
            effective_context=_effective_context_summaries(aggregates),
        )
    )


def save_context_rot_report(
    report: ContextRotEvaluationReport, output_path: Path
) -> None:
    """Serialize and save the context-rot report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report.model_dump(), file_handle, indent=2, ensure_ascii=False)


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.1f}%"


def format_context_rot_summary_table(report: ContextRotEvaluationReport) -> str:
    """Format model and assistant metrics without mislabelling full latency as TTFT."""
    header = (
        f"{'Strategy':<14} | {'Target':<8} | {'Depth':<8} | {'OK/All':<7} | "
        f"{'Model Recall':<12} | {'Model Halluc':<12} | {'RAI Recall':<10} | "
        f"{'RAI Halluc':<10} | {'E2E ms':>9} | {'Prompt tok':>10}"
    )
    divider = "-" * len(header)
    lines = [
        divider,
        f" CONTEXT ROT V2 — {report.model_name} ({report.backend_name})",
        divider,
        header,
        divider,
    ]
    for aggregate in report.aggregates:
        target = (
            "control"
            if aggregate.strategy == "graph_memory"
            else str(aggregate.target_context_tokens)
        )
        depth = aggregate.depth or (
            "control" if aggregate.strategy == "graph_memory" else "OVERALL"
        )
        prompt_tokens = aggregate.mean_backend_reported_prompt_tokens
        prompt_text = "n/a" if prompt_tokens is None else f"{prompt_tokens:.0f}"
        lines.append(
            f"{aggregate.strategy:<14} | {target:<8} | {depth:<8} | "
            f"{aggregate.successful_case_count:>2}/{aggregate.attempted_case_count:<4} | "
            f"{_format_rate(aggregate.model_needle_recall_accuracy):<12} | "
            f"{_format_rate(aggregate.model_absent_hallucination_rate):<12} | "
            f"{_format_rate(aggregate.assistant_needle_recall_accuracy):<10} | "
            f"{_format_rate(aggregate.assistant_absent_hallucination_rate):<10} | "
            f"{aggregate.mean_end_to_end_latency_ms:>9.1f} | {prompt_text:>10}"
        )
    lines.append(divider)
    return "\n".join(lines)
