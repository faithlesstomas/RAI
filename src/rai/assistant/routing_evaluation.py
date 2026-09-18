"""Deterministic equal-budget comparison for adaptive context routing."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    _utc_now,
)

from .context import AssistantContextBuilder, SUFFICIENCY_POLICY_VERSION
from .judging import JUDGE_VERSION, judge_answer
from .ports import AssistantModelBackend, MemoryGraphStore
from .records import AssistantContextPackage, ConversationTurn, InferenceRequest

HistoryHorizon = Literal["short", "medium", "long"]
RoutingStrategy = Literal["adaptive", "always_memory"]
_ROUTING_EVALUATION_PRODUCER = ProducerIdentity(
    producer_id="assistant-routing-evaluator",
    kind="evaluation",
    version="1.0.0",
)


class ContextRoutingEvaluationCase(BaseModel):
    """One frozen question and expected source set for a history horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    horizon: HistoryHorizon
    query_turn: ConversationTurn
    relevant_source_ids: tuple[str, ...] = Field(min_length=1)
    expected_answer_phrases: tuple[str, ...] = ()
    forbidden_answer_phrases: tuple[str, ...] = ()
    expected_abstention: bool = False


class ContextRoutingMeasurement(BaseModel):
    """Selection quality and cost for one strategy/case pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    trial_index: int = Field(default=1, ge=1)
    horizon: HistoryHorizon
    strategy: RoutingStrategy
    routing_decision: str
    selected_source_ids: tuple[str, ...]
    relevant_source_ids: tuple[str, ...]
    source_recall: float = Field(ge=0.0, le=1.0)
    correct: bool
    context_characters: int = Field(ge=0)
    context_character_budget: int = Field(ge=1)
    sufficiency_score: float = Field(ge=0.0, le=1.0)
    fallback_used: bool
    answer_correct: bool | None = None
    answer_abstained: bool | None = None
    answer_text: str | None = None
    raw_model_text: str | None = None
    grounding_override: bool = False
    raw_answer_correct: bool | None = None
    raw_answer_abstained: bool | None = None
    answer_evaluation_failed: bool = False
    answer_failure_code: str | None = None
    answer_latency_ms: float | None = Field(default=None, ge=0.0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    backend_name: str | None = None
    model_name: str | None = None
    judge_version: str | None = None


class ContextRoutingEvaluationRun(BaseModel):
    """Reproducible short/medium/long comparison under a fixed budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime = Field(default_factory=_utc_now)
    policy_version: str = SUFFICIENCY_POLICY_VERSION
    context_character_budget: int = Field(ge=1)
    adaptive_threshold: float = Field(ge=0.0, le=1.0)
    trials: int = Field(default=1, ge=1)
    measurements: tuple[ContextRoutingMeasurement, ...]
    aggregates: tuple[ContextRoutingAggregate, ...] = ()


class ContextRoutingAggregate(BaseModel):
    """Horizon/strategy summary with headline and successful-answer accuracy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: HistoryHorizon
    strategy: RoutingStrategy
    case_count: int = Field(ge=1)
    source_accuracy: float = Field(ge=0.0, le=1.0)
    answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    evaluated_answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_evaluation_failures: int = Field(ge=0)
    mean_context_characters: float = Field(ge=0.0)


def aggregate_context_routing(
    measurements: tuple[ContextRoutingMeasurement, ...],
) -> tuple[ContextRoutingAggregate, ...]:
    """Aggregate each fixed history horizon and routing strategy."""
    aggregates: list[ContextRoutingAggregate] = []
    for horizon in ("short", "medium", "long"):
        for strategy in ("adaptive", "always_memory"):
            selected = tuple(
                item
                for item in measurements
                if item.horizon == horizon and item.strategy == strategy
            )
            if not selected:
                continue
            answered = tuple(
                item for item in selected if item.answer_correct is not None
            )
            answer_evaluation_attempted = any(
                item.answer_correct is not None or item.answer_evaluation_failed
                for item in selected
            )
            aggregates.append(
                ContextRoutingAggregate(
                    horizon=horizon,
                    strategy=strategy,
                    case_count=len(selected),
                    source_accuracy=(
                        sum(item.correct for item in selected) / len(selected)
                    ),
                    answer_accuracy=(
                        sum(item.answer_correct is True for item in answered)
                        / len(selected)
                        if answer_evaluation_attempted
                        else None
                    ),
                    evaluated_answer_accuracy=(
                        sum(item.answer_correct is True for item in answered)
                        / len(answered)
                        if answered
                        else None
                    ),
                    raw_answer_accuracy=(
                        sum(item.raw_answer_correct is True for item in answered)
                        / len(selected)
                        if answer_evaluation_attempted
                        else None
                    ),
                    answer_evaluation_failures=sum(
                        item.answer_evaluation_failed for item in selected
                    ),
                    mean_context_characters=(
                        sum(item.context_characters for item in selected)
                        / len(selected)
                    ),
                )
            )
    return tuple(aggregates)


def _measurement(
    case: ContextRoutingEvaluationCase,
    strategy: RoutingStrategy,
    context_character_budget: int,
    package: AssistantContextPackage,
) -> ContextRoutingMeasurement:
    manifest = package.manifest
    selected = tuple(
        dict.fromkeys(
            (
                *manifest.recent_turn_ids,
                *manifest.durable_memory_ids,
                *manifest.episodic_turn_ids,
                *manifest.external_evidence_ids,
            )
        )
    )
    relevant = set(case.relevant_source_ids)
    recalled = relevant & set(selected)
    recall = len(recalled) / len(relevant)
    return ContextRoutingMeasurement(
        case_id=case.case_id,
        horizon=case.horizon,
        strategy=strategy,
        routing_decision=manifest.routing_decision,
        selected_source_ids=selected,
        relevant_source_ids=case.relevant_source_ids,
        source_recall=recall,
        correct=recall == 1.0,
        context_characters=manifest.actual_characters,
        context_character_budget=context_character_budget,
        sufficiency_score=manifest.sufficiency_score,
        fallback_used=manifest.fallback_used,
    )


async def _evaluate_answer(  # noqa: PLR0913
    measurement: ContextRoutingMeasurement,
    case: ContextRoutingEvaluationCase,
    package: AssistantContextPackage,
    backend: AssistantModelBackend | None,
    max_input_tokens: int,
    max_output_tokens: int,
    max_latency_seconds: float,
) -> ContextRoutingMeasurement:
    if backend is None:
        return measurement
    now = _utc_now()
    budget = InferenceBudget(
        producer=_ROUTING_EVALUATION_PRODUCER,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_agent_turns=1,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=max_latency_seconds,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=max_latency_seconds),
    )
    request = InferenceRequest(
        producer=_ROUTING_EVALUATION_PRODUCER,
        session_id=case.query_turn.session_id,
        turn_id=case.query_turn.record_id,
        request_id=f"routing-answer-{case.case_id}-{measurement.strategy}",
        context=package,
        budget=budget,
        model_name=str(getattr(backend, "model_name", "unknown")),
        system_instruction=(
            "Answer using only supplied evidence. If evidence is insufficient, "
            "explicitly say that you do not know."
        ),
    )
    started = time.perf_counter()
    try:
        generated = await asyncio.wait_for(
            backend.generate(request, CancellationToken()),
            timeout=max_latency_seconds,
        )
    except TimeoutError:
        return measurement.model_copy(
            update={
                "answer_evaluation_failed": True,
                "answer_failure_code": "ANSWER_EVALUATION_TIMEOUT",
                "answer_latency_ms": (time.perf_counter() - started) * 1_000,
            }
        )
    latency_ms = (time.perf_counter() - started) * 1_000
    if isinstance(generated, Failure):
        return measurement.model_copy(
            update={
                "answer_evaluation_failed": True,
                "answer_failure_code": generated.failure().code,
                "answer_latency_ms": latency_ms,
            }
        )
    candidate = generated.unwrap()
    correct, abstained = judge_answer(
        candidate.text,
        expected_phrases=case.expected_answer_phrases,
        forbidden_phrases=case.forbidden_answer_phrases,
        expected_abstention=case.expected_abstention,
    )
    raw_model_text = candidate.metadata.get("raw_model_output")
    if not isinstance(raw_model_text, str) or not raw_model_text.strip():
        raw_model_text = candidate.text
    raw_correct, raw_abstained = judge_answer(
        raw_model_text,
        expected_phrases=case.expected_answer_phrases,
        forbidden_phrases=case.forbidden_answer_phrases,
        expected_abstention=case.expected_abstention,
    )
    return measurement.model_copy(
        update={
            "answer_correct": correct,
            "answer_abstained": abstained,
            "answer_text": candidate.text,
            "raw_model_text": raw_model_text,
            "grounding_override": candidate.metadata.get("grounding_override") is True,
            "raw_answer_correct": raw_correct,
            "raw_answer_abstained": raw_abstained,
            "answer_latency_ms": latency_ms,
            "tokens_in": candidate.tokens_in,
            "tokens_out": candidate.tokens_out,
            "backend_name": str(getattr(backend, "backend_name", "unknown")),
            "model_name": str(getattr(backend, "model_name", "unknown")),
            "judge_version": JUDGE_VERSION,
        }
    )


async def evaluate_context_routing(  # noqa: PLR0913
    store: MemoryGraphStore,
    cases: tuple[ContextRoutingEvaluationCase, ...],
    *,
    profile_scope: str = "default",
    adaptive_threshold: float = 0.75,
    context_character_budget: int = 8_000,
    max_recent_turns: int = 10,
    max_memories: int = 5,
    max_episodic_turns: int = 5,
    answer_backend: AssistantModelBackend | None = None,
    max_input_tokens: int = 4096,
    max_output_tokens: int = 256,
    max_latency_seconds: float = 60.0,
) -> Result[ContextRoutingEvaluationRun, ActionFailure]:
    """Compare adaptive fallback with an always-compact-memory baseline."""
    adaptive = AssistantContextBuilder(
        store,
        profile_scope=profile_scope,
        memory_sufficiency_threshold=adaptive_threshold,
        max_context_characters=context_character_budget,
        max_recent_turns=max_recent_turns,
        max_memories=max_memories,
        max_episodic_turns=max_episodic_turns,
    )
    always_memory = AssistantContextBuilder(
        store,
        profile_scope=profile_scope,
        memory_sufficiency_threshold=0.0,
        max_context_characters=context_character_budget,
        max_recent_turns=max_recent_turns,
        max_memories=max_memories,
        max_episodic_turns=max_episodic_turns,
        enable_recent_route=False,
    )
    measurements: list[ContextRoutingMeasurement] = []
    for case in cases:
        accepted = await store.accept_turn(case.query_turn)
        if isinstance(accepted, Failure):
            return Failure(accepted.failure())
        for strategy, builder in (
            ("adaptive", adaptive),
            ("always_memory", always_memory),
        ):
            built = await builder.build_context(case.query_turn)
            if isinstance(built, Failure):
                return Failure(built.failure())
            package = built.unwrap()
            measurement = _measurement(
                case,
                strategy,
                context_character_budget,
                package,
            )
            measurements.append(
                await _evaluate_answer(
                    measurement,
                    case,
                    package,
                    answer_backend,
                    max_input_tokens,
                    max_output_tokens,
                    max_latency_seconds,
                )
            )
    finalized = tuple(measurements)
    return Success(
        ContextRoutingEvaluationRun(
            context_character_budget=context_character_budget,
            adaptive_threshold=adaptive_threshold,
            measurements=finalized,
            aggregates=aggregate_context_routing(finalized),
        )
    )
