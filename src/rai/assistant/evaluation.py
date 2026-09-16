"""Equal-budget retrieval evaluation for evidence-first assistant memory."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass, _utc_now

from .ports import AssistantEvidence, AssistantEvidenceProvider, MemoryGraphStore
from .query import MemoryQueryResolver
from .records import ConversationTurn, MemoryRecord
from .summary import validate_grounded_summary

RetrievalChannel = Literal[
    "raw_turns_bm25",
    "claims_bm25",
    "summaries_grounded",
]


class RetrievalEvaluationCase(BaseModel):
    """One fixed query with channel-specific ground-truth source IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_raw_turn_ids: tuple[str, ...] = ()
    relevant_claim_ids: tuple[str, ...] = ()
    relevant_summary_source_ids: tuple[str, ...] = ()
    data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL)


class RetrievalAnswerEvaluator(Protocol):
    """Evaluate whether an answerer used one bounded retrieval context correctly."""

    async def evaluate(
        self,
        case: RetrievalEvaluationCase,
        channel: RetrievalChannel,
        context_items: tuple[dict[str, object], ...],
    ) -> Result[bool, ActionFailure]: ...


class RetrievalChannelMeasurement(BaseModel):
    """Comparable output for one case and retrieval channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    channel: RetrievalChannel
    retrieved_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...] = ()
    relevant_ids: tuple[str, ...]
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    abstained: bool
    latency_ms: float = Field(ge=0.0)
    context_characters: int = Field(ge=0)
    answer_correct: bool | None = None
    answer_utilization_evaluated: bool = False
    answer_evaluation_failed: bool = False
    answer_failure_code: str | None = None
    energy_joules: float | None = Field(default=None, ge=0.0)
    retrieval_failed: bool = False
    failure_code: str | None = None


class RetrievalEvaluationRun(BaseModel):
    """Reproducible run manifest proving equal channel budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_version: str = Field(default="rai-retrieval-floor-v1", min_length=1)
    generated_at: datetime = Field(default_factory=_utc_now)
    profile_scope: str
    retrieval_limit: int = Field(ge=1)
    context_character_budget: int = Field(ge=1)
    measurements: tuple[RetrievalChannelMeasurement, ...]


def _metrics(  # noqa: PLR0913
    *,
    case_id: str,
    channel: RetrievalChannel,
    retrieved_ids: tuple[str, ...],
    artifact_ids: tuple[str, ...],
    relevant_ids: tuple[str, ...],
    latency_ms: float,
    context_characters: int,
    failure_code: str | None = None,
) -> RetrievalChannelMeasurement:
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)
    true_positive = len(retrieved & relevant)
    recall = true_positive / len(relevant) if relevant else 1.0
    precision = true_positive / len(retrieved) if retrieved else 0.0
    return RetrievalChannelMeasurement(
        case_id=case_id,
        channel=channel,
        retrieved_ids=retrieved_ids,
        artifact_ids=artifact_ids,
        relevant_ids=relevant_ids,
        recall=recall,
        precision=precision,
        abstained=not retrieved_ids,
        latency_ms=latency_ms,
        context_characters=context_characters,
        retrieval_failed=failure_code is not None,
        failure_code=failure_code,
    )


def _bounded_raw_items(
    items: tuple[tuple[ConversationTurn, str], ...], character_budget: int
) -> tuple[tuple[str, ...], int, tuple[dict[str, object], ...]]:
    selected: list[str] = []
    context_items: list[dict[str, object]] = []
    used = 0
    for turn, _reason in items:
        item_size = len(turn.text)
        if used + item_size > character_budget:
            continue
        selected.append(turn.record_id)
        context_items.append(
            {
                "source_id": turn.record_id,
                "source_type": "conversation_turn",
                "content": {"role": turn.role, "text": turn.text},
            }
        )
        used += item_size
    return tuple(selected), used, tuple(context_items)


def _bounded_claim_items(
    items: tuple[tuple[MemoryRecord, str], ...], character_budget: int
) -> tuple[tuple[str, ...], int, tuple[dict[str, object], ...]]:
    selected: list[str] = []
    context_items: list[dict[str, object]] = []
    used = 0
    for memory, _reason in items:
        item_size = len(json.dumps(memory.content, ensure_ascii=False))
        if used + item_size > character_budget:
            continue
        selected.append(memory.record_id)
        context_items.append(
            {
                "source_id": memory.record_id,
                "source_type": "memory_record",
                "content": {"topic": memory.topic, "content": memory.content},
            }
        )
        used += item_size
    return tuple(selected), used, tuple(context_items)


def _bounded_summary_items(
    items: tuple[AssistantEvidence, ...], character_budget: int
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    int,
    tuple[dict[str, object], ...],
    str | None,
]:
    covered_source_ids: list[str] = []
    artifact_ids: list[str] = []
    context_items: list[dict[str, object]] = []
    used = 0
    for evidence in items:
        invalid = validate_grounded_summary(evidence)
        if invalid is not None:
            return (), (), 0, (), invalid.code
        item_size = len(json.dumps(evidence.content, ensure_ascii=False))
        if used + item_size > character_budget:
            continue
        artifact_ids.append(evidence.source_id)
        covered_source_ids.extend(
            str(source_id) for source_id in evidence.content["source_memory_ids"]
        )
        context_items.append(
            {
                "source_id": evidence.source_id,
                "source_type": evidence.source_type,
                "content": evidence.content,
            }
        )
        used += item_size
    return (
        tuple(dict.fromkeys(covered_source_ids)),
        tuple(artifact_ids),
        used,
        tuple(context_items),
        None,
    )


async def _evaluate_answer_use(
    measurement: RetrievalChannelMeasurement,
    evaluator: RetrievalAnswerEvaluator | None,
    case: RetrievalEvaluationCase,
    context_items: tuple[dict[str, object], ...],
) -> RetrievalChannelMeasurement:
    if evaluator is None or measurement.retrieval_failed:
        return measurement
    try:
        result = await evaluator.evaluate(case, measurement.channel, context_items)
    except Exception:  # noqa: BLE001
        return measurement.model_copy(
            update={
                "answer_evaluation_failed": True,
                "answer_failure_code": "ANSWER_EVALUATOR_EXCEPTION",
            }
        )
    if isinstance(result, Failure):
        return measurement.model_copy(
            update={
                "answer_evaluation_failed": True,
                "answer_failure_code": result.failure().code,
            }
        )
    return measurement.model_copy(
        update={
            "answer_correct": result.unwrap(),
            "answer_utilization_evaluated": True,
        }
    )


async def evaluate_retrieval_floor(  # noqa: PLR0913
    store: MemoryGraphStore,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    profile_scope: str = "default",
    retrieval_limit: int = 5,
    context_character_budget: int = 8_000,
    summary_provider: AssistantEvidenceProvider | None = None,
    answer_evaluator: RetrievalAnswerEvaluator | None = None,
) -> Result[RetrievalEvaluationRun, ActionFailure]:
    """Evaluate raw, claim and optional grounded-summary channels equally."""
    measurements: list[RetrievalChannelMeasurement] = []
    for case in cases:
        query = MemoryQueryResolver.resolve(case.query_text, profile_scope)

        started = time.perf_counter()
        raw_result = await store.retrieve_relevant_turns(
            profile_scope=profile_scope,
            query=query,
            data_classes=case.data_classes,
            limit=retrieval_limit,
        )
        raw_latency = (time.perf_counter() - started) * 1_000
        if isinstance(raw_result, Failure):
            failure = raw_result.failure()
            measurement = _metrics(
                case_id=case.case_id,
                channel="raw_turns_bm25",
                retrieved_ids=(),
                artifact_ids=(),
                relevant_ids=case.relevant_raw_turn_ids,
                latency_ms=raw_latency,
                context_characters=0,
                failure_code=failure.code,
            )
            measurements.append(measurement)
        else:
            raw_items = raw_result.unwrap()
            raw_ids, raw_characters, raw_context = _bounded_raw_items(
                raw_items, context_character_budget
            )
            measurement = _metrics(
                case_id=case.case_id,
                channel="raw_turns_bm25",
                retrieved_ids=raw_ids,
                artifact_ids=raw_ids,
                relevant_ids=case.relevant_raw_turn_ids,
                latency_ms=raw_latency,
                context_characters=raw_characters,
            )
            measurements.append(
                await _evaluate_answer_use(
                    measurement, answer_evaluator, case, raw_context
                )
            )

        started = time.perf_counter()
        claim_result = await store.retrieve_relevant_memories(
            profile_scope=profile_scope,
            query=query,
            data_classes=case.data_classes,
            limit=retrieval_limit,
        )
        claim_latency = (time.perf_counter() - started) * 1_000
        if isinstance(claim_result, Failure):
            failure = claim_result.failure()
            measurement = _metrics(
                case_id=case.case_id,
                channel="claims_bm25",
                retrieved_ids=(),
                artifact_ids=(),
                relevant_ids=case.relevant_claim_ids,
                latency_ms=claim_latency,
                context_characters=0,
                failure_code=failure.code,
            )
            measurements.append(measurement)
        else:
            claim_items = claim_result.unwrap()
            claim_ids, claim_characters, claim_context = _bounded_claim_items(
                claim_items, context_character_budget
            )
            measurement = _metrics(
                case_id=case.case_id,
                channel="claims_bm25",
                retrieved_ids=claim_ids,
                artifact_ids=claim_ids,
                relevant_ids=case.relevant_claim_ids,
                latency_ms=claim_latency,
                context_characters=claim_characters,
            )
            measurements.append(
                await _evaluate_answer_use(
                    measurement, answer_evaluator, case, claim_context
                )
            )

        if summary_provider is not None:
            started = time.perf_counter()
            summary_result = await summary_provider.retrieve(
                query=query,
                data_classes=case.data_classes,
                limit=retrieval_limit,
            )
            summary_latency = (time.perf_counter() - started) * 1_000
            if isinstance(summary_result, Failure):
                measurement = _metrics(
                    case_id=case.case_id,
                    channel="summaries_grounded",
                    retrieved_ids=(),
                    artifact_ids=(),
                    relevant_ids=case.relevant_summary_source_ids,
                    latency_ms=summary_latency,
                    context_characters=0,
                    failure_code=summary_result.failure().code,
                )
                measurements.append(measurement)
            else:
                (
                    summary_source_ids,
                    summary_artifact_ids,
                    summary_characters,
                    summary_context,
                    summary_failure,
                ) = _bounded_summary_items(
                    summary_result.unwrap(), context_character_budget
                )
                measurement = _metrics(
                    case_id=case.case_id,
                    channel="summaries_grounded",
                    retrieved_ids=summary_source_ids,
                    artifact_ids=summary_artifact_ids,
                    relevant_ids=case.relevant_summary_source_ids,
                    latency_ms=summary_latency,
                    context_characters=summary_characters,
                    failure_code=summary_failure,
                )
                measurements.append(
                    await _evaluate_answer_use(
                        measurement, answer_evaluator, case, summary_context
                    )
                )

    return Success(
        RetrievalEvaluationRun(
            profile_scope=profile_scope,
            retrieval_limit=retrieval_limit,
            context_character_budget=context_character_budget,
            measurements=tuple(measurements),
        )
    )
