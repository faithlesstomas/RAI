"""Equal-budget retrieval evaluation for evidence-first assistant memory."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass, _utc_now

from .ports import MemoryGraphStore
from .query import MemoryQueryResolver
from .records import ConversationTurn, MemoryRecord


class RetrievalEvaluationCase(BaseModel):
    """One fixed query with channel-specific ground-truth source IDs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_raw_turn_ids: tuple[str, ...] = ()
    relevant_claim_ids: tuple[str, ...] = ()
    data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL)


class RetrievalChannelMeasurement(BaseModel):
    """Comparable output for one case and retrieval channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    channel: Literal["raw_turns_bm25", "claims_bm25"]
    retrieved_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    abstained: bool
    latency_ms: float = Field(ge=0.0)
    context_characters: int = Field(ge=0)
    answer_correct: bool | None = None
    answer_utilization_evaluated: bool = False
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
    channel: Literal["raw_turns_bm25", "claims_bm25"],
    retrieved_ids: tuple[str, ...],
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
        relevant_ids=relevant_ids,
        recall=recall,
        precision=precision,
        abstained=not retrieved_ids,
        latency_ms=latency_ms,
        context_characters=context_characters,
        retrieval_failed=failure_code is not None,
        failure_code=failure_code,
    )


def _bounded_raw_ids(
    items: tuple[tuple[ConversationTurn, str], ...], character_budget: int
) -> tuple[tuple[str, ...], int]:
    selected: list[str] = []
    used = 0
    for turn, _reason in items:
        item_size = len(turn.text)
        if used + item_size > character_budget:
            continue
        selected.append(turn.record_id)
        used += item_size
    return tuple(selected), used


def _bounded_claim_ids(
    items: tuple[tuple[MemoryRecord, str], ...], character_budget: int
) -> tuple[tuple[str, ...], int]:
    selected: list[str] = []
    used = 0
    for memory, _reason in items:
        item_size = len(json.dumps(memory.content, ensure_ascii=False))
        if used + item_size > character_budget:
            continue
        selected.append(memory.record_id)
        used += item_size
    return tuple(selected), used


async def evaluate_retrieval_floor(
    store: MemoryGraphStore,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    profile_scope: str = "default",
    retrieval_limit: int = 5,
    context_character_budget: int = 8_000,
) -> Result[RetrievalEvaluationRun, ActionFailure]:
    """Evaluate raw-turn and claim BM25 channels under identical budgets."""
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
            measurements.append(
                _metrics(
                    case_id=case.case_id,
                    channel="raw_turns_bm25",
                    retrieved_ids=(),
                    relevant_ids=case.relevant_raw_turn_ids,
                    latency_ms=raw_latency,
                    context_characters=0,
                    failure_code=failure.code,
                )
            )
        else:
            raw_items = raw_result.unwrap()
            raw_ids, raw_characters = _bounded_raw_ids(
                raw_items, context_character_budget
            )
            measurements.append(
                _metrics(
                    case_id=case.case_id,
                    channel="raw_turns_bm25",
                    retrieved_ids=raw_ids,
                    relevant_ids=case.relevant_raw_turn_ids,
                    latency_ms=raw_latency,
                    context_characters=raw_characters,
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
            measurements.append(
                _metrics(
                    case_id=case.case_id,
                    channel="claims_bm25",
                    retrieved_ids=(),
                    relevant_ids=case.relevant_claim_ids,
                    latency_ms=claim_latency,
                    context_characters=0,
                    failure_code=failure.code,
                )
            )
        else:
            claim_items = claim_result.unwrap()
            claim_ids, claim_characters = _bounded_claim_ids(
                claim_items, context_character_budget
            )
            measurements.append(
                _metrics(
                    case_id=case.case_id,
                    channel="claims_bm25",
                    retrieved_ids=claim_ids,
                    relevant_ids=case.relevant_claim_ids,
                    latency_ms=claim_latency,
                    context_characters=claim_characters,
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
