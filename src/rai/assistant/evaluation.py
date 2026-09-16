"""Equal-budget retrieval evaluation for evidence-first assistant memory."""

from __future__ import annotations

import asyncio
import json
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    _utc_now,
)

from .ports import (
    AssistantEvidence,
    AssistantEvidenceProvider,
    AssistantModelBackend,
    MemoryGraphStore,
)
from .query import MemoryQueryResolver
from .records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryRecord,
    make_assistant_failure,
)
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
    expected_answer_phrases: tuple[str, ...] = ()
    forbidden_answer_phrases: tuple[str, ...] = ()
    expected_abstention: bool = False


class RetrievalCorpusTurn(BaseModel):
    """Frozen source turn used to seed a retrieval benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    data_class: DataClass = DataClass.LOCAL
    domain_scope: str = Field(default="general", min_length=1)
    purpose: str = Field(default="assistant", min_length=1)


class RetrievalCorpusClaim(BaseModel):
    """Frozen admitted claim used to isolate retrieval from extraction quality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(min_length=1)
    source_turn_id: str = Field(min_length=1)
    kind: str = Field(default="claim", min_length=1)
    topic: str = Field(min_length=1)
    content: dict[str, Any]
    data_class: DataClass = DataClass.LOCAL
    domain_scope: str = Field(default="general", min_length=1)
    purpose: str = Field(default="assistant", min_length=1)
    epistemic_status: Literal[
        "asserted", "observed", "inferred", "uncertain", "contested"
    ] = "asserted"


class RetrievalEvaluationCorpus(BaseModel):
    """Versioned benchmark inputs, expected evidence and answer behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    turns: tuple[RetrievalCorpusTurn, ...] = Field(min_length=1)
    claims: tuple[RetrievalCorpusClaim, ...] = Field(min_length=1)
    cases: tuple[RetrievalEvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> RetrievalEvaluationCorpus:
        """Reject ambiguous IDs and expectations pointing outside the corpus."""
        turn_ids = tuple(turn.turn_id for turn in self.turns)
        claim_ids = tuple(claim.memory_id for claim in self.claims)
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError("retrieval corpus contains duplicate turn IDs")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("retrieval corpus contains duplicate claim IDs")
        unknown_sources = {claim.source_turn_id for claim in self.claims} - set(
            turn_ids
        )
        if unknown_sources:
            raise ValueError(
                "claims reference missing source turns: "
                + ", ".join(sorted(unknown_sources))
            )
        for case in self.cases:
            if set(case.relevant_raw_turn_ids) - set(turn_ids):
                raise ValueError(f"case {case.case_id} references unknown raw turns")
            if (
                set(case.relevant_claim_ids) | set(case.relevant_summary_source_ids)
            ) - set(claim_ids):
                raise ValueError(f"case {case.case_id} references unknown claims")
        return self


class RetrievalAnswerEvaluation(BaseModel):
    """Inspectable backend answer plus independent deterministic judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer_text: str = Field(min_length=1)
    correct: bool
    abstained: bool
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    judge_version: str = Field(min_length=1)
    energy_joules: float | None = Field(default=None, ge=0.0)
    provider_cost: float | None = Field(default=None, ge=0.0)


class RetrievalAnswerEvaluator(Protocol):
    """Evaluate whether an answerer used one bounded retrieval context correctly."""

    async def evaluate(
        self,
        case: RetrievalEvaluationCase,
        channel: RetrievalChannel,
        context_items: tuple[dict[str, object], ...],
    ) -> Result[RetrievalAnswerEvaluation, ActionFailure]: ...


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
    answer_text: str | None = None
    answer_abstained: bool | None = None
    answer_latency_ms: float | None = Field(default=None, ge=0.0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    backend_name: str | None = None
    model_name: str | None = None
    judge_version: str | None = None
    provider_cost: float | None = Field(default=None, ge=0.0)
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


class RetrievalChannelAggregate(BaseModel):
    """Channel-level quality and resource report for one evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: RetrievalChannel
    case_count: int = Field(ge=0)
    mean_recall: float = Field(ge=0.0, le=1.0)
    mean_precision: float = Field(ge=0.0, le=1.0)
    answer_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_abstention_rate: float = Field(ge=0.0, le=1.0)
    answer_abstention_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_failures: int = Field(ge=0)
    answer_evaluation_failures: int = Field(ge=0)
    mean_retrieval_latency_ms: float = Field(ge=0.0)
    mean_answer_latency_ms: float | None = Field(default=None, ge=0.0)
    mean_context_characters: float = Field(ge=0.0)
    total_tokens_in: int = Field(ge=0)
    total_tokens_out: int = Field(ge=0)
    total_provider_cost: float | None = Field(default=None, ge=0.0)
    total_energy_joules: float | None = Field(default=None, ge=0.0)
    energy_measurement_coverage: float = Field(ge=0.0, le=1.0)


_EVALUATION_PRODUCER = ProducerIdentity(
    producer_id="assistant-retrieval-evaluator",
    kind="evaluation",
    version="1.0.0",
)
_ABSTENTION_MARKERS = (
    "nie wiem",
    "nie mam wystarczających",
    "brak wystarczających",
    "nie mogę odpowiedzieć",
    "i don't know",
    "insufficient evidence",
    "cannot answer",
)


def load_retrieval_evaluation_corpus(path: str | Path) -> RetrievalEvaluationCorpus:
    """Load and validate a complete, versioned retrieval benchmark fixture."""
    return RetrievalEvaluationCorpus.model_validate_json(Path(path).read_text("utf-8"))


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _item_data_class(item: dict[str, object]) -> DataClass:
    value = item.get("data_class", DataClass.LOCAL)
    return value if isinstance(value, DataClass) else DataClass(str(value))


def _judge_answer(case: RetrievalEvaluationCase, answer_text: str) -> tuple[bool, bool]:
    normalized = _normalized_text(answer_text)
    abstained = any(marker in normalized for marker in _ABSTENTION_MARKERS)
    required = all(
        _normalized_text(phrase) in normalized
        for phrase in case.expected_answer_phrases
    )
    forbidden = any(
        _normalized_text(phrase) in normalized
        for phrase in case.forbidden_answer_phrases
    )
    if case.expected_abstention:
        return abstained and not forbidden, abstained
    return required and not forbidden and not abstained, abstained


def _answer_context_content(
    case: RetrievalEvaluationCase,
    channel: RetrievalChannel,
    context_items: tuple[dict[str, object], ...],
) -> dict[str, object]:
    content: dict[str, object] = {
        "current_turn": {"role": "user", "text": case.query_text},
        "recent_turns": (),
        "episodic_evidence": (),
        "durable_memories": (),
        "grounded_summaries": (),
        "external_evidence": (),
    }
    if channel == "raw_turns_bm25":
        content["episodic_evidence"] = tuple(
            {
                "source_id": item["source_id"],
                "timestamp": "benchmark-source",
                "text": (
                    item.get("content", {}).get("text", "")
                    if isinstance(item.get("content"), dict)
                    else ""
                ),
            }
            for item in context_items
        )
    elif channel == "claims_bm25":
        content["durable_memories"] = tuple(
            {
                "record_id": item["source_id"],
                **(item["content"] if isinstance(item.get("content"), dict) else {}),
            }
            for item in context_items
        )
    else:
        content["grounded_summaries"] = tuple(context_items)
    return content


class BackendRetrievalAnswerEvaluator:
    """Answer from a product backend and judge without asking it to self-grade."""

    def __init__(
        self,
        backend: AssistantModelBackend,
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 256,
        max_latency_seconds: float = 60.0,
    ) -> None:
        self.backend = backend
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_latency_seconds = max_latency_seconds

    async def evaluate(
        self,
        case: RetrievalEvaluationCase,
        channel: RetrievalChannel,
        context_items: tuple[dict[str, object], ...],
    ) -> Result[RetrievalAnswerEvaluation, ActionFailure]:
        now = _utc_now()
        session_id = f"evaluation-{case.case_id}"
        turn_id = f"query-{case.case_id}-{channel}"
        manifest = AssistantContextManifest(
            producer=_EVALUATION_PRODUCER,
            session_id=session_id,
            turn_id=turn_id,
            routing_decision=(
                "raw_evidence_fallback"
                if channel == "raw_turns_bm25"
                else "compact_memory"
            ),
            evidence_character_budget=sum(
                len(json.dumps(item, ensure_ascii=False)) for item in context_items
            ),
            actual_characters=sum(
                len(json.dumps(item, ensure_ascii=False)) for item in context_items
            ),
            items=tuple(
                AssistantContextManifestItem(
                    source_id=str(item["source_id"]),
                    source_type=str(item["source_type"]),
                    layer=channel,
                    data_class=_item_data_class(item),
                    ranking_reason="selected by equal-budget retrieval evaluation",
                    fields=("content",),
                )
                for item in context_items
            ),
        )
        context = AssistantContextPackage(
            producer=_EVALUATION_PRODUCER,
            session_id=session_id,
            turn_id=turn_id,
            manifest=manifest,
            content=_answer_context_content(case, channel, context_items),
            provenance=tuple(
                ProvenanceReference(
                    source_id=str(item["source_id"]),
                    source_type=str(item["source_type"]),
                    source_version="1.0.0",
                    relation="EVALUATED_WITH",
                    producer=_EVALUATION_PRODUCER,
                )
                for item in context_items
            ),
        )
        budget = InferenceBudget(
            producer=_EVALUATION_PRODUCER,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_agent_turns=1,
            max_tool_calls=0,
            max_images=0,
            max_audio_seconds=0,
            max_latency_seconds=self.max_latency_seconds,
            max_provider_cost=0,
            max_ram_bytes=2 * 1024 * 1024 * 1024,
            max_vram_bytes=0,
            cancellation_deadline=now + timedelta(seconds=self.max_latency_seconds),
        )
        request = InferenceRequest(
            producer=_EVALUATION_PRODUCER,
            session_id=session_id,
            turn_id=turn_id,
            request_id=f"answer-{case.case_id}-{channel}",
            context=context,
            budget=budget,
            model_name=str(getattr(self.backend, "model_name", "unknown")),
            system_instruction=(
                "Answer the user using only the supplied memory evidence. "
                "If it does not support an answer, explicitly say that you do not know."
            ),
        )
        started = time.perf_counter()
        try:
            generated = await asyncio.wait_for(
                self.backend.generate(request, CancellationToken()),
                timeout=self.max_latency_seconds,
            )
        except TimeoutError:
            return Failure(
                make_assistant_failure(
                    code="ANSWER_EVALUATION_TIMEOUT",
                    message="answer backend exceeded the evaluation latency budget",
                    request_id=request.request_id,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1_000
        if isinstance(generated, Failure):
            return Failure(generated.failure())
        candidate = generated.unwrap()
        correct, abstained = _judge_answer(case, candidate.text)
        energy = candidate.metadata.get("energy_joules")
        cost = candidate.metadata.get("provider_cost")
        return Success(
            RetrievalAnswerEvaluation(
                answer_text=candidate.text,
                correct=correct,
                abstained=abstained,
                tokens_in=candidate.tokens_in,
                tokens_out=candidate.tokens_out,
                latency_ms=latency_ms,
                backend_name=str(getattr(self.backend, "backend_name", "unknown")),
                model_name=str(getattr(self.backend, "model_name", "unknown")),
                judge_version="deterministic-phrase-and-abstention-v1",
                energy_joules=(
                    float(energy)
                    if isinstance(energy, (int, float)) and not isinstance(energy, bool)
                    else None
                ),
                provider_cost=(
                    float(cost)
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                    else None
                ),
            )
        )


async def seed_retrieval_evaluation_corpus(
    store: MemoryGraphStore,
    corpus: RetrievalEvaluationCorpus,
) -> Result[None, ActionFailure]:
    """Seed frozen raw turns and admitted claims, keeping extraction out of M4."""
    claims_by_turn: dict[str, list[RetrievalCorpusClaim]] = {}
    for claim in corpus.claims:
        claims_by_turn.setdefault(claim.source_turn_id, []).append(claim)
    known_turns = {turn.turn_id for turn in corpus.turns}
    missing_sources = set(claims_by_turn) - known_turns
    if missing_sources:
        return Failure(
            make_assistant_failure(
                code="INVALID_EVALUATION_CORPUS",
                message=(
                    "claims reference missing source turns: "
                    + ", ".join(sorted(missing_sources))
                ),
            )
        )

    for source in corpus.turns:
        turn = ConversationTurn(
            record_id=source.turn_id,
            producer=_EVALUATION_PRODUCER,
            session_id=source.session_id,
            role="user",
            text=source.text,
            data_class=source.data_class,
            domain_scope=source.domain_scope,
            purpose=source.purpose,
            status="COMPLETED",
            metadata={"evaluation_corpus": corpus.corpus_version},
        )
        accepted = await store.accept_turn(turn)
        if isinstance(accepted, Failure):
            return Failure(accepted.failure())

        source_claims = tuple(claims_by_turn.get(source.turn_id, ()))
        if not source_claims:
            continue
        memories = tuple(
            MemoryRecord(
                record_id=claim.memory_id,
                producer=_EVALUATION_PRODUCER,
                kind=claim.kind,
                topic=claim.topic,
                content=claim.content,
                source_turn_id=source.turn_id,
                data_class=claim.data_class,
                domain_scope=claim.domain_scope,
                purpose=claim.purpose,
                epistemic_status=claim.epistemic_status,
                provenance=(
                    ProvenanceReference(
                        source_id=source.turn_id,
                        source_type="conversation_turn",
                        source_version="1.0.0",
                        relation="DERIVED_FROM",
                        producer=_EVALUATION_PRODUCER,
                    ),
                ),
            )
            for claim in source_claims
        )
        manifest_id = f"evaluation-manifest-{source.turn_id}"
        manifest = AssistantContextManifest(
            record_id=manifest_id,
            producer=_EVALUATION_PRODUCER,
            session_id=source.session_id,
            turn_id=source.turn_id,
        )
        response = AssistantResponse(
            record_id=f"evaluation-response-{source.turn_id}",
            producer=_EVALUATION_PRODUCER,
            session_id=source.session_id,
            turn_id=f"evaluation-assistant-{source.turn_id}",
            user_turn_id=source.turn_id,
            request_id=f"evaluation-request-{source.turn_id}",
            manifest_id=manifest_id,
            text="Evaluation corpus ingestion completed.",
            admitted_memory_ids=tuple(memory.record_id for memory in memories),
        )
        committed = await store.commit_terminal(
            response=response,
            manifest=manifest,
            assistant_turn=None,
            memories=memories,
            relations=(),
        )
        if isinstance(committed, Failure):
            return Failure(committed.failure())
    return Success(None)


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
                "data_class": turn.data_class,
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
                "data_class": memory.data_class,
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
                "data_class": evidence.data_class,
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
    evaluation = result.unwrap()
    return measurement.model_copy(
        update={
            "answer_correct": evaluation.correct,
            "answer_utilization_evaluated": True,
            "answer_text": evaluation.answer_text,
            "answer_abstained": evaluation.abstained,
            "answer_latency_ms": evaluation.latency_ms,
            "tokens_in": evaluation.tokens_in,
            "tokens_out": evaluation.tokens_out,
            "backend_name": evaluation.backend_name,
            "model_name": evaluation.model_name,
            "judge_version": evaluation.judge_version,
            "energy_joules": evaluation.energy_joules,
            "provider_cost": evaluation.provider_cost,
        }
    )


def aggregate_retrieval_run(
    run: RetrievalEvaluationRun,
) -> tuple[RetrievalChannelAggregate, ...]:
    """Aggregate quality and resource use without hiding missing measurements."""
    aggregates: list[RetrievalChannelAggregate] = []
    for channel in (
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
    ):
        items = tuple(item for item in run.measurements if item.channel == channel)
        if not items:
            continue
        evaluated = tuple(item for item in items if item.answer_utilization_evaluated)
        energy_values = tuple(
            item.energy_joules for item in evaluated if item.energy_joules is not None
        )
        cost_values = tuple(
            item.provider_cost for item in evaluated if item.provider_cost is not None
        )
        answer_latencies = tuple(
            item.answer_latency_ms
            for item in evaluated
            if item.answer_latency_ms is not None
        )
        aggregates.append(
            RetrievalChannelAggregate(
                channel=channel,
                case_count=len(items),
                mean_recall=sum(item.recall for item in items) / len(items),
                mean_precision=sum(item.precision for item in items) / len(items),
                answer_accuracy=(
                    sum(item.answer_correct is True for item in evaluated)
                    / len(evaluated)
                    if evaluated
                    else None
                ),
                retrieval_abstention_rate=(
                    sum(item.abstained for item in items) / len(items)
                ),
                answer_abstention_rate=(
                    sum(item.answer_abstained is True for item in evaluated)
                    / len(evaluated)
                    if evaluated
                    else None
                ),
                retrieval_failures=sum(item.retrieval_failed for item in items),
                answer_evaluation_failures=sum(
                    item.answer_evaluation_failed for item in items
                ),
                mean_retrieval_latency_ms=(
                    sum(item.latency_ms for item in items) / len(items)
                ),
                mean_answer_latency_ms=(
                    sum(answer_latencies) / len(answer_latencies)
                    if answer_latencies
                    else None
                ),
                mean_context_characters=(
                    sum(item.context_characters for item in items) / len(items)
                ),
                total_tokens_in=sum(item.tokens_in or 0 for item in evaluated),
                total_tokens_out=sum(item.tokens_out or 0 for item in evaluated),
                total_provider_cost=(sum(cost_values) if cost_values else None),
                total_energy_joules=(sum(energy_values) if energy_values else None),
                energy_measurement_coverage=(
                    len(energy_values) / len(evaluated) if evaluated else 0.0
                ),
            )
        )
    return tuple(aggregates)


async def evaluate_retrieval_floor(  # noqa: PLR0913
    store: MemoryGraphStore,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    profile_scope: str = "default",
    retrieval_limit: int = 5,
    context_character_budget: int = 8_000,
    summary_provider: AssistantEvidenceProvider | None = None,
    answer_evaluator: RetrievalAnswerEvaluator | None = None,
    corpus_version: str = "rai-retrieval-floor-v1",
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
            corpus_version=corpus_version,
            profile_scope=profile_scope,
            retrieval_limit=retrieval_limit,
            context_character_budget=context_character_budget,
            measurements=tuple(measurements),
        )
    )
