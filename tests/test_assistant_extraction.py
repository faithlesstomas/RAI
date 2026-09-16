"""Conformance tests for schema-constrained assistant memory extraction."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from returns.result import Failure, Result, Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.backends.local import LocalAssistantBackend
from rai.assistant.context import (
    AssistantContextBuilder,
    assess_memory_sufficiency,
)
from rai.assistant.evidence import RichHistoryEvidenceProvider
from rai.assistant.evaluation import (
    BackendRetrievalAnswerEvaluator,
    RetrievalAnswerEvaluation,
    RetrievalChannel,
    RetrievalEvaluationCase,
    aggregate_retrieval_run,
    evaluate_retrieval_floor,
    load_retrieval_evaluation_corpus,
    seed_retrieval_evaluation_corpus,
)
from rai.assistant.extraction import SchemaConstrainedMemoryExtractor
from rai.assistant.ports import MemoryQuery
from rai.assistant.query import MemoryQueryResolver, domain_scope_matches
from rai.assistant.records import (
    ConversationTurn,
    MemoryRecord,
    MemoryRelationKind,
    make_assistant_failure,
)
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.assistant.summary import GroundedClaimSummaryProvider
from rai.inference.protocols import InferenceResult
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    Episode,
    ProducerIdentity,
    ProvenanceReference,
    _utc_now,
)

PRODUCER = ProducerIdentity(
    producer_id="assistant-extraction-test", kind="test", version="1.0.0"
)
EVALUATION_CONTEXT_BUDGET = 4096
EXTRACTED_CLAIM_CONFIDENCE = 0.94
STATIC_EVALUATOR_TOKEN_COUNT = 2
EVALUATION_CHANNEL_COUNT = 3
MEMORY_SUFFICIENCY_THRESHOLD = 0.75
MIN_EXPECTED_PROJECT_PRECISION = 0.5
EXPECTED_PARTIAL_QUERY_COVERAGE = 0.5


class _StaticEngine:
    def __init__(self, text: str) -> None:
        self.text = text
        self.model_name = "schema-test-model"
        self.is_loaded = True

    async def load(self):  # noqa: ANN202
        return Success(None)

    async def generate(self, **kwargs):  # noqa: ANN003, ANN202
        del kwargs
        return Success(InferenceResult(text=self.text))

    async def unload(self):  # noqa: ANN202
        return Success(None)


class _EvidenceBoundEngine:
    """Return the supported answer only when its evidence reached the prompt."""

    def __init__(self, expected_evidence: str, supported_answer: str) -> None:
        self.expected_evidence = expected_evidence
        self.supported_answer = supported_answer
        self.model_name = "evidence-bound-test-model"
        self.is_loaded = True
        self.prompts: list[str] = []

    async def load(self):  # noqa: ANN202
        return Success(None)

    async def generate(self, **kwargs):  # noqa: ANN003, ANN202
        prompt = str(kwargs.get("prompt", ""))
        self.prompts.append(prompt)
        answer = (
            self.supported_answer
            if self.expected_evidence.casefold() in prompt.casefold()
            else "Nie mam wystarczających źródeł, aby odpowiedzieć."
        )
        return Success(InferenceResult(text=answer))

    async def unload(self):  # noqa: ANN202
        return Success(None)


class _StaticRichHistory:
    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.calls = 0

    def query(self, **filters):  # noqa: ANN003, ANN202
        del filters
        self.calls += 1
        return (self.episode,)


class _StaticAnswerEvaluator:
    def __init__(self, fail_channel: RetrievalChannel | None = None) -> None:
        self.calls: list[RetrievalChannel] = []
        self.fail_channel = fail_channel

    async def evaluate(
        self,
        case: RetrievalEvaluationCase,
        channel: RetrievalChannel,
        context_items: tuple[dict[str, object], ...],
    ) -> Result[RetrievalAnswerEvaluation, ActionFailure]:
        del case
        self.calls.append(channel)
        if channel == self.fail_channel:
            return Failure(
                make_assistant_failure(
                    code="ANSWER_USE_FAILED",
                    message="test answer evaluator failure",
                )
            )
        return Success(
            RetrievalAnswerEvaluation(
                answer_text="context used" if context_items else "no context",
                correct=bool(context_items),
                abstained=not context_items,
                tokens_in=STATIC_EVALUATOR_TOKEN_COUNT,
                tokens_out=STATIC_EVALUATOR_TOKEN_COUNT,
                latency_ms=1.0,
                backend_name="static",
                model_name="static",
                judge_version="static-v1",
            )
        )


def _turn(record_id: str, text: str) -> ConversationTurn:
    return ConversationTurn(
        record_id=record_id,
        producer=PRODUCER,
        session_id="extraction-session",
        role="user",
        text=text,
    )


def _extraction_json(
    source_id: str,
    source_span: str,
    *,
    start: int = 0,
    value: str = "Aurora",
    statement_type: str = "assertion",
) -> str:
    end = start + len(source_span)
    return (
        "{"
        '"candidates":[{'
        f'"source_id":"{source_id}",'
        '"subject":"user","predicate":"current_project",'
        f'"value":"{value}","claim_kind":"fact",'
        f'"statement_type":"{statement_type}",'
        '"modality":"direct","negated":false,'
        '"scope":"project",'
        f'"source_span":"{source_span}","span_start":{start},"span_end":{end},'
        '"confidence":0.94,"salience":0.9,"valid_from":null,"valid_until":null'
        '}],"confidence":0.93}'
    )


@pytest.mark.asyncio
async def test_schema_extractor_preserves_arbitrary_claim_and_exact_span() -> None:
    text = "Pracuję nad projektem Aurora i dziś porządkuję testy."
    span = "Pracuję nad projektem Aurora"
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-project", span)),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )

    result = await extractor.extract(_turn("turn-project", text), CancellationToken())

    assert isinstance(result, Success)
    proposal = result.unwrap()[0]
    assert proposal.source_span == span
    assert text[proposal.span_start : proposal.span_end] == span
    assert proposal.content["predicate"] == "current_project"
    assert proposal.content["value"] == "Aurora"
    assert proposal.topic == "claim.project.user.current.project"


@pytest.mark.asyncio
async def test_schema_extractor_rejects_hallucinated_source_span() -> None:
    text = "Pracuję nad projektem Aurora."
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-project", "Projekt Borealis")),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )

    result = await extractor.extract(_turn("turn-project", text), CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "INVALID_MEMORY_EXTRACTION"


@pytest.mark.asyncio
async def test_service_admits_model_extracted_claim_and_recalls_it_cross_session(
    tmp_path: Path,
) -> None:
    first_text = "Pracuję nad projektem Aurora i dziś porządkuję testy."
    span = "Pracuję nad projektem Aurora"
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-project", span)),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )
    store = SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        memory_extractor=extractor,
    )
    await service.start()

    remembered = await service.accept_turn(_turn("turn-project", first_text))
    service.memory_extractor = None
    recalled = await service.accept_turn(
        ConversationTurn(
            record_id="turn-recall",
            producer=PRODUCER,
            session_id="other-session",
            role="user",
            text="Nad jakim projektem pracuję?",
        )
    )

    assert isinstance(remembered, Success)
    assert remembered.unwrap().admitted_memory_ids
    assert isinstance(recalled, Success)
    assert "Aurora" in recalled.unwrap().text
    context = await store.get_context_package(recalled.unwrap().manifest_id)
    assert isinstance(context, Success)
    package = context.unwrap()
    assert package is not None
    assert package.manifest.routing_decision == "compact_memory"
    assert package.manifest.episodic_turn_ids == ()
    assert package.manifest.sufficiency_factors["query_coverage"] == 1.0
    assert package.manifest.sufficiency_factors["evidence_quality"] == pytest.approx(
        EXTRACTED_CLAIM_CONFIDENCE
    )
    assert package.manifest.sufficiency_reasons
    operations = await store.list_memory_operations()
    assert isinstance(operations, Success)
    assert [operation.operation for operation in operations.unwrap()] == [
        "REFLECT",
        "REMEMBER",
    ]


@pytest.mark.asyncio
async def test_partial_high_confidence_claim_does_not_suppress_raw_evidence(
    tmp_path: Path,
) -> None:
    source_text = "Pracuję nad projektem Aurora."
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-partial-source", source_text[:-1])),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )
    store = SQLiteMemoryGraphStore(tmp_path / "partial-sufficiency.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        memory_extractor=extractor,
    )
    await service.start()
    seeded = await service.accept_turn(_turn("turn-partial-source", source_text))
    service.memory_extractor = None

    recalled = await service.accept_turn(
        ConversationTurn(
            record_id="turn-partial-query",
            producer=PRODUCER,
            session_id="partial-query-session",
            role="user",
            text="Jaki jest budżet i termin projektu Aurora?",
        )
    )

    assert isinstance(seeded, Success)
    assert isinstance(recalled, Success)
    context = await store.get_context_package(recalled.unwrap().manifest_id)
    assert isinstance(context, Success)
    package = context.unwrap()
    assert package is not None
    assert package.manifest.routing_decision == "raw_evidence_fallback"
    assert package.manifest.fallback_used is True
    assert package.manifest.sufficiency_score < MEMORY_SUFFICIENCY_THRESHOLD
    assert (
        package.manifest.sufficiency_factors["query_coverage"]
        == EXPECTED_PARTIAL_QUERY_COVERAGE
    )
    assert package.manifest.episodic_turn_ids == ("turn-partial-source",)
    assert any(
        reason.startswith("compact_memory:insufficient(coverage=0.500")
        for reason in package.manifest.rejected_routes
    )


@pytest.mark.asyncio
async def test_context_recovers_relevant_raw_turn_when_no_claim_was_extracted(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "episodic.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
    )
    await service.start()
    source = _turn(
        "turn-raw-source",
        "Wczoraj naprawiłem parser AST w projekcie Aurora.",
    )
    first = await service.accept_turn(source)

    recall = await service.accept_turn(
        ConversationTurn(
            record_id="turn-raw-recall",
            producer=PRODUCER,
            session_id="separate-session",
            role="user",
            text="Co naprawiłem w projekcie Aurora?",
        )
    )

    assert isinstance(first, Success)
    assert first.unwrap().admitted_memory_ids == ()
    assert isinstance(recall, Success)
    context = await store.get_context_package(recall.unwrap().manifest_id)
    assert isinstance(context, Success)
    package = context.unwrap()
    assert package is not None
    assert package.manifest.routing_decision == "raw_evidence_fallback"
    assert package.manifest.episodic_turn_ids == ("turn-raw-source",)
    evidence = package.content["episodic_evidence"]
    assert isinstance(evidence, (list, tuple))
    assert evidence[0]["text"] == source.text
    assert any(
        item.source_id == source.record_id and item.layer == "episodic_evidence"
        for item in package.manifest.items
    )


@pytest.mark.asyncio
async def test_raw_turn_retrieval_respects_profile_scope(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "profiles.sqlite3")
    alpha = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        profile_scope="alpha",
    )
    await alpha.start()
    await alpha.accept_turn(_turn("turn-alpha", "Pracuję nad projektem Aurora."))
    beta = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        profile_scope="beta",
    )
    await beta.start()

    recall = await beta.accept_turn(
        ConversationTurn(
            record_id="turn-beta",
            producer=PRODUCER,
            session_id="beta-session",
            role="user",
            text="Nad jakim projektem pracuję?",
        )
    )

    assert isinstance(recall, Success)
    context = await store.get_context_package(recall.unwrap().manifest_id)
    assert isinstance(context, Success)
    package = context.unwrap()
    assert package is not None
    assert package.manifest.episodic_turn_ids == ()


@pytest.mark.asyncio
async def test_context_retrieves_rich_history_as_external_evidence(
    tmp_path: Path,
) -> None:
    now = _utc_now()
    episode = Episode(
        record_id="episode-aurora",
        timestamp=now,
        producer=PRODUCER,
        started_at=now - timedelta(minutes=20),
        ended_at=now,
        observation_ids=("observation-1",),
        provenance=(
            ProvenanceReference(
                source_id="observation-1",
                source_type="observation",
                source_version="1.0.0",
                relation="DERIVED_FROM",
                producer=PRODUCER,
            ),
        ),
        applications=("codex",),
        projects=("Aurora",),
        activity_types=("coding",),
        confidence=0.91,
        data_class=DataClass.LOCAL,
    )
    history = _StaticRichHistory(episode)
    provider = RichHistoryEvidenceProvider(history)
    store = SQLiteMemoryGraphStore(tmp_path / "rich-history-context.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        context_builder=AssistantContextBuilder(
            store=store,
            evidence_providers=(provider,),
        ),
    )
    await service.start()

    response = await service.accept_turn(
        ConversationTurn(
            record_id="turn-history-query",
            producer=PRODUCER,
            session_id="history-session",
            role="user",
            text="Co robiłem w projekcie Aurora?",
        )
    )

    assert isinstance(response, Success)
    context = await store.get_context_package(response.unwrap().manifest_id)
    assert isinstance(context, Success)
    package = context.unwrap()
    assert package is not None
    assert package.manifest.routing_decision == "raw_evidence_fallback"
    assert package.manifest.external_evidence_ids == (episode.record_id,)
    assert history.calls == 1
    evidence = package.content["external_evidence"]
    assert isinstance(evidence, (list, tuple))
    assert evidence[0]["source_type"] == "rich_history_episode"
    assert evidence[0]["content"]["projects"] == ("Aurora",)


@pytest.mark.asyncio
async def test_conflict_requires_correction_then_preserves_qualified_relations(
    tmp_path: Path,
) -> None:
    first_text = "Pracuję nad projektem Aurora."
    engine = _StaticEngine(
        _extraction_json("turn-project-a", first_text[:-1], value="Aurora")
    )
    extractor = SchemaConstrainedMemoryExtractor(
        engine,  # type: ignore[arg-type]
        model_name="schema-test-model",
    )
    store = SQLiteMemoryGraphStore(tmp_path / "conflict.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        memory_extractor=extractor,
    )
    await service.start()
    first = await service.accept_turn(_turn("turn-project-a", first_text))

    conflict_text = "Pracuję nad projektem Borealis."
    engine.text = _extraction_json(
        "turn-project-b", conflict_text[:-1], value="Borealis"
    )
    conflict = await service.accept_turn(_turn("turn-project-b", conflict_text))

    correction_text = "Korekta: pracuję nad projektem Borealis."
    correction_span = "pracuję nad projektem Borealis"
    engine.text = _extraction_json(
        "turn-project-c",
        correction_span,
        start=9,
        value="Borealis",
        statement_type="correction",
    )
    corrected = await service.accept_turn(_turn("turn-project-c", correction_text))

    assert isinstance(first, Success)
    assert isinstance(conflict, Success)
    assert conflict.unwrap().admitted_memory_ids == ()
    assert "explicit correction" in conflict.unwrap().text
    assert isinstance(corrected, Success)
    assert len(corrected.unwrap().admitted_memory_ids) == 1
    current = await store.retrieve_relevant_memories(
        query=MemoryQuery(topic="claim.project.user.current.project")
    )
    assert isinstance(current, Success)
    assert current.unwrap()[0][0].content["value"] == "Borealis"
    supersedes = await store.get_relations(kind=MemoryRelationKind.SUPERSEDES)
    contradicts = await store.get_relations(kind=MemoryRelationKind.CONTRADICTS)
    updates = await store.get_relations(kind=MemoryRelationKind.UPDATES)
    assert isinstance(supersedes, Success)
    assert isinstance(contradicts, Success)
    assert isinstance(updates, Success)
    assert len(supersedes.unwrap()) == 1
    assert len(contradicts.unwrap()) == 1
    assert len(updates.unwrap()) == 1
    assert contradicts.unwrap()[0].epistemic_status == "contested"
    assert contradicts.unwrap()[0].provenance
    assert contradicts.unwrap()[0].policy_outcome == "ALLOW"
    assert contradicts.unwrap()[0].eligible is True


@pytest.mark.asyncio
async def test_retrieval_floor_compares_raw_and_claim_bm25_under_equal_budgets(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    text = "Pracuję nad projektem Aurora i porządkuję testy."
    span = "Pracuję nad projektem Aurora"
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-eval", span)),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )
    store = SQLiteMemoryGraphStore(tmp_path / "evaluation.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        memory_extractor=extractor,
    )
    await service.start()
    admitted = await service.accept_turn(_turn("turn-eval", text))
    assert isinstance(admitted, Success)
    memory_id = admitted.unwrap().admitted_memory_ids[0]
    summary_provider = GroundedClaimSummaryProvider(store)
    evaluator = _StaticAnswerEvaluator()

    summary_result = await summary_provider.retrieve(
        MemoryQueryResolver.resolve("Co pamiętasz o projekcie Aurora?"),
        (DataClass.PUBLIC, DataClass.LOCAL),
        limit=1,
    )
    assert isinstance(summary_result, Success)
    summary = summary_result.unwrap()[0]
    assert summary.content["source_memory_ids"] == (memory_id,)
    assert summary.content["source_turn_ids"] == ("turn-eval",)
    assert summary.content["source_coverage"] == 1.0
    assert summary.content["confidence"] == EXTRACTED_CLAIM_CONFIDENCE
    assert summary.content["modalities"] == ("direct",)

    evaluated = await evaluate_retrieval_floor(
        store,
        (
            RetrievalEvaluationCase(
                case_id="project-recall",
                query_text="Co pamiętasz o projekcie Aurora?",
                relevant_raw_turn_ids=("turn-eval",),
                relevant_claim_ids=(memory_id,),
                relevant_summary_source_ids=(memory_id,),
            ),
        ),
        retrieval_limit=1,
        context_character_budget=EVALUATION_CONTEXT_BUDGET,
        summary_provider=summary_provider,
        answer_evaluator=evaluator,
    )

    assert isinstance(evaluated, Success)
    run = evaluated.unwrap()
    assert run.retrieval_limit == 1
    assert run.context_character_budget == EVALUATION_CONTEXT_BUDGET
    assert {item.channel for item in run.measurements} == {
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
    }
    assert all(item.recall == 1.0 for item in run.measurements)
    assert all(item.precision == 1.0 for item in run.measurements)
    assert all(
        item.context_characters <= EVALUATION_CONTEXT_BUDGET
        for item in run.measurements
    )
    assert all(not item.retrieval_failed for item in run.measurements)
    assert all(item.answer_utilization_evaluated for item in run.measurements)
    assert all(item.answer_correct is True for item in run.measurements)
    assert all(not item.answer_evaluation_failed for item in run.measurements)
    assert all(item.answer_text == "context used" for item in run.measurements)
    assert all(
        item.tokens_in == STATIC_EVALUATOR_TOKEN_COUNT for item in run.measurements
    )
    assert set(evaluator.calls) == {
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
    }
    summary_measurement = next(
        item for item in run.measurements if item.channel == "summaries_grounded"
    )
    assert summary_measurement.retrieved_ids == (memory_id,)
    assert summary_measurement.artifact_ids[0].startswith("grounded-summary:")

    failed_answer_run = await evaluate_retrieval_floor(
        store,
        (
            RetrievalEvaluationCase(
                case_id="answer-failure",
                query_text="Co pamiętasz o projekcie Aurora?",
                relevant_raw_turn_ids=("turn-eval",),
                relevant_claim_ids=(memory_id,),
            ),
        ),
        retrieval_limit=1,
        context_character_budget=EVALUATION_CONTEXT_BUDGET,
        answer_evaluator=_StaticAnswerEvaluator(fail_channel="claims_bm25"),
    )
    assert isinstance(failed_answer_run, Success)
    failed_claim_measurement = next(
        item
        for item in failed_answer_run.unwrap().measurements
        if item.channel == "claims_bm25"
    )
    assert not failed_claim_measurement.retrieval_failed
    assert failed_claim_measurement.answer_evaluation_failed
    assert not failed_claim_measurement.answer_utilization_evaluated
    assert failed_claim_measurement.answer_failure_code == "ANSWER_USE_FAILED"

    aggregates = aggregate_retrieval_run(run)
    assert {item.channel for item in aggregates} == {
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
    }
    assert all(item.answer_accuracy == 1.0 for item in aggregates)
    assert all(
        item.total_tokens_in == STATIC_EVALUATOR_TOKEN_COUNT for item in aggregates
    )
    assert all(item.total_energy_joules is None for item in aggregates)
    assert all(item.energy_measurement_coverage == 0.0 for item in aggregates)

    deleted = await store.delete_turn("turn-eval")
    assert isinstance(deleted, Success)
    after_delete = await summary_provider.retrieve(
        MemoryQueryResolver.resolve("Co pamiętasz o projekcie Aurora?"),
        (DataClass.PUBLIC, DataClass.LOCAL),
        limit=1,
    )
    assert isinstance(after_delete, Success)
    assert after_delete.unwrap() == ()


@pytest.mark.asyncio
async def test_versioned_corpus_runs_through_backend_and_independent_judge(
    tmp_path: Path,
) -> None:
    corpus_path = (
        Path(__file__).parent
        / "fixtures"
        / "assistant"
        / "v1"
        / "retrieval-floor.corpus.json"
    )
    corpus = load_retrieval_evaluation_corpus(corpus_path)
    assert corpus.corpus_version == "rai-assistant-retrieval-floor-v1"
    assert {case.case_id for case in corpus.cases} >= {
        "personal-name",
        "project-goal",
        "system-state",
        "conversation-commitment",
        "correction-current",
        "unsupported-abstention",
    }

    store = SQLiteMemoryGraphStore(tmp_path / "corpus.sqlite3")
    await store.start()
    seeded = await seed_retrieval_evaluation_corpus(store, corpus)
    assert isinstance(seeded, Success)

    project_case = next(case for case in corpus.cases if case.case_id == "project-goal")
    engine = _EvidenceBoundEngine(
        expected_evidence="asystenta",
        supported_answer="Celem projektu Aurora jest lokalna pamięć asystenta.",
    )
    backend = LocalAssistantBackend(
        engine=engine,
        model_name="evaluation-static-model",
        backend_name="evaluation-local-adapter",
    )
    started = await backend.start()
    assert isinstance(started, Success)
    evaluated = await evaluate_retrieval_floor(
        store,
        (project_case,),
        retrieval_limit=2,
        context_character_budget=EVALUATION_CONTEXT_BUDGET,
        summary_provider=GroundedClaimSummaryProvider(store),
        answer_evaluator=BackendRetrievalAnswerEvaluator(backend),
        corpus_version=corpus.corpus_version,
    )

    assert isinstance(evaluated, Success)
    assert evaluated.unwrap().corpus_version == corpus.corpus_version
    measurements = evaluated.unwrap().measurements
    assert {item.channel for item in measurements} == {
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
    }
    assert all(item.answer_utilization_evaluated for item in measurements)
    assert all(item.answer_correct is True for item in measurements), [
        (
            item.channel,
            item.retrieved_ids,
            item.answer_text,
            item.answer_correct,
            item.recall,
            item.precision,
        )
        for item in measurements
    ]
    assert all(item.recall == 1.0 for item in measurements)
    assert all(item.precision >= MIN_EXPECTED_PROJECT_PRECISION for item in measurements)
    assert all(item.retrieved_ids for item in measurements)
    assert all(item.answer_text is not None for item in measurements)
    assert all(item.backend_name == "evaluation-local-adapter" for item in measurements)
    assert all(item.judge_version for item in measurements)
    assert all(item.tokens_out is not None for item in measurements)
    assert all(item.answer_latency_ms is not None for item in measurements)
    assert all(item.energy_joules is None for item in measurements)
    assert len(engine.prompts) == EVALUATION_CHANNEL_COUNT
    assert all(
        engine.expected_evidence.casefold() in prompt.casefold()
        for prompt in engine.prompts
    )

    engine.expected_evidence = "evidence-that-is-not-present"
    abstention_cases = tuple(
        case
        for case in corpus.cases
        if case.case_id in {"unsupported-abstention", "private-purpose-isolation"}
    )
    abstention_run = await evaluate_retrieval_floor(
        store,
        abstention_cases,
        retrieval_limit=2,
        context_character_budget=EVALUATION_CONTEXT_BUDGET,
        summary_provider=GroundedClaimSummaryProvider(store),
        answer_evaluator=BackendRetrievalAnswerEvaluator(backend),
    )
    assert isinstance(abstention_run, Success)
    for item in abstention_run.unwrap().measurements:
        assert item.answer_correct is True and item.answer_abstained is True, (
            item.case_id,
            item.channel,
            item.retrieved_ids,
            item.answer_text,
            item.answer_correct,
            item.answer_abstained,
        )
    private_measurements = tuple(
        item
        for item in abstention_run.unwrap().measurements
        if item.case_id == "private-purpose-isolation"
    )
    private_source_ids = {
        "corpus-turn-private-hobby",
        "corpus-memory-private-hobby",
    }
    assert all(
        private_source_ids.isdisjoint(item.retrieved_ids)
        for item in private_measurements
    )

    stopped = await backend.stop()
    assert isinstance(stopped, Success)
    await store.stop()


def test_query_resolver_and_domain_policy_support_hierarchical_scopes() -> None:
    project_query = MemoryQueryResolver.resolve("Jaki jest cel projektu Aurora?")
    broad_query = MemoryQueryResolver.resolve("Co pamiętasz?")
    conversation_query = MemoryQueryResolver.resolve(
        "Co ustaliliśmy zrobić przed dodaniem embeddingów?"
    )

    assert project_query.domain_scopes == ("global", "project:aurora")
    assert domain_scope_matches("project", project_query.domain_scopes)
    assert domain_scope_matches("project:aurora", project_query.domain_scopes)
    assert not domain_scope_matches("project:borealis", project_query.domain_scopes)
    assert not domain_scope_matches("personal", project_query.domain_scopes)
    assert domain_scope_matches("system:rai", broad_query.domain_scopes)
    assert domain_scope_matches("project:aurora", conversation_query.domain_scopes)


def test_memory_sufficiency_requires_query_coverage_and_evidence_quality() -> None:
    query = MemoryQuery(
        keywords=("Aurora", "budżet", "termin"),
        raw_text="Jaki jest budżet i termin projektu Aurora?",
        domain_scopes=("global", "project:aurora"),
    )
    partial = MemoryRecord(
        producer=PRODUCER,
        source_turn_id="turn-partial",
        topic="claim.project.aurora.goal",
        domain_scope="project:aurora",
        content={
            "value": "Aurora",
            "source_span": "Projekt Aurora",
            "confidence": 1.0,
            "modality": "direct",
        },
    )
    complete_without_confidence = partial.model_copy(
        update={
            "record_id": "memory-no-confidence",
            "content": {
                "value": "Aurora, budżet i termin",
                "source_span": "Aurora ma budżet i termin",
                "modality": "direct",
            },
        }
    )
    complete = complete_without_confidence.model_copy(
        update={
            "record_id": "memory-complete",
            "content": {
                **complete_without_confidence.content,
                "confidence": 0.9,
            },
        }
    )
    contested = complete.model_copy(
        update={"record_id": "memory-contested", "epistemic_status": "contested"}
    )

    partial_assessment = assess_memory_sufficiency(query, (partial,))
    unknown_assessment = assess_memory_sufficiency(
        query, (complete_without_confidence,)
    )
    complete_assessment = assess_memory_sufficiency(query, (complete,))
    contested_assessment = assess_memory_sufficiency(query, (contested,))

    assert partial_assessment.query_coverage == pytest.approx(1 / 3)
    assert partial_assessment.score < MEMORY_SUFFICIENCY_THRESHOLD
    assert unknown_assessment.query_coverage == 1.0
    assert unknown_assessment.evidence_quality == 0.0
    assert unknown_assessment.score < MEMORY_SUFFICIENCY_THRESHOLD
    assert complete_assessment.query_coverage == 1.0
    assert complete_assessment.score >= MEMORY_SUFFICIENCY_THRESHOLD
    assert contested_assessment.evidence_quality == 0.0
    assert contested_assessment.score < MEMORY_SUFFICIENCY_THRESHOLD


@pytest.mark.asyncio
async def test_retrieval_isolated_by_domain_and_purpose(tmp_path: Path) -> None:
    text = "Pracuję nad projektem Aurora."
    span = text[:-1]
    topic = "claim.project.user.current.project"
    extractor = SchemaConstrainedMemoryExtractor(
        _StaticEngine(_extraction_json("turn-domain", span)),  # type: ignore[arg-type]
        model_name="schema-test-model",
    )
    store = SQLiteMemoryGraphStore(tmp_path / "domain.sqlite3")
    service = AssistantService(
        store=store,
        backend=DeterministicAssistantBackend(),
        memory_extractor=extractor,
    )
    await service.start()
    admitted = await service.accept_turn(_turn("turn-domain", text))
    assert isinstance(admitted, Success)

    project_query = MemoryQuery(
        topic=topic,
        keywords=("Aurora",),
        raw_text="Co pamiętasz o projekcie Aurora?",
        domain_scopes=("global", "project"),
    )
    personal_query = MemoryQuery(
        topic=topic,
        keywords=("Aurora",),
        raw_text="Co pamiętasz o projekcie Aurora?",
        domain_scopes=("global", "personal"),
    )
    wrong_purpose_query = MemoryQuery(
        topic=topic,
        keywords=("Aurora",),
        raw_text="Co pamiętasz o projekcie Aurora?",
        domain_scopes=("global", "project"),
        purpose="analytics",
    )

    project_memories = await store.retrieve_relevant_memories(query=project_query)
    personal_memories = await store.retrieve_relevant_memories(query=personal_query)
    wrong_purpose_memories = await store.retrieve_relevant_memories(
        query=wrong_purpose_query
    )
    project_turns = await store.retrieve_relevant_turns(
        profile_scope="default",
        query=project_query,
        data_classes=(DataClass.LOCAL,),
    )
    personal_turns = await store.retrieve_relevant_turns(
        profile_scope="default",
        query=personal_query,
        data_classes=(DataClass.LOCAL,),
    )

    assert isinstance(project_memories, Success)
    assert project_memories.unwrap()[0][0].domain_scope == "project:aurora"
    assert isinstance(personal_memories, Success)
    assert personal_memories.unwrap() == ()
    assert isinstance(wrong_purpose_memories, Success)
    assert wrong_purpose_memories.unwrap() == ()
    assert isinstance(project_turns, Success)
    assert project_turns.unwrap()[0][0].domain_scope == "project:aurora"
    assert isinstance(personal_turns, Success)
    assert personal_turns.unwrap() == ()
