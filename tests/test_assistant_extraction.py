"""Conformance tests for schema-constrained assistant memory extraction."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from returns.result import Failure, Result, Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.backends.local import LocalAssistantBackend
from rai.assistant.context import AssistantContextBuilder
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
from rai.assistant.query import MemoryQueryResolver
from rai.assistant.records import (
    ConversationTurn,
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
    operations = await store.list_memory_operations()
    assert isinstance(operations, Success)
    assert [operation.operation for operation in operations.unwrap()] == [
        "REFLECT",
        "REMEMBER",
    ]


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
    backend = LocalAssistantBackend(
        engine=_StaticEngine("Celem projektu Aurora jest lokalna pamięć asystenta."),
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
    assert all(item.answer_correct is True for item in measurements)
    assert all(item.answer_text is not None for item in measurements)
    assert all(item.backend_name == "evaluation-local-adapter" for item in measurements)
    assert all(item.judge_version for item in measurements)
    assert all(item.tokens_out is not None for item in measurements)
    assert all(item.answer_latency_ms is not None for item in measurements)
    assert all(item.energy_joules is None for item in measurements)

    backend.engine.text = "Nie mam wystarczających źródeł, aby odpowiedzieć."
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
    assert all(
        item.answer_correct is True and item.answer_abstained is True
        for item in abstention_run.unwrap().measurements
    )
    private_measurements = tuple(
        item
        for item in abstention_run.unwrap().measurements
        if item.case_id == "private-purpose-isolation"
    )
    assert all(not item.retrieved_ids for item in private_measurements)

    stopped = await backend.stop()
    assert isinstance(stopped, Success)
    await store.stop()


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
        domain_scopes=("general", "project"),
    )
    personal_query = MemoryQuery(
        topic=topic,
        keywords=("Aurora",),
        raw_text="Co pamiętasz o projekcie Aurora?",
        domain_scopes=("general", "personal"),
    )
    wrong_purpose_query = MemoryQuery(
        topic=topic,
        keywords=("Aurora",),
        raw_text="Co pamiętasz o projekcie Aurora?",
        domain_scopes=("general", "project"),
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
    assert project_memories.unwrap()[0][0].domain_scope == "project"
    assert isinstance(personal_memories, Success)
    assert personal_memories.unwrap() == ()
    assert isinstance(wrong_purpose_memories, Success)
    assert wrong_purpose_memories.unwrap() == ()
    assert isinstance(project_turns, Success)
    assert project_turns.unwrap()[0][0].domain_scope == "project"
    assert isinstance(personal_turns, Success)
    assert personal_turns.unwrap() == ()
