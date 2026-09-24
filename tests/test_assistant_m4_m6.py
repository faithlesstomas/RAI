"""Exit-evidence tests for Rich Assistant milestones M4 through M6."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest
from returns.result import Failure, Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.benchmark import run_retrieval_benchmark
from rai.assistant.context import AssistantContextBuilder
from rai.assistant.derived import VerifiedDerivedClaim, write_verified_derived_claim
from rai.assistant.energy import EnergyMeasurement, LinuxEnergyMeter
from rai.assistant.evaluation import (
    RetrievalEvaluationCase,
    evaluate_retrieval_floor,
    load_retrieval_evaluation_corpus,
    seed_retrieval_evaluation_corpus,
)
from rai.assistant.query import MemoryQueryResolver, domain_scope_matches
from rai.assistant.judging import judge_answer
from rai.assistant.records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    MemoryRecord,
    MemoryRelationKind,
)
from rai.assistant.retrieval import (
    MultiChannelMemoryRetriever,
    weighted_reciprocal_rank_fusion,
)
from rai.assistant.routing_evaluation import (
    ContextRoutingEvaluationCase,
    evaluate_context_routing,
)
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.records import ProducerIdentity

PRODUCER = ProducerIdentity(
    producer_id="assistant-m4-m6-test", kind="test", version="1.0.0"
)
EXPECTED_RAPL_SAMPLES = 2
RETRIEVAL_LIMIT = 2
ADVANCED_CHANNEL_COUNT = 3
ROUTING_AGGREGATE_COUNT = 6
BENCHMARK_TRIALS = 2
CORPUS_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "assistant"
    / "v1"
    / "retrieval-floor.corpus.json"
)


@pytest.mark.asyncio
async def test_linux_energy_meter_reads_rapl_delta(tmp_path: Path) -> None:
    package = tmp_path / "powercap" / "intel-rapl:0"
    package.mkdir(parents=True)
    energy = package / "energy_uj"
    maximum = package / "max_energy_range_uj"
    energy.write_text("100", encoding="ascii")
    maximum.write_text("10000", encoding="ascii")

    meter = LinuxEnergyMeter(
        powercap_root=tmp_path / "powercap",
        hwmon_root=tmp_path / "hwmon",
    )
    started = await meter.start()
    assert isinstance(started, Success)
    energy.write_text("2100", encoding="ascii")
    finished = await started.unwrap().finish()

    assert isinstance(finished, Success)
    assert finished.unwrap().joules == pytest.approx(0.002)
    assert finished.unwrap().source == "linux-rapl"
    assert finished.unwrap().scope == "cpu-package"
    assert finished.unwrap().sample_count == EXPECTED_RAPL_SAMPLES


def test_weighted_rrf_is_rank_based_and_deterministic() -> None:
    fused = weighted_reciprocal_rank_fusion(
        (
            ("lexical", ("a", "b"), 1.0),
            ("dense", ("b", "c"), 2.0),
        ),
        rank_constant=10,
    )

    assert tuple(identifier for identifier, _score in fused) == ("b", "c", "a")
    assert dict(fused)["b"] == pytest.approx(1 / 12 + 2 / 11)


@pytest.mark.asyncio
async def test_graph_retrieval_uses_only_authenticated_eligible_edges(
    tmp_path: Path,
) -> None:
    corpus = load_retrieval_evaluation_corpus(CORPUS_PATH)
    store = SQLiteMemoryGraphStore(tmp_path / "graph.sqlite3")
    assert isinstance(await store.start(), Success)
    assert isinstance(await seed_retrieval_evaluation_corpus(store, corpus), Success)
    retriever = MultiChannelMemoryRetriever(store, graph_depth=2)

    result = await retriever.retrieve(
        MemoryQueryResolver.resolve("Jaki jest cel projektu Aurora?"),
        data_classes=corpus.cases[0].data_classes,
        limit=RETRIEVAL_LIMIT,
    )

    assert isinstance(result, Success)
    selection = result.unwrap()
    relation_ids = {
        relation_id
        for path in selection.graph_paths
        for relation_id in path.relation_ids
    }
    assert "corpus-relation-project-supports-commitment" in relation_ids
    assert "corpus-relation-poisoned-demo" not in relation_ids
    assert set(dict(selection.channel_ids)) == {
        "claims_bm25",
        "dense_local",
        "rrf_fused",
        "graph_bounded",
    }
    assert all(
        len(ids) <= RETRIEVAL_LIMIT for ids in dict(selection.channel_ids).values()
    )

    query_turn = ConversationTurn(
        record_id="graph-manifest-query",
        producer=PRODUCER,
        session_id="graph-manifest-session",
        role="user",
        text="Jaki jest cel projektu Aurora?",
        metadata={"profile_scope": "default"},
    )
    await store.accept_turn(query_turn)
    context = await AssistantContextBuilder(
        store,
        advanced_retriever=retriever,
        max_memories=RETRIEVAL_LIMIT,
    ).build_context(query_turn)
    assert isinstance(context, Success)
    manifest = context.unwrap().manifest
    assert set(manifest.retrieval_channel_ids) == {
        "claims_bm25",
        "dense_local",
        "rrf_fused",
        "graph_bounded",
    }
    assert any(
        "corpus-relation-project-supports-commitment" in path["relation_ids"]
        for path in manifest.graph_paths
    )
    await store.stop()


@pytest.mark.asyncio
async def test_advanced_channels_share_retrieval_and_context_budgets(
    tmp_path: Path,
) -> None:
    corpus = load_retrieval_evaluation_corpus(CORPUS_PATH)
    store = SQLiteMemoryGraphStore(tmp_path / "equal-budget.sqlite3")
    await store.start()
    await seed_retrieval_evaluation_corpus(store, corpus)
    case = next(case for case in corpus.cases if case.case_id == "project-goal")

    result = await evaluate_retrieval_floor(
        store,
        (case,),
        retrieval_limit=RETRIEVAL_LIMIT,
        context_character_budget=500,
        advanced_retriever=MultiChannelMemoryRetriever(store),
    )

    assert isinstance(result, Success)
    run = result.unwrap()
    advanced = tuple(
        item
        for item in run.measurements
        if item.channel in {"dense_local", "rrf_fused", "graph_bounded"}
    )
    assert len(advanced) == ADVANCED_CHANNEL_COUNT
    assert all(len(item.retrieved_ids) <= run.retrieval_limit for item in advanced)
    assert all(
        item.context_characters <= run.context_character_budget for item in advanced
    )
    graph = next(item for item in advanced if item.channel == "graph_bounded")
    assert "corpus-relation-project-supports-commitment" in graph.artifact_ids
    await store.stop()


@pytest.mark.asyncio
async def test_router_compares_short_medium_and_long_histories(tmp_path: Path) -> None:
    corpus = load_retrieval_evaluation_corpus(CORPUS_PATH)
    store = SQLiteMemoryGraphStore(tmp_path / "routing.sqlite3")
    await store.start()
    await seed_retrieval_evaluation_corpus(store, corpus)
    cases = tuple(
        ContextRoutingEvaluationCase(
            case_id=case.case_id,
            horizon=case.history_horizon,
            query_turn=ConversationTurn(
                record_id=f"routing-query-{case.case_id}",
                producer=PRODUCER,
                session_id=case.routing_session_id,
                role="user",
                text=case.query_text,
                metadata={"profile_scope": "default"},
            ),
            relevant_source_ids=(
                case.relevant_raw_turn_ids
                if case.history_horizon == "short"
                else case.relevant_claim_ids
                if case.history_horizon == "long"
                else tuple(
                    dict.fromkeys(
                        (*case.relevant_raw_turn_ids, *case.relevant_claim_ids)
                    )
                )
            ),
        )
        for case in corpus.cases
        if case.history_horizon is not None and case.routing_session_id is not None
    )

    result = await evaluate_context_routing(
        store,
        cases,
        context_character_budget=4096,
    )

    assert isinstance(result, Success)
    measurements = result.unwrap().measurements
    assert {item.horizon for item in measurements} == {"short", "medium", "long"}
    assert {item.strategy for item in measurements} == {"adaptive", "always_memory"}
    assert all(
        aggregate.answer_accuracy is None for aggregate in result.unwrap().aggregates
    )
    case_ids = {item.case_id for item in measurements}
    for case_id in case_ids:
        adaptive = next(
            item
            for item in measurements
            if item.case_id == case_id and item.strategy == "adaptive"
        )
        baseline = next(
            item
            for item in measurements
            if item.case_id == case_id and item.strategy == "always_memory"
        )
        assert adaptive.source_recall >= baseline.source_recall
        assert adaptive.context_characters <= adaptive.context_character_budget
    short_adaptive = tuple(
        item
        for item in measurements
        if item.horizon == "short" and item.strategy == "adaptive"
    )
    assert len(short_adaptive) == BENCHMARK_TRIALS
    assert all(
        item.routing_decision == "recent_conversation" for item in short_adaptive
    )
    await store.stop()


@pytest.mark.asyncio
async def test_verified_writeback_is_source_covered_and_scope_preserving(
    tmp_path: Path,
) -> None:
    corpus = load_retrieval_evaluation_corpus(CORPUS_PATH)
    store = SQLiteMemoryGraphStore(tmp_path / "derived.sqlite3")
    await store.start()
    await seed_retrieval_evaluation_corpus(store, corpus)
    finding = VerifiedDerivedClaim(
        verification_id="verify-aurora-order",
        verifier_id="deterministic-evidence-verifier",
        verifier_version="1.0.0",
        policy_version="derived-writeback-v1",
        topic="derived.project.aurora.retrieval_plan",
        content={"finding": "BM25 precedes embedding evaluation"},
        source_memory_ids=(
            "corpus-memory-project",
            "corpus-memory-commitment",
        ),
        domain_scope="project:aurora",
        confidence=0.9,
    )

    written = await write_verified_derived_claim(store, finding)

    assert isinstance(written, Success)
    memory = written.unwrap()
    assert memory.kind == "derived_claim"
    assert memory.epistemic_status == "inferred"
    assert memory.content["source_coverage"] == 1.0
    assert memory.content["source_memory_ids"] == finding.source_memory_ids
    assert {item.source_id for item in memory.provenance} == set(
        finding.source_memory_ids
    )
    relations = await store.get_relations(
        target_id=memory.record_id,
        kind=MemoryRelationKind.SUPPORTS,
    )
    assert isinstance(relations, Success)
    assert {relation.source_id for relation in relations.unwrap()} == set(
        finding.source_memory_ids
    )
    assert all(relation.provenance for relation in relations.unwrap())

    cross_domain = finding.model_copy(
        update={
            "verification_id": "verify-cross-domain",
            "source_memory_ids": (
                "corpus-memory-project",
                "corpus-memory-name",
            ),
        }
    )
    rejected = await write_verified_derived_claim(store, cross_domain)
    assert isinstance(rejected, Failure)
    assert rejected.failure().code == "DERIVED_DOMAIN_MISMATCH"

    deleted = await store.delete_turn("corpus-turn-commitment")
    assert isinstance(deleted, Success)
    after_delete = await store.get_memory(memory.record_id)
    assert isinstance(after_delete, Success)
    assert after_delete.unwrap() is None
    await store.stop()


@pytest.mark.asyncio
async def test_derived_writeback_rejects_unknown_source_confidence(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "unknown-confidence.sqlite3")
    await store.start()
    turn = ConversationTurn(
        record_id="unknown-confidence-turn",
        producer=PRODUCER,
        session_id="unknown-confidence-session",
        role="user",
        text="Jawnie podana informacja bez oceny pewności.",
        status="COMPLETED",
    )
    assert isinstance(await store.accept_turn(turn), Success)
    memory = MemoryRecord(
        record_id="unknown-confidence-memory",
        producer=PRODUCER,
        kind="claim",
        topic="claim.test.unknown-confidence",
        content={"fact": "test"},
        source_turn_id=turn.record_id,
    )
    manifest = AssistantContextManifest(
        record_id="unknown-confidence-manifest",
        producer=PRODUCER,
        session_id=turn.session_id,
        turn_id=turn.record_id,
    )
    response = AssistantResponse(
        record_id="unknown-confidence-response",
        producer=PRODUCER,
        session_id=turn.session_id,
        turn_id="unknown-confidence-assistant",
        user_turn_id=turn.record_id,
        request_id="unknown-confidence-request",
        manifest_id=manifest.record_id,
        text="stored",
    )
    assert isinstance(
        await store.commit_terminal(response, manifest, None, (memory,), ()), Success
    )

    written = await write_verified_derived_claim(
        store,
        VerifiedDerivedClaim(
            verification_id="unknown-confidence-verification",
            verifier_id="test-verifier",
            verifier_version="1.0.0",
            policy_version="derived-writeback-v1",
            topic="derived.test.unknown-confidence",
            content={"finding": "unsafe without source confidence"},
            source_memory_ids=(memory.record_id,),
        ),
    )

    assert isinstance(written, Failure)
    assert written.failure().code == "DERIVED_SOURCE_CONFIDENCE_UNKNOWN"


def test_answer_judge_handles_polish_inflection_abstention_and_plan_modality() -> None:
    correct, abstained = judge_answer(
        "Celem jest zbudowanie lokalnej pamięci asystenta w Warszawie.",
        expected_phrases=("lokalna pamięć asystenta", "Warszawa"),
        forbidden_phrases=(),
        expected_abstention=False,
    )
    assert correct and not abstained

    correct, abstained = judge_answer(
        "Nie mam informacji o ulubionym filmie.",
        expected_phrases=(),
        forbidden_phrases=(),
        expected_abstention=True,
    )
    assert correct and abstained

    correct, abstained = judge_answer(
        "Nie ma informacji o instrumencie; brak jest danych w źródłach.",
        expected_phrases=(),
        forbidden_phrases=(),
        expected_abstention=True,
    )
    assert correct and abstained

    correct, _ = judge_answer(
        "Zespół zmierzył baseline BM25.",
        expected_phrases=("BM25",),
        forbidden_phrases=("zmierzył baseline",),
        expected_abstention=False,
    )
    assert not correct


@pytest.mark.asyncio
async def test_memory_intent_without_evidence_abstains_before_generation(
    tmp_path: Path,
) -> None:
    service = AssistantService(
        SQLiteMemoryGraphStore(tmp_path / "abstention.sqlite3"),
        DeterministicAssistantBackend(),
    )
    await service.start()
    turn = ConversationTurn(
        record_id="missing-memory-question",
        producer=PRODUCER,
        session_id="missing-memory-session",
        role="user",
        text="Co pamiętasz o moim ulubionym filmie?",
    )

    response = await service.accept_turn(turn)

    assert isinstance(response, Success)
    assert "Nie mam wystarczających" in response.unwrap().text
    manifest = await service.store.get_manifest(response.unwrap().manifest_id)
    assert isinstance(manifest, Success)
    assert manifest.unwrap() is not None
    assert manifest.unwrap().evidence_required is True
    assert manifest.unwrap().routing_decision == "no_evidence"
    await service.stop()


def test_unknown_legacy_scope_fails_closed_for_restrictive_queries() -> None:
    scopes = MemoryQueryResolver.resolve("Jaki jest cel projektu Aurora?").domain_scopes
    assert domain_scope_matches("global", scopes)
    assert not domain_scope_matches("unknown", scopes)
    assert not domain_scope_matches("general", scopes)


@pytest.mark.asyncio
async def test_legacy_general_rows_migrate_to_fail_closed_unknown(
    tmp_path: Path,
) -> None:
    database = tmp_path / "scope-migration.sqlite3"
    corpus = load_retrieval_evaluation_corpus(CORPUS_PATH)
    store = SQLiteMemoryGraphStore(database)
    await store.start()
    await seed_retrieval_evaluation_corpus(store, corpus)
    await store.stop()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE turns SET domain_scope = 'general' "
            "WHERE turn_id = 'corpus-turn-project'"
        )
        connection.execute(
            "UPDATE memories SET domain_scope = 'general' "
            "WHERE memory_id = 'corpus-memory-project'"
        )
        connection.execute(
            "DELETE FROM assistant_schema_metadata WHERE key = 'domain_scope_semantics'"
        )

    migrated = SQLiteMemoryGraphStore(database)
    await migrated.start()
    turn = await migrated.get_turn("corpus-turn-project")
    memory = await migrated.get_memory("corpus-memory-project")

    assert isinstance(turn, Success) and turn.unwrap() is not None
    assert turn.unwrap().domain_scope == "unknown"
    assert isinstance(memory, Success) and memory.unwrap() is not None
    assert memory.unwrap()[0].domain_scope == "unknown"
    await migrated.stop()


@pytest.mark.asyncio
async def test_benchmark_writes_six_channels_and_routing_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark.json"
    result = await run_retrieval_benchmark(
        DeterministicAssistantBackend(),
        corpus_path=CORPUS_PATH,
        output_path=output,
        retrieval_limit=RETRIEVAL_LIMIT,
        context_character_budget=4096,
        max_latency_seconds=5.0,
        trials=BENCHMARK_TRIALS,
    )

    assert isinstance(result, Success)
    artifact = result.unwrap()
    assert output.exists()
    assert artifact.artifact_version == "rai-assistant-retrieval-benchmark-v2"
    assert artifact.judge_version == "deterministic-phrase-and-abstention-v4"
    assert artifact.model_artifact_version == "1.0.0"
    assert artifact.run.trials == BENCHMARK_TRIALS
    assert {item.trial_index for item in artifact.run.measurements} == {1, 2}
    assert {aggregate.channel for aggregate in artifact.aggregates} == {
        "raw_turns_bm25",
        "claims_bm25",
        "summaries_grounded",
        "dense_local",
        "rrf_fused",
        "graph_bounded",
    }
    assert artifact.routing_run is not None
    assert artifact.routing_run.trials == BENCHMARK_TRIALS
    assert {item.trial_index for item in artifact.routing_run.measurements} == {1, 2}
    assert {item.horizon for item in artifact.routing_run.measurements} == {
        "short",
        "medium",
        "long",
    }
    assert all(
        item.answer_correct is not None for item in artifact.routing_run.measurements
    )
    assert len(artifact.routing_run.aggregates) == ROUTING_AGGREGATE_COUNT


@pytest.mark.asyncio
async def test_benchmark_can_require_complete_energy_measurement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist.json"
    result = await run_retrieval_benchmark(
        DeterministicAssistantBackend(),
        corpus_path=CORPUS_PATH,
        output_path=output,
        energy_meter=None,
        require_energy=True,
        retrieval_limit=1,
        context_character_budget=2048,
        max_latency_seconds=5.0,
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "ENERGY_MEASUREMENT_INCOMPLETE"
    assert not output.exists()


@pytest.mark.asyncio
async def test_benchmark_requires_model_artifact_identity(tmp_path: Path) -> None:
    output = tmp_path / "unidentified-model.json"
    backend = DeterministicAssistantBackend()
    backend.model_artifact_version = None  # type: ignore[assignment]

    result = await run_retrieval_benchmark(
        backend,
        corpus_path=CORPUS_PATH,
        output_path=output,
        max_latency_seconds=5.0,
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "MODEL_ARTIFACT_VERSION_UNAVAILABLE"
    assert not output.exists()
