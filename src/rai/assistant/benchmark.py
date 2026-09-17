"""Runnable retrieval benchmark orchestration with isolated storage."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, ProducerIdentity

from .energy import EnergyMeter
from .evaluation import (
    BackendRetrievalAnswerEvaluator,
    RetrievalBenchmarkArtifact,
    RetrievalEvaluationRun,
    aggregate_retrieval_run,
    evaluate_retrieval_floor,
    load_retrieval_evaluation_corpus,
    seed_retrieval_evaluation_corpus,
)
from .ports import AssistantModelBackend
from .records import ConversationTurn, make_assistant_failure
from .retrieval import MultiChannelMemoryRetriever
from .routing_evaluation import (
    ContextRoutingEvaluationCase,
    ContextRoutingEvaluationRun,
    aggregate_context_routing,
    evaluate_context_routing,
)
from .store import SQLiteMemoryGraphStore
from .summary import GroundedClaimSummaryProvider

_BENCHMARK_PRODUCER = ProducerIdentity(
    producer_id="assistant-routing-benchmark",
    kind="evaluation",
    version="1.0.0",
)


async def run_retrieval_benchmark(  # noqa: PLR0911, PLR0912, PLR0913
    backend: AssistantModelBackend,
    *,
    corpus_path: Path,
    output_path: Path,
    energy_meter: EnergyMeter | None = None,
    require_energy: bool = False,
    retrieval_limit: int = 5,
    context_character_budget: int = 8_000,
    max_input_tokens: int = 4096,
    max_output_tokens: int = 256,
    max_latency_seconds: float = 60.0,
    trials: int = 1,
) -> Result[RetrievalBenchmarkArtifact, ActionFailure]:
    """Run every retrieval channel against one frozen corpus and write JSON atomically."""
    try:
        corpus = load_retrieval_evaluation_corpus(corpus_path)
    except (OSError, ValueError) as exc:
        return Failure(
            make_assistant_failure(
                code="INVALID_RETRIEVAL_CORPUS",
                message=str(exc),
                request_id=str(corpus_path),
            )
        )

    with TemporaryDirectory(prefix="rai-retrieval-benchmark-") as directory:
        store = SQLiteMemoryGraphStore(Path(directory) / "memory.sqlite3")
        store_started = await store.start()
        if isinstance(store_started, Failure):
            return Failure(store_started.failure())
        backend_started = await backend.start()
        if isinstance(backend_started, Failure):
            await store.stop()
            return Failure(backend_started.failure())
        try:
            if not getattr(backend, "model_artifact_version", None):
                return Failure(
                    make_assistant_failure(
                        code="MODEL_ARTIFACT_VERSION_UNAVAILABLE",
                        message=(
                            "benchmark requires an inspectable model artifact version "
                            "or digest"
                        ),
                        request_id=str(getattr(backend, "model_name", "unknown")),
                    )
                )
            seeded = await seed_retrieval_evaluation_corpus(store, corpus)
            if isinstance(seeded, Failure):
                return Failure(seeded.failure())
            retriever = MultiChannelMemoryRetriever(store)
            evaluator = BackendRetrievalAnswerEvaluator(
                backend,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_latency_seconds=max_latency_seconds,
                energy_meter=energy_meter,
            )
            if trials < 1:
                return Failure(
                    make_assistant_failure(
                        code="INVALID_BENCHMARK_TRIALS",
                        message="benchmark trials must be at least 1",
                        request_id=corpus.corpus_version,
                    )
                )
            retrieval_measurements = []
            for trial_index in range(1, trials + 1):
                result = await evaluate_retrieval_floor(
                    store,
                    corpus.cases,
                    retrieval_limit=retrieval_limit,
                    context_character_budget=context_character_budget,
                    summary_provider=GroundedClaimSummaryProvider(store),
                    advanced_retriever=retriever,
                    answer_evaluator=evaluator,
                    corpus_version=corpus.corpus_version,
                )
                if isinstance(result, Failure):
                    return Failure(result.failure())
                retrieval_measurements.extend(
                    measurement.model_copy(update={"trial_index": trial_index})
                    for measurement in result.unwrap().measurements
                )
            run = RetrievalEvaluationRun(
                corpus_version=corpus.corpus_version,
                profile_scope="default",
                retrieval_limit=retrieval_limit,
                context_character_budget=context_character_budget,
                trials=trials,
                measurements=tuple(retrieval_measurements),
            )
            aggregates = aggregate_retrieval_run(run)
            if require_energy and any(
                aggregate.energy_measurement_coverage < 1.0
                for aggregate in aggregates
            ):
                return Failure(
                    make_assistant_failure(
                        code="ENERGY_MEASUREMENT_INCOMPLETE",
                        message="at least one benchmark channel lacks measured energy",
                        request_id=corpus.corpus_version,
                    )
                )
            routing_cases = tuple(
                ContextRoutingEvaluationCase(
                    case_id=case.case_id,
                    horizon=case.history_horizon,
                    query_turn=ConversationTurn(
                        record_id=f"routing-query-{case.case_id}",
                        producer=_BENCHMARK_PRODUCER,
                        session_id=case.routing_session_id,
                        role="user",
                        text=case.query_text,
                        metadata={"profile_scope": "default", "benchmark": True},
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
                    expected_answer_phrases=case.expected_answer_phrases,
                    forbidden_answer_phrases=case.forbidden_answer_phrases,
                    expected_abstention=case.expected_abstention,
                )
                for case in corpus.cases
                if case.history_horizon is not None
                and case.routing_session_id is not None
                and (case.relevant_raw_turn_ids or case.relevant_claim_ids)
            )
            routing_run = None
            if routing_cases:
                routing_measurements = []
                for trial_index in range(1, trials + 1):
                    routing_result = await evaluate_context_routing(
                        store,
                        routing_cases,
                        context_character_budget=context_character_budget,
                        max_memories=retrieval_limit,
                        max_episodic_turns=retrieval_limit,
                        answer_backend=backend,
                        max_input_tokens=max_input_tokens,
                        max_output_tokens=max_output_tokens,
                        max_latency_seconds=max_latency_seconds,
                    )
                    if isinstance(routing_result, Failure):
                        return Failure(routing_result.failure())
                    routing_measurements.extend(
                        measurement.model_copy(update={"trial_index": trial_index})
                        for measurement in routing_result.unwrap().measurements
                    )
                finalized_routing = tuple(routing_measurements)
                routing_run = ContextRoutingEvaluationRun(
                    context_character_budget=context_character_budget,
                    adaptive_threshold=0.75,
                    trials=trials,
                    measurements=finalized_routing,
                    aggregates=aggregate_context_routing(finalized_routing),
                )
            artifact = RetrievalBenchmarkArtifact(
                run=run,
                aggregates=aggregates,
                backend_name=str(getattr(backend, "backend_name", "unknown")),
                model_name=str(getattr(backend, "model_name", "unknown")),
                model_artifact_version=getattr(backend, "model_artifact_version", None),
                prompt_template_version=str(
                    getattr(backend, "prompt_template_version", "unknown")
                ),
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_latency_seconds=max_latency_seconds,
                dense_embedding_version=retriever.embedder.version,
                rrf_weights={
                    "claims_bm25": retriever.lexical_weight,
                    "dense_local": retriever.dense_weight,
                    "graph_bounded": retriever.graph_weight,
                },
                graph_depth=retriever.graph_depth,
                minimum_relation_confidence=retriever.minimum_relation_confidence,
                energy_required=require_energy,
                routing_run=routing_run,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
            temporary.write_text(
                artifact.model_dump_json(indent=2), encoding="utf-8"
            )
            temporary.replace(output_path)
            return Success(artifact)
        finally:
            await backend.stop()
            await store.stop()
