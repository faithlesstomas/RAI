"""Long-context needle-in-a-haystack and context-rot A/B benchmark.

Compares:
- Strategy A (raw_context): Naive long chat window with planted needle at start/middle/end.
- Strategy B (graph_memory): RAI selective memory architecture with compact claim evidence.

Measures attention degradation (Lost-in-the-Middle), hallucination on absent facts,
TTFT / generation latency, and token efficiency as context saturates up to 8k+ tokens.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import json
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
    make_assistant_failure,
)

NeedleDepth = Literal["start", "middle", "end"]
EvaluationStrategy = Literal["raw_context", "graph_memory"]

_ROT_EVALUATION_PRODUCER = ProducerIdentity(
    producer_id="assistant-context-rot-evaluator",
    kind="evaluation",
    version="1.0.0",
)


class ContextRotScenario(BaseModel):
    """A planted fact scenario with query, claim, and absent question."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    description: str
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
    """A realistic desktop/system dialog exchange to inflate context."""

    model_config = ConfigDict(frozen=True)

    distractor_id: str
    user_text: str
    assistant_text: str


class ContextRotCorpus(BaseModel):
    """Complete versioned corpus for context rot testing."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    description: str
    scenarios: tuple[ContextRotScenario, ...]
    distractors: tuple[DistractorItem, ...]


def load_context_rot_corpus(path: Path) -> ContextRotCorpus:
    """Load and validate the context rot corpus from a JSON fixture."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ContextRotCorpus.model_validate(data)


def estimate_tokens(text: str) -> int:
    """Fast conservative heuristic for token length in multilingual code/text."""
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
    """Synthesize conversation turns reaching target_tokens with needle placed at depth."""
    needle_pair = [
        {"role": "user", "text": needle_user},
        {"role": "assistant", "text": needle_assistant},
    ]
    needle_tokens = estimate_tokens(needle_user) + estimate_tokens(needle_assistant)
    remaining_tokens = max(0, target_tokens - needle_tokens)

    pairs: list[list[dict[str, str]]] = []
    accumulated_tokens = 0
    distractor_idx = 0
    total_distractors = len(distractors)

    while accumulated_tokens < remaining_tokens and total_distractors > 0:
        item = distractors[distractor_idx % total_distractors]
        pair = [
            {"role": "user", "text": item.user_text},
            {"role": "assistant", "text": item.assistant_text},
        ]
        pair_tokens = estimate_tokens(item.user_text) + estimate_tokens(item.assistant_text)
        pairs.append(pair)
        accumulated_tokens += pair_tokens
        distractor_idx += 1

    num_pairs = len(pairs)
    if num_pairs == 0:
        insert_pos = 0
    elif depth == "start":
        insert_pos = max(0, int(num_pairs * 0.10))
    elif depth == "middle":
        insert_pos = int(num_pairs * 0.50)
    else:  # "end"
        insert_pos = min(num_pairs, max(0, int(num_pairs * 0.90)))

    pairs.insert(insert_pos, needle_pair)

    # Flatten pairs into flat turns
    haystack_turns: list[dict[str, str]] = []
    for pair in pairs:
        haystack_turns.extend(pair)

    return haystack_turns


class ContextRotCaseMeasurement(BaseModel):
    """Result of one inference trial under a specific strategy and saturation."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    strategy: EvaluationStrategy
    target_tokens: int
    depth: NeedleDepth
    trial_index: int
    query_type: Literal["needle", "absent"]
    query_text: str
    delivered_text: str
    raw_model_output: str
    answer_correct: bool
    answer_abstained: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    context_turns_count: int


class ContextRotAggregate(BaseModel):
    """Aggregated metrics across trials for a strategy, token step, and depth."""

    model_config = ConfigDict(frozen=True)

    strategy: EvaluationStrategy
    target_tokens: int
    depth: NeedleDepth | None = None
    case_count: int
    needle_recall_accuracy: float
    absent_hallucination_rate: float
    mean_latency_ms: float
    mean_tokens_in: float
    mean_tokens_out: float


class ContextRotEvaluationReport(BaseModel):
    """Complete artifact reporting context rot benchmark execution and aggregates."""

    model_config = ConfigDict(frozen=True)

    artifact_version: str = "rai-assistant-context-rot-v1"
    generated_at: str
    backend_name: str
    model_name: str
    token_steps: tuple[int, ...]
    depths: tuple[NeedleDepth, ...]
    total_cases: int
    measurements: tuple[ContextRotCaseMeasurement, ...]
    aggregates: tuple[ContextRotAggregate, ...]


def _aggregate_measurements(
    measurements: Sequence[ContextRotCaseMeasurement],
) -> list[ContextRotAggregate]:
    """Calculate summary aggregates per strategy, token step, and depth."""
    aggregates: list[ContextRotAggregate] = []

    strategies: list[EvaluationStrategy] = ["raw_context", "graph_memory"]
    token_steps = sorted(list({m.target_tokens for m in measurements}))
    depths: list[NeedleDepth] = ["start", "middle", "end"]

    for strat in strategies:
        for tokens in token_steps:
            # Per-depth aggregates
            for depth in depths:
                group = [
                    m
                    for m in measurements
                    if m.strategy == strat
                    and m.target_tokens == tokens
                    and m.depth == depth
                ]
                if not group:
                    continue
                needle_cases = [m for m in group if m.query_type == "needle"]
                absent_cases = [m for m in group if m.query_type == "absent"]

                recall_acc = (
                    sum(1 for m in needle_cases if m.answer_correct) / len(needle_cases)
                    if needle_cases
                    else 0.0
                )
                hallucination_rate = (
                    sum(1 for m in absent_cases if not m.answer_abstained)
                    / len(absent_cases)
                    if absent_cases
                    else 0.0
                )
                mean_lat = sum(m.latency_ms for m in group) / len(group)
                mean_tin = sum(m.tokens_in for m in group) / len(group)
                mean_tout = sum(m.tokens_out for m in group) / len(group)

                aggregates.append(
                    ContextRotAggregate(
                        strategy=strat,
                        target_tokens=tokens,
                        depth=depth,
                        case_count=len(group),
                        needle_recall_accuracy=recall_acc,
                        absent_hallucination_rate=hallucination_rate,
                        mean_latency_ms=mean_lat,
                        mean_tokens_in=mean_tin,
                        mean_tokens_out=mean_tout,
                    )
                )

            # Strategy-wide overall aggregate for this token step (depth=None)
            step_group = [
                m
                for m in measurements
                if m.strategy == strat and m.target_tokens == tokens
            ]
            if step_group:
                needle_cases = [m for m in step_group if m.query_type == "needle"]
                absent_cases = [m for m in step_group if m.query_type == "absent"]
                recall_acc = (
                    sum(1 for m in needle_cases if m.answer_correct) / len(needle_cases)
                    if needle_cases
                    else 0.0
                )
                hallucination_rate = (
                    sum(1 for m in absent_cases if not m.answer_abstained)
                    / len(absent_cases)
                    if absent_cases
                    else 0.0
                )
                mean_lat = sum(m.latency_ms for m in step_group) / len(step_group)
                mean_tin = sum(m.tokens_in for m in step_group) / len(step_group)
                mean_tout = sum(m.tokens_out for m in step_group) / len(step_group)

                aggregates.append(
                    ContextRotAggregate(
                        strategy=strat,
                        target_tokens=tokens,
                        depth=None,
                        case_count=len(step_group),
                        needle_recall_accuracy=recall_acc,
                        absent_hallucination_rate=hallucination_rate,
                        mean_latency_ms=mean_lat,
                        mean_tokens_in=mean_tin,
                        mean_tokens_out=mean_tout,
                    )
                )

    return aggregates


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
    max_output_tokens: int,
    max_latency_seconds: float,
) -> tuple[str, str, bool, bool, float, int, int]:
    """Execute inference and return (delivered, raw, correct, abstained, latency, tokens_in, tokens_out)."""
    now = _utc_now()
    budget = InferenceBudget(
        producer=_ROT_EVALUATION_PRODUCER,
        max_input_tokens=16384,
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

    total_chars = sum(len(t.get("text", "")) for t in recent_turns) + len(query_text)
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

    cancellation = CancellationToken()
    start_time = time.monotonic()
    result = await backend.generate(request, cancellation)
    latency_ms = (time.monotonic() - start_time) * 1000.0

    if isinstance(result, Failure):
        return ("", "", False, False, latency_ms, 0, 0)

    candidate = result.unwrap()
    delivered = candidate.text
    raw_output = candidate.metadata.get("raw_model_output")
    backend_name = str(getattr(backend, "backend_name", "unknown"))
    if (
        not isinstance(raw_output, str)
        or not raw_output.strip()
        or candidate.metadata.get("grounding_override") is True
        or backend_name == "deterministic"
    ):
        raw_output = delivered

    correct, abstained = judge_answer(
        raw_output,
        expected_phrases=expected_phrases,
        forbidden_phrases=forbidden_phrases,
        expected_abstention=expected_abstention,
    )

    tokens_in = candidate.tokens_in
    if recent_turns and tokens_in <= len(query_text.split()):
        context_tokens = sum(estimate_tokens(t.get("text", "")) for t in recent_turns)
        tokens_in = context_tokens + estimate_tokens(query_text)

    return (
        delivered,
        raw_output,
        correct,
        abstained,
        latency_ms,
        tokens_in,
        candidate.tokens_out,
    )


async def evaluate_context_rot(  # noqa: PLR0913
    backend: AssistantModelBackend,
    *,
    corpus: ContextRotCorpus,
    token_steps: Sequence[int] = (1024, 2048, 4096, 7500),
    depths: Sequence[NeedleDepth] = ("start", "middle", "end"),
    trials: int = 1,
    max_output_tokens: int = 128,
    max_latency_seconds: float = 60.0,
) -> Result[ContextRotEvaluationReport, ActionFailure]:
    """Execute the Context Rot A/B benchmark matrix."""
    backend_res = await backend.start()
    if isinstance(backend_res, Failure):
        return Failure(backend_res.failure())

    backend_name = str(getattr(backend, "backend_name", "unknown"))
    model_name = str(getattr(backend, "model_name", "unknown"))
    measurements: list[ContextRotCaseMeasurement] = []

    try:
        for token_step in token_steps:
            for depth in depths:
                for scenario in corpus.scenarios:
                    # Synthesize haystack with needle at specified depth
                    haystack_turns = generate_haystack(
                        distractors=corpus.distractors,
                        needle_user=scenario.needle_user,
                        needle_assistant=scenario.needle_assistant,
                        target_tokens=token_step,
                        depth=depth,
                    )

                    for trial in range(1, trials + 1):
                        # --- Strategy A: raw_context ---
                        # Needle query
                        sess_a = f"sess-raw-{scenario.scenario_id}-{token_step}-{depth}-{trial}"
                        (
                            del_a,
                            raw_a,
                            corr_a,
                            abst_a,
                            lat_a,
                            tin_a,
                            tout_a,
                        ) = await _run_single_inference(
                            backend=backend,
                            session_id=sess_a,
                            turn_id=f"turn-needle-{_new_id()}",
                            query_text=scenario.query_text,
                            recent_turns=haystack_turns,
                            durable_memories=[],
                            expected_phrases=scenario.expected_phrases,
                            forbidden_phrases=scenario.forbidden_phrases,
                            expected_abstention=False,
                            max_output_tokens=max_output_tokens,
                            max_latency_seconds=max_latency_seconds,
                        )
                        measurements.append(
                            ContextRotCaseMeasurement(
                                scenario_id=scenario.scenario_id,
                                strategy="raw_context",
                                target_tokens=token_step,
                                depth=depth,
                                trial_index=trial,
                                query_type="needle",
                                query_text=scenario.query_text,
                                delivered_text=del_a,
                                raw_model_output=raw_a,
                                answer_correct=corr_a,
                                answer_abstained=abst_a,
                                latency_ms=lat_a,
                                tokens_in=tin_a,
                                tokens_out=tout_a,
                                context_turns_count=len(haystack_turns),
                            )
                        )

                        # Absent query (Hallucination test)
                        if scenario.absent_query_text:
                            (
                                del_a_abs,
                                raw_a_abs,
                                corr_a_abs,
                                abst_a_abs,
                                lat_a_abs,
                                tin_a_abs,
                                tout_a_abs,
                            ) = await _run_single_inference(
                                backend=backend,
                                session_id=sess_a,
                                turn_id=f"turn-absent-{_new_id()}",
                                query_text=scenario.absent_query_text,
                                recent_turns=haystack_turns,
                                durable_memories=[],
                                expected_phrases=scenario.absent_expected_phrases,
                                forbidden_phrases=scenario.absent_forbidden_phrases,
                                expected_abstention=True,
                                max_output_tokens=max_output_tokens,
                                max_latency_seconds=max_latency_seconds,
                            )
                            measurements.append(
                                ContextRotCaseMeasurement(
                                    scenario_id=scenario.scenario_id,
                                    strategy="raw_context",
                                    target_tokens=token_step,
                                    depth=depth,
                                    trial_index=trial,
                                    query_type="absent",
                                    query_text=scenario.absent_query_text,
                                    delivered_text=del_a_abs,
                                    raw_model_output=raw_a_abs,
                                    answer_correct=corr_a_abs,
                                    answer_abstained=abst_a_abs,
                                    latency_ms=lat_a_abs,
                                    tokens_in=tin_a_abs,
                                    tokens_out=tout_a_abs,
                                    context_turns_count=len(haystack_turns),
                                )
                            )

                        # --- Strategy B: graph_memory ---
                        sess_b = f"sess-mem-{scenario.scenario_id}-{token_step}-{depth}-{trial}"
                        claim_item = {
                            "topic": scenario.claim.get("topic", "needle.fact"),
                            "content": scenario.claim.get("content", {}),
                        }
                        (
                            del_b,
                            raw_b,
                            corr_b,
                            abst_b,
                            lat_b,
                            tin_b,
                            tout_b,
                        ) = await _run_single_inference(
                            backend=backend,
                            session_id=sess_b,
                            turn_id=f"turn-mem-needle-{_new_id()}",
                            query_text=scenario.query_text,
                            recent_turns=[],
                            durable_memories=[claim_item],
                            expected_phrases=scenario.expected_phrases,
                            forbidden_phrases=scenario.forbidden_phrases,
                            expected_abstention=False,
                            max_output_tokens=max_output_tokens,
                            max_latency_seconds=max_latency_seconds,
                        )
                        measurements.append(
                            ContextRotCaseMeasurement(
                                scenario_id=scenario.scenario_id,
                                strategy="graph_memory",
                                target_tokens=token_step,
                                depth=depth,
                                trial_index=trial,
                                query_type="needle",
                                query_text=scenario.query_text,
                                delivered_text=del_b,
                                raw_model_output=raw_b,
                                answer_correct=corr_b,
                                answer_abstained=abst_b,
                                latency_ms=lat_b,
                                tokens_in=tin_b,
                                tokens_out=tout_b,
                                context_turns_count=0,
                            )
                        )

                        # Absent query in Strategy B
                        if scenario.absent_query_text:
                            (
                                del_b_abs,
                                raw_b_abs,
                                corr_b_abs,
                                abst_b_abs,
                                lat_b_abs,
                                tin_b_abs,
                                tout_b_abs,
                            ) = await _run_single_inference(
                                backend=backend,
                                session_id=sess_b,
                                turn_id=f"turn-mem-absent-{_new_id()}",
                                query_text=scenario.absent_query_text,
                                recent_turns=[],
                                durable_memories=[claim_item],
                                expected_phrases=scenario.absent_expected_phrases,
                                forbidden_phrases=scenario.absent_forbidden_phrases,
                                expected_abstention=True,
                                max_output_tokens=max_output_tokens,
                                max_latency_seconds=max_latency_seconds,
                            )
                            measurements.append(
                                ContextRotCaseMeasurement(
                                    scenario_id=scenario.scenario_id,
                                    strategy="graph_memory",
                                    target_tokens=token_step,
                                    depth=depth,
                                    trial_index=trial,
                                    query_type="absent",
                                    query_text=scenario.absent_query_text,
                                    delivered_text=del_b_abs,
                                    raw_model_output=raw_b_abs,
                                    answer_correct=corr_b_abs,
                                    answer_abstained=abst_b_abs,
                                    latency_ms=lat_b_abs,
                                    tokens_in=tin_b_abs,
                                    tokens_out=tout_b_abs,
                                    context_turns_count=0,
                                )
                            )

    finally:
        await backend.stop()

    aggregates = _aggregate_measurements(measurements)
    report = ContextRotEvaluationReport(
        artifact_version="rai-assistant-context-rot-v1",
        generated_at=_utc_now().isoformat(),
        backend_name=backend_name,
        model_name=model_name,
        token_steps=tuple(token_steps),
        depths=tuple(depths),
        total_cases=len(measurements),
        measurements=tuple(measurements),
        aggregates=tuple(aggregates),
    )
    return Success(report)


def save_context_rot_report(
    report: ContextRotEvaluationReport, output_path: Path
) -> None:
    """Serialize and save the context rot report to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)


def format_context_rot_summary_table(report: ContextRotEvaluationReport) -> str:
    """Format an ASCII table summarizing Strategy A vs Strategy B comparison."""
    lines: list[str] = [
        "==================================================================================================",
        f" CONTEXT ROT BENCHMARK REPORT — {report.model_name} ({report.backend_name})",
        "==================================================================================================",
        f"{'Strategy':<14} | {'Target':<7} | {'Depth':<8} | {'Recall Acc':<10} | {'Haluc Rate':<10} | {'Mean TTFT (ms)':<15} | {'Tokens In':<9}",
        "---------------+---------+----------+------------+------------+-----------------+----------",
    ]

    for agg in report.aggregates:
        depth_str = agg.depth or "OVERALL"
        lines.append(
            f"{agg.strategy:<14} | "
            f"{agg.target_tokens:<7} | "
            f"{depth_str:<8} | "
            f"{agg.needle_recall_accuracy * 100.0:>9.1f}% | "
            f"{agg.absent_hallucination_rate * 100.0:>9.1f}% | "
            f"{agg.mean_latency_ms:>15.1f} | "
            f"{agg.mean_tokens_in:>9.0f}"
        )

    lines.append(
        "=================================================================================================="
    )
    return "\n".join(lines)
