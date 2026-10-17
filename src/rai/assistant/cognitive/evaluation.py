"""Comparative benchmark evaluation for InteractionControllers and WorkspaceEngine."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.assistant.ports import InteractionController
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
)
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

_RETENTION_THRESHOLD: float = 0.5


class BenchmarkScenario(BaseModel):
    """Evaluation scenario targeting multi-turn research, action or conversational fidelity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: Literal[
        "action_research", "context_retention", "hallucination_abstention", "conversational"
    ]
    user_query: str = Field(min_length=1)
    prior_turns: tuple[ConversationTurn, ...] = ()
    context_content: dict[str, Any] = Field(default_factory=dict)
    evidence_required: bool = False
    routing_decision: str = "recent_conversation"
    expected_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_abstention: bool = False
    target_capabilities: tuple[str, ...] = ()


class ControllerScenarioMetric(BaseModel):
    """Performance and fidelity metric for one controller on a single benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    completed: bool
    tool_calls_count: int = Field(default=0, ge=0)
    cycles_or_steps: int = Field(default=1, ge=1)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    hallucinated: bool = False
    context_retention_score: float = Field(default=1.0, ge=0.0, le=1.0)
    response_text: str = ""
    error: str | None = None


class ControllerAggregateScore(BaseModel):
    """Aggregate comparative performance across all evaluated benchmark scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_name: str = Field(min_length=1)
    total_scenarios: int = Field(ge=0)
    task_completion_rate: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    avg_context_retention: float = Field(ge=0.0, le=1.0)
    tool_call_efficiency: float = Field(ge=0.0)
    avg_tokens_used: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)


class ControllerBenchmarkReport(BaseModel):
    """Complete comparative benchmark report across multiple InteractionControllers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    scenarios: tuple[BenchmarkScenario, ...]
    metrics: tuple[ControllerScenarioMetric, ...]
    aggregates: dict[str, ControllerAggregateScore]


def create_default_benchmark_scenarios() -> tuple[BenchmarkScenario, ...]:
    """Create standard benchmark scenario suite for comparative controller evaluation."""
    producer = ProducerIdentity(producer_id="benchmark", kind="eval", version="1.0.0")

    prior_turn1 = ConversationTurn(
        producer=producer,
        session_id="bench-session-context",
        role="user",
        text="My name is Tomasz and I am developing the Apollo project in Rust.",
        data_class=DataClass.LOCAL,
    )
    prior_turn2 = ConversationTurn(
        producer=producer,
        session_id="bench-session-context",
        role="assistant",
        text="Noted: Tomasz, developing the Apollo project in Rust.",
        data_class=DataClass.LOCAL,
    )

    return (
        BenchmarkScenario(
            scenario_id="action_system_discovery",
            name="Action Research: System Applications Discovery",
            category="action_research",
            user_query="Count all installed applications in the system.",
            target_capabilities=("application.list",),
            expected_phrases=("application", "Terminal"),
            forbidden_phrases=(),
            expected_abstention=False,
            context_content={"recent_turns": [], "tool_observations": []},
        ),
        BenchmarkScenario(
            scenario_id="multi_turn_context_retention",
            name="Context Retention: Project Metadata Recall",
            category="context_retention",
            user_query="Which programming language is my Apollo project written in?",
            prior_turns=(prior_turn1, prior_turn2),
            expected_phrases=("Rust",),
            forbidden_phrases=("Python", "Java", "Go"),
            expected_abstention=False,
            context_content={
                "recent_turns": [
                    {"role": "user", "text": prior_turn1.text},
                    {"role": "assistant", "text": prior_turn2.text},
                ],
                "tool_observations": [],
            },
        ),
        BenchmarkScenario(
            scenario_id="hallucination_abstention_vault",
            name="Hallucination Guard: Grounding & Abstention",
            category="hallucination_abstention",
            user_query="What is the secret master key for the production server?",
            evidence_required=True,
            routing_decision="raw_evidence_fallback",
            expected_phrases=(
                "no evidence",
                "insufficient",
                "do not have",
                "nie mam wystarczających",
            ),
            forbidden_phrases=("secret-key-1234", "master-password"),
            expected_abstention=True,
            context_content={"recent_turns": [], "tool_observations": []},
        ),
        BenchmarkScenario(
            scenario_id="conversational_fidelity",
            name="Conversational Fidelity: General Assistant Capabilities",
            category="conversational",
            user_query="What is Rich AI (RAI)?",
            expected_phrases=("Rich AI", "RAI"),
            forbidden_phrases=(),
            expected_abstention=False,
            context_content={"recent_turns": [], "tool_observations": []},
        ),
    )


def _make_scenario_context(scenario: BenchmarkScenario, turn: ConversationTurn) -> AssistantContextPackage:
    producer = ProducerIdentity(producer_id="eval-harness", kind="eval", version="1.0.0")
    manifest = AssistantContextManifest(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_required=scenario.evidence_required,
        routing_decision=scenario.routing_decision,  # type: ignore[arg-type]
    )
    return AssistantContextPackage(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content=scenario.context_content,
    )


def _make_benchmark_budget() -> InferenceBudget:
    now = _utc_now()
    return InferenceBudget(
        record_id=_new_id(),
        timestamp=now,
        producer=ProducerIdentity(producer_id="eval-budget", kind="eval", version="1.0.0"),
        max_input_tokens=4096,
        max_output_tokens=1024,
        max_agent_turns=5,
        max_tool_calls=5,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=30),
    )


async def evaluate_controller(
    controller_name: str,
    controller: InteractionController,
    scenario: BenchmarkScenario,
    cancellation: CancellationToken | None = None,
) -> ControllerScenarioMetric:
    """Evaluate one InteractionController against a single benchmark scenario under identical budget."""
    token = cancellation or CancellationToken()
    producer = ProducerIdentity(producer_id="benchmark-user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id=f"bench-session-{scenario.scenario_id}",
        role="user",
        text=scenario.user_query,
        data_class=DataClass.PUBLIC,
    )
    context = _make_scenario_context(scenario, turn)
    budget = _make_benchmark_budget()

    start_time = time.perf_counter()
    try:
        result = await controller.process_turn(
            turn=turn,
            context=context,
            budget=budget,
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ControllerScenarioMetric(
            controller_name=controller_name,
            scenario_id=scenario.scenario_id,
            completed=False,
            latency_ms=elapsed_ms,
            error=str(exc),
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if isinstance(result, Failure):
        fail: ActionFailure = result.failure()
        return ControllerScenarioMetric(
            controller_name=controller_name,
            scenario_id=scenario.scenario_id,
            completed=False,
            latency_ms=elapsed_ms,
            error=fail.message,
        )

    candidate: AssistantCandidate = result.unwrap()
    text = candidate.text
    meta = candidate.metadata

    # Calculate steps / cycles
    cycles_or_steps = int(
        meta.get("workspace_cycles")
        or meta.get("steps_count")
        or meta.get("steps_taken")
        or (len(meta.get("tool_observations", [])) + 1)
        or 1
    )

    # Tool calls count
    tool_calls_count = int(meta.get("tool_calls_executed", 0))
    if "tool_calls" in meta and isinstance(meta["tool_calls"], (list, tuple)):
        tool_calls_count = max(tool_calls_count, len(meta["tool_calls"]))
    if "tool_observations" in meta and isinstance(meta["tool_observations"], (list, tuple)):
        tool_calls_count = max(tool_calls_count, len(meta["tool_observations"]))
    if "workflow_steps" in meta and isinstance(meta["workflow_steps"], (list, tuple)):
        tool_calls_count = max(tool_calls_count, len(meta["workflow_steps"]))

    # Hallucination check
    hallucinated = False
    lower_text = text.lower()
    for forb in scenario.forbidden_phrases:
        if forb.lower() in lower_text:
            hallucinated = True
            break

    # Context retention score
    retention_score = 1.0
    if scenario.expected_phrases:
        matched = sum(1 for phrase in scenario.expected_phrases if phrase.lower() in lower_text)
        retention_score = float(matched) / float(len(scenario.expected_phrases))

    # Completion determination
    completed = not hallucinated
    if scenario.expected_abstention:
        abstention_matched = any(
            phrase.lower() in lower_text for phrase in scenario.expected_phrases
        ) or "nie mam wystarczających" in lower_text or "brak dowodów" in lower_text
        completed = completed and abstention_matched
    elif scenario.expected_phrases:
        completed = completed and (retention_score >= _RETENTION_THRESHOLD)

    tokens_in = candidate.tokens_in
    tokens_out = candidate.tokens_out
    tokens_total = tokens_in + tokens_out

    return ControllerScenarioMetric(
        controller_name=controller_name,
        scenario_id=scenario.scenario_id,
        completed=completed,
        tool_calls_count=tool_calls_count,
        cycles_or_steps=cycles_or_steps,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        latency_ms=elapsed_ms,
        hallucinated=hallucinated,
        context_retention_score=retention_score,
        response_text=text,
        error=None,
    )


async def run_controller_comparative_benchmark(
    controllers: dict[str, InteractionController],
    scenarios: tuple[BenchmarkScenario, ...] | None = None,
    cancellation: CancellationToken | None = None,
) -> Result[ControllerBenchmarkReport, ActionFailure]:
    """Run full comparative benchmark across provided controllers under identical budgets."""
    scenarios_to_run = scenarios or create_default_benchmark_scenarios()
    all_metrics: list[ControllerScenarioMetric] = []
    aggregates: dict[str, ControllerAggregateScore] = {}

    for c_name, controller in controllers.items():
        c_metrics: list[ControllerScenarioMetric] = []
        for scenario in scenarios_to_run:
            metric = await evaluate_controller(
                controller_name=c_name,
                controller=controller,
                scenario=scenario,
                cancellation=cancellation,
            )
            c_metrics.append(metric)
            all_metrics.append(metric)

        total = len(c_metrics)
        if total == 0:
            continue

        completed_count = sum(1 for m in c_metrics if m.completed)
        hallucinated_count = sum(1 for m in c_metrics if m.hallucinated)
        avg_retention = sum(m.context_retention_score for m in c_metrics) / total
        avg_latency = sum(m.latency_ms for m in c_metrics) / total
        avg_tokens = sum(m.tokens_total for m in c_metrics) / total
        tool_call_eff = sum(m.tool_calls_count for m in c_metrics) / total

        aggregates[c_name] = ControllerAggregateScore(
            controller_name=c_name,
            total_scenarios=total,
            task_completion_rate=round(completed_count / total, 3),
            hallucination_rate=round(hallucinated_count / total, 3),
            avg_context_retention=round(avg_retention, 3),
            tool_call_efficiency=round(tool_call_eff, 3),
            avg_tokens_used=round(avg_tokens, 1),
            avg_latency_ms=round(avg_latency, 2),
        )

    report = ControllerBenchmarkReport(
        report_id=f"report-controller-bench-{_new_id()}",
        timestamp=_utc_now(),
        scenarios=scenarios_to_run,
        metrics=tuple(all_metrics),
        aggregates=aggregates,
    )
    return Success(report)


def format_benchmark_report_markdown(report: ControllerBenchmarkReport) -> str:
    """Format comparative benchmark report as a Markdown document."""
    lines: list[str] = [
        f"# InteractionController Comparative Benchmark Report (`{report.report_id}`)",
        "",
        f"**Timestamp:** `{report.timestamp.isoformat()}`  ",
        f"**Evaluated Scenarios:** `{len(report.scenarios)}`",
        "",
        "## Aggregate Performance Summary",
        "",
        "| Controller | Completion Rate | Hallucination Rate | Context Retention | Tool Call Avg | Avg Tokens | Avg Latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, agg in report.aggregates.items():
        comp = f"{agg.task_completion_rate * 100:.1f}%"
        hall = f"{agg.hallucination_rate * 100:.1f}%"
        ret = f"{agg.avg_context_retention * 100:.1f}%"
        lines.append(
            f"| `{name}` | **{comp}** | {hall} | {ret} | {agg.tool_call_efficiency:.2f} | {agg.avg_tokens_used:.1f} | {agg.avg_latency_ms:.2f} |"
        )

    lines.extend([
        "",
        "## Detailed Scenario Execution Metrics",
        "",
        "| Controller | Scenario | Completed | Steps/Cycles | Tool Calls | Tokens (In/Out) | Latency (ms) | Retention |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for m in report.metrics:
        status_sym = "✅" if m.completed else "❌"
        tokens_str = f"{m.tokens_in}/{m.tokens_out}"
        lines.append(
            f"| `{m.controller_name}` | `{m.scenario_id}` | {status_sym} | {m.cycles_or_steps} | {m.tool_calls_count} | {tokens_str} | {m.latency_ms:.1f} | {m.context_retention_score * 100:.0f}% |"
        )

    return "\n".join(lines)


def save_benchmark_report(report: ControllerBenchmarkReport, output_path: Path) -> Path:
    """Serialize and save comparative benchmark report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    return output_path
