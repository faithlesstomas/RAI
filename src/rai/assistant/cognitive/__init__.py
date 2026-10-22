"""Cognitive architecture package implementing Global Workspace Theory (GWT) deliberation."""

from __future__ import annotations

from .arbiter import SalienceArbiter
from .models import WorkspaceContribution, WorkspaceContributionKind, WorkspaceState
from .probe import JLensProbe
from .specialists import (
    ActionReasonerSpecialist,
    AnswerDraftSpecialist,
    CognitiveSpecialist,
    GroundingEvaluationSpecialist,
    RetrieverSpecialist,
)
from .evaluation import (
    BenchmarkScenario,
    ControllerAggregateScore,
    ControllerBenchmarkReport,
    ControllerScenarioMetric,
    create_default_benchmark_scenarios,
    evaluate_controller,
    format_benchmark_report_markdown,
    run_controller_comparative_benchmark,
    save_benchmark_report,
)
from .workspace import WorkspaceEngine

__all__ = [
    "ActionReasonerSpecialist",
    "AnswerDraftSpecialist",
    "CognitiveSpecialist",
    "GroundingEvaluationSpecialist",
    "JLensProbe",
    "RetrieverSpecialist",
    "SalienceArbiter",
    "WorkspaceContribution",
    "WorkspaceContributionKind",
    "WorkspaceEngine",
    "WorkspaceState",
    "BenchmarkScenario",
    "ControllerAggregateScore",
    "ControllerBenchmarkReport",
    "ControllerScenarioMetric",
    "create_default_benchmark_scenarios",
    "evaluate_controller",
    "format_benchmark_report_markdown",
    "run_controller_comparative_benchmark",
    "save_benchmark_report",
]
