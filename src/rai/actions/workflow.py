"""Bounded agentic workflows and deterministic step controller.

Enforces:
1. Strict finite state transitions:
   PLANNING -> EVALUATING -> EXECUTING -> OBSERVING -> SYNTHESIZING -> COMPLETE
   (alongside backwards-compatible ResearchWorkflow lifecycle).
2. Hard limits: max tool calls, max total steps, timeouts.
3. Multi-stage approval isolation: approval tokens are scoped and single-use per action.
4. Unified token, latency, and tool-call accounting recorded into UsageLedger.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.ports import (
    AssistantActionHandler,
    AssistantModelBackend,
    InteractionController,
)
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
    ToolObservation,
    make_assistant_failure,
)
from rai.assistant.tool_calling import (
    extract_model_tool_call,
    format_tool_signature,
    intent_from_tool_call,
    strip_tool_call_tags,
)
from rai.diagnostics import trace
from rai.kernel.ports import CancellationToken, UsageLedger
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    DataClass,
    InferenceBudget,
    KernelRecord,
    ProducerIdentity,
    UsageRecord,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.skills.models import SkillRegistry

PRODUCER = ProducerIdentity(producer_id="bounded-step-controller", kind="service", version="1.0.0")
_MEMORY_ABSTENTION_TEXT: str = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)
_MAX_PREAMBLE_ATTACH_CHARS: int = 200


class AgentWorkflowState(str, Enum):
    """Generic finite state machine states for universal agentic workflows."""

    PLANNING = "PLANNING"
    EVALUATING = "EVALUATING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ALLOWED_GENERIC_TRANSITIONS: dict[AgentWorkflowState, set[AgentWorkflowState]] = {
    AgentWorkflowState.PLANNING: {
        AgentWorkflowState.EVALUATING,
        AgentWorkflowState.SYNTHESIZING,
        AgentWorkflowState.COMPLETE,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.EVALUATING: {
        AgentWorkflowState.EXECUTING,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.EXECUTING: {
        AgentWorkflowState.OBSERVING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.OBSERVING: {
        AgentWorkflowState.SYNTHESIZING,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.SYNTHESIZING: {
        AgentWorkflowState.COMPLETE,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.COMPLETE: set(),
    AgentWorkflowState.FAILED: set(),
    AgentWorkflowState.CANCELLED: set(),
}


_MAX_OBSERVATION_CHARS: int = 64000
_MAX_COMPACT_APPS_FULL: int = 1024


def _state_str(state: object) -> str:
    return state.value if isinstance(state, Enum) else str(state)


def _should_synthesize_action_candidate(  # noqa: PLR0911
    action_cand: AssistantCandidate | None,
) -> bool:
    """Determine if an action candidate should enter controller deliberation/synthesis."""
    if action_cand is None:
        return False
    meta = action_cand.metadata
    if meta.get("requires_synthesis") is True:
        return True
    if meta.get("untrusted_content") is True and not meta.get("active_choice_frame"):
        return True
    if any(
        k in meta
        for k in (
            "choice_rejected",
            "action_clarification",
            "intent_failure",
            "action_failure",
        )
    ):
        return False
    act_res = meta.get("action_result")
    if meta.get("active_choice_frame") and not isinstance(act_res, dict):
        return False
    if meta.get("action_proposal"):
        return False
    if isinstance(act_res, dict):
        if (
            act_res.get("record_type") == "action_failure"
            or act_res.get("status") == "FAILED"
        ):
            return False
        cap = str(act_res.get("capability") or "")
        output = act_res.get("output")
        if (
            cap == "activity.query"
            and isinstance(output, dict)
            and not output.get("episodes")
        ):
            return False
        return True
    return False


def _compact_observation_content(  # noqa: PLR0912
    output: object,
    fallback_text: str = "",
    capability: str = "",
) -> dict[str, Any]:
    """Compact tool output so counts and names survive observation length limits."""
    if not isinstance(output, dict):
        res: dict[str, Any] = {"raw_text": str(output)}
        if fallback_text and fallback_text != str(output):
            res["formatted_text"] = fallback_text
        return res

    compacted: dict[str, Any] = {}
    apps = output.get("applications")
    if isinstance(apps, (list, tuple)):
        compacted["total_count"] = int(output.get("total_count", len(apps)))
        if "requested_filter" in output:
            compacted["requested_filter"] = output["requested_filter"]
        if apps and isinstance(apps[0], dict):
            names = [
                f"{a.get('name', '')} ({a.get('desktop_id', '')})"
                if a.get("desktop_id")
                else str(a.get("name", ""))
                for a in apps
                if isinstance(a, dict)
            ]
        else:
            raw_names = output.get("application_names")
            names = (
                [str(n) for n in raw_names]
                if isinstance(raw_names, (list, tuple))
                else [str(a) for a in apps]
            )
        compacted["application_names"] = names[:1024]
        if len(apps) <= _MAX_COMPACT_APPS_FULL:
            compacted["applications"] = [
                {k: v for k, v in a.items() if k not in ("handle", "expires_at")}
                if isinstance(a, dict)
                else a
                for a in apps
            ]
        for k, v in output.items():
            if k not in (
                "total_count",
                "requested_filter",
                "application_names",
                "applications",
            ):
                compacted[k] = v
        return compacted

    for list_key in ("documents", "processes", "entries", "results"):
        items = output.get(list_key)
        if isinstance(items, (list, tuple)):
            compacted["total_count"] = int(output.get("total_count", len(items)))
            compacted[list_key] = [
                {
                    k: v
                    for k, v in item.items()
                    if k not in ("handle", "expires_at", "fingerprint")
                }
                if isinstance(item, dict)
                else item
                for item in items
            ]
            for k, v in output.items():
                if k not in ("total_count", list_key):
                    compacted[k] = v
            return compacted

    for k, v in output.items():
        compacted[k] = v

    if (
        fallback_text
        and "formatted_text" not in compacted
        and "text" not in compacted
        and "raw_text" not in compacted
        and capability != "application.list"
    ):
        compacted["formatted_text"] = fallback_text
    return compacted


def _format_older_observation_summary(  # noqa: PLR0912
    obs: Any,  # noqa: ANN401
) -> str:
    """Format an older observation into a concise 1-line summary for context compaction."""
    max_chars = 120
    if not isinstance(obs, dict):
        text = str(obs).replace("\n", " ").strip()
        return text[:max_chars] + "..." if len(text) > max_chars else text

    parts: list[str] = []
    if "status" in obs:
        parts.append(f"status={obs['status']}")
    if "code" in obs:
        parts.append(f"code={obs['code']}")
    if "total_count" in obs:
        parts.append(f"total_count={obs['total_count']}")
    if "capability" in obs:
        parts.append(f"capability={obs['capability']}")
    if "title" in obs:
        parts.append(f"title={obs['title']!r}")
    if "query" in obs:
        parts.append(f"query={obs['query']!r}")
    if "summary" in obs:
        s = str(obs["summary"]).replace("\n", " ").strip()
        parts.append(f"summary={s[:80]!r}")
    elif "message" in obs:
        m = str(obs["message"]).replace("\n", " ").strip()
        parts.append(f"message={m[:80]!r}")
    elif "application_names" in obs and isinstance(obs["application_names"], list):
        names = obs["application_names"]
        parts.append(f"apps={names[:3]!r} (total {len(names)})")
    elif "results" in obs and isinstance(obs["results"], list):
        parts.append(f"results_count={len(obs['results'])}")
    elif "raw_text" in obs:
        r = str(obs["raw_text"]).replace("\n", " ").strip()
        parts.append(f"text={r[:80]!r}")

    if not parts:
        dumped = json.dumps(obs, ensure_ascii=False)
        return dumped[:max_chars] + "..." if len(dumped) > max_chars else dumped

    return ", ".join(parts)



class BoundedWorkflowLimits(BaseModel):
    """Upper bounds preventing unbounded agent loops across any capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int | None = Field(default=8, ge=1)
    max_iterations: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=5, ge=0)
    max_tokens_budget: int | None = Field(default=16384, ge=256)
    max_wall_time_seconds: float | None = Field(default=120.0, ge=1.0)


class BoundedWorkflowStep(BaseModel):
    """Audited step in a generic bounded workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    state_before: AgentWorkflowState
    state_after: AgentWorkflowState
    action_taken: str = Field(min_length=1)
    capability: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] | None = None
    approval_id: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class BoundedWorkflow(KernelRecord):
    """Durable record of a generic bounded agentic workflow."""

    record_type: Literal["bounded_workflow"] = "bounded_workflow"
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    state: AgentWorkflowState = AgentWorkflowState.PLANNING
    limits: BoundedWorkflowLimits = Field(default_factory=BoundedWorkflowLimits)
    tool_calls_executed: int = Field(default=0, ge=0)
    steps: tuple[BoundedWorkflowStep, ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    approvals_used: tuple[str, ...] = ()
    final_synthesis: str | None = None
    data_class: DataClass = DataClass.PUBLIC


# Backwards-compatible specialized research workflow records
class ResearchWorkflowState(str, Enum):
    """Explicit lifecycle states for bounded research flows."""

    PLANNING = "PLANNING"
    SEARCHING = "SEARCHING"
    SELECTING_SOURCES = "SELECTING_SOURCES"
    READING_PAGE = "READING_PAGE"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResearchWorkflowLimits(BaseModel):
    """Hard upper bounds preventing unbounded agent loops in research workflows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_searches: int = Field(default=1, ge=1, le=5)
    max_page_reads: int = Field(default=2, ge=0, le=10)
    max_total_steps: int = Field(default=8, ge=1, le=20)


class ResearchWorkflowStep(BaseModel):
    """Single audited step within the research workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    state_before: ResearchWorkflowState
    state_after: ResearchWorkflowState
    action_taken: str = Field(min_length=1)
    capability: str | None = None
    target_resource: str | None = None
    approval_id: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class ResearchWorkflow(KernelRecord):
    """Durable record of a bounded research task."""

    record_type: Literal["research_workflow"] = "research_workflow"
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    state: ResearchWorkflowState = ResearchWorkflowState.PLANNING
    limits: ResearchWorkflowLimits = Field(default_factory=ResearchWorkflowLimits)
    searches_executed: int = Field(default=0, ge=0)
    page_reads_executed: int = Field(default=0, ge=0)
    steps: tuple[ResearchWorkflowStep, ...] = ()
    search_approval_id: str | None = None
    read_approval_ids: tuple[str, ...] = ()
    found_results: tuple[dict[str, Any], ...] = ()
    read_pages: tuple[dict[str, Any], ...] = ()
    final_synthesis: str | None = None
    data_class: DataClass = DataClass.PUBLIC


class BoundedStepController(InteractionController):
    """Universal controller governing bounded agentic workflows across any capabilities."""

    def __init__(  # noqa: PLR0913
        self,
        ledger: UsageLedger | None = None,
        capabilities: CapabilityService | None = None,
        backend: AssistantModelBackend | None = None,
        actions: AssistantActionHandler | None = None,
        limits: BoundedWorkflowLimits | None = None,
        skills: SkillRegistry | None = None,
        context_builder: object = None,
        max_full_observations: int = 3,
        bypass_intent_recognizer: bool = True,
    ) -> None:
        self.ledger = ledger
        self.capabilities = capabilities
        self.backend = backend
        self.actions = actions
        self.limits = limits or BoundedWorkflowLimits()
        self.skills = skills
        self.context_builder = context_builder
        self.max_full_observations = max_full_observations
        self.bypass_intent_recognizer = bypass_intent_recognizer

    async def process_turn(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        turn: ConversationTurn,
        context: AssistantContextPackage,
        budget: InferenceBudget,
        token: CancellationToken,
    ) -> Result[AssistantCandidate, ActionFailure]:
        """Execute a turn under the bounded agentic loop state machine."""
        manifest = context.manifest
        initial_observations: list[dict[str, Any]] = []
        initial_obs_records: list[dict[str, Any]] = list(
            context.content.get("tool_observations", [])
        )
        initial_obs_ids: list[str] = list(manifest.tool_observation_ids)
        initial_action_metadata: dict[str, Any] = {}
        executed_calls: set[tuple[str, str]] = set()

        if self.actions is not None and hasattr(self.actions, "handle"):
            if self.bypass_intent_recognizer:
                try:
                    action_cand = await self.actions.handle(turn, token, choice_frame_only=True)
                except TypeError:
                    action_cand = await self.actions.handle(turn, token)
            else:
                action_cand = await self.actions.handle(turn, token)
            if token.cancelled:
                return Failure(
                    make_assistant_failure(
                        code="CANCELLED",
                        message="Turn processing cancelled by user",
                        request_id=turn.record_id,
                    )
                )
            if action_cand is not None:
                act_res = action_cand.metadata.get("action_result") or {}
                if (
                    act_res.get("status") == "CANCELLED"
                    or act_res.get("error_code") == "CANCELLED"
                    or act_res.get("code") == "CANCELLED"
                ):
                    return Failure(
                        make_assistant_failure(
                            code="CANCELLED",
                            message="Turn action cancelled by user",
                            request_id=turn.record_id,
                        )
                    )
            requires_synthesis = _should_synthesize_action_candidate(action_cand)
            if action_cand is not None and not requires_synthesis:
                return Success(
                    action_cand.model_copy(
                        update={
                            "metadata": {
                                **action_cand.metadata,
                                "updated_manifest": manifest,
                                "latency_ms": 0.0,
                            }
                        }
                    )
                )
            if action_cand is not None and requires_synthesis:
                act_res = action_cand.metadata.get("action_result") or {}
                act_prop = action_cand.metadata.get("action_proposal") or {}
                output = act_res.get("output") if isinstance(act_res, dict) else {}
                capability = (
                    (act_prop.get("capability") if isinstance(act_prop, dict) else None)
                    or (act_res.get("capability") if isinstance(act_res, dict) else None)
                    or "tool"
                )
                summary = (
                    (output.get("title") if isinstance(output, dict) else None)
                    or (action_cand.text.splitlines()[0] if action_cand.text else "")
                    or str(capability)
                )
                obs_content = _compact_observation_content(
                    output,
                    fallback_text=action_cand.text,
                    capability=str(capability),
                )
                obs = ToolObservation(
                    source_id=(
                        (act_res.get("request_id") if isinstance(act_res, dict) else None)
                        or _new_id()
                    ),
                    source_turn_id=turn.record_id,
                    capability=str(capability),
                    status="SUCCESS",
                    summary=str(summary),
                    untrusted_content=bool(
                        action_cand.metadata.get("untrusted_content", False)
                    ),
                    content=obs_content,
                )
                initial_observations.append(obs_content)
                initial_obs_records.append(obs.model_dump(mode="json"))
                initial_obs_ids.append(obs.source_id)
                executed_calls.add((str(capability), "{}"))
                initial_action_metadata = {
                    **action_cand.metadata,
                    "synthesized_from_action": True,
                }

        if (
            not initial_observations
            and manifest.evidence_required
            and manifest.routing_decision == "no_evidence"
        ):
            return Success(
                AssistantCandidate(
                    text=_MEMORY_ABSTENTION_TEXT,
                    metadata={
                        "deterministic_abstention": True,
                        "reason": "memory_evidence_required_but_unavailable",
                        "updated_manifest": manifest,
                        "latency_ms": 0.0,
                    },
                )
            )

        if self.capabilities is None or self.backend is None:
            return Failure(
                ActionFailure(
                    request_id=f"bounded:{turn.record_id}",
                    capability="bounded_step_controller",
                    code="BACKEND_UNAVAILABLE",
                    message="Capabilities service or backend is not connected to BoundedStepController",
                    producer=PRODUCER,
                )
            )

        task_id = f"task:{turn.record_id}"
        last_candidate: AssistantCandidate | None = None
        preamble_box: list[str] = []
        reasoning_chunks: list[str] = []
        model_step_count = 0
        last_manifest: AssistantContextManifest = manifest.model_copy(
            update={"tool_observation_ids": tuple(initial_obs_ids)}
        )

        async def step_decider(  # noqa: PLR0912, PLR0915
            workflow: BoundedWorkflow,
            cancellation: CancellationToken,
        ) -> Result[dict[str, Any], ActionFailure]:
            nonlocal last_candidate, last_manifest, model_step_count
            descriptors = self.capabilities.registry.descriptors() if self.capabilities else ()
            actions_desc = "\n".join(
                format_tool_signature(d.name, d.description) for d in descriptors
            )
            skills_section = ""
            if self.skills is not None:
                avail_caps = tuple(d.name for d in descriptors)
                skills_section = self.skills.format_skills_instruction(avail_caps)

            all_obs = (*initial_observations, *workflow.observations)
            cutoff = max(0, len(all_obs) - self.max_full_observations)
            obs_lines = []
            tool_obs_records = list(initial_obs_records)
            obs_ids = list(initial_obs_ids)
            for idx, obs in enumerate(all_obs):
                i = idx + 1
                cap_name: str | None = None
                args_dict: dict[str, Any] | None = None
                if idx < len(initial_observations) and idx < len(initial_obs_records):
                    cap_name = str(initial_obs_records[idx].get("capability") or "") or None
                elif idx >= len(initial_observations):
                    step_obs_idx = idx - len(initial_observations)
                    obs_steps = [
                        s for s in workflow.steps
                        if s.observation is not None or s.state_after == AgentWorkflowState.OBSERVING
                    ]
                    if step_obs_idx < len(obs_steps):
                        cap_name = obs_steps[step_obs_idx].capability
                        args_dict = obs_steps[step_obs_idx].arguments

                clean_args = (
                    {k: v for k, v in args_dict.items() if k != "task_id" and v not in ("", None)}
                    if isinstance(args_dict, dict)
                    else None
                )
                obs_to_format = obs
                if isinstance(obs, dict):
                    obs_dict = dict(obs)
                    if cap_name and "capability" not in obs_dict:
                        obs_dict["capability"] = cap_name
                    if clean_args and "query" in clean_args and "query" not in obs_dict:
                        obs_dict["query"] = clean_args["query"]
                    obs_to_format = obs_dict

                if idx < cutoff:
                    summary_text = _format_older_observation_summary(obs_to_format)
                    obs_lines.append(f"Observation {i} (summary): {summary_text}")
                    if i > len(initial_observations):
                        step_obs_idx = i - len(initial_observations)
                        obs_id = f"{workflow.task_id}:obs:{step_obs_idx}"
                        obs_ids.append(obs_id)
                        tool_obs_records.append(
                            ToolObservation(
                                source_id=obs_id,
                                source_turn_id=turn.record_id,
                                capability=cap_name or "workflow_step",
                                status="SUCCESS",
                                summary=summary_text[:120],
                                untrusted_content=True,
                                content={"summary": summary_text},
                            ).model_dump(mode="json")
                        )
                else:
                    compacted_obs = (
                        _compact_observation_content(obs_to_format) if isinstance(obs_to_format, dict) else obs_to_format
                    )
                    if isinstance(compacted_obs, dict):
                        if cap_name and "capability" not in compacted_obs:
                            compacted_obs = {"capability": cap_name, **compacted_obs}
                        if clean_args and "query" in clean_args and "query" not in compacted_obs:
                            compacted_obs["query"] = clean_args["query"]
                    obs_summary = json.dumps(compacted_obs, ensure_ascii=False)
                    if len(obs_summary) > _MAX_OBSERVATION_CHARS:
                        obs_summary = obs_summary[:_MAX_OBSERVATION_CHARS] + "... (truncated)"
                    obs_lines.append(f"Observation {i}: {obs_summary}")
                    if i > len(initial_observations):
                        step_obs_idx = i - len(initial_observations)
                        obs_id = f"{workflow.task_id}:obs:{step_obs_idx}"
                        obs_ids.append(obs_id)
                        tool_obs_records.append(
                            ToolObservation(
                                source_id=obs_id,
                                source_turn_id=turn.record_id,
                                capability=cap_name or "workflow_step",
                                status="SUCCESS",
                                summary=obs_summary[:120],
                                untrusted_content=True,
                                content=(
                                    compacted_obs
                                    if isinstance(compacted_obs, dict)
                                    else {"value": compacted_obs}
                                ),
                            ).model_dump(mode="json")
                        )
            obs_text = "\n".join(obs_lines) if obs_lines else "None yet."
            prompt_content = f"User Goal: {workflow.goal}\n\nTool Observations:\n{obs_text}"

            closing_guidance = (
                "IMPORTANT: Review the Tool Observations above first. "
                "Do NOT re-invoke a tool that has already been executed with the same arguments. "
                "When the observations contain the required information, answer the user's "
                "specific request directly in their language without tool tags; do not repeat "
                "raw tool templates."
                if all_obs
                else (
                    "When you have collected all required information to answer the user request, "
                    "or if no tools are needed, provide the final comprehensive answer directly "
                    "without tool tags."
                )
            )
            workflow_instruction = (
                "You are an agent executing user tasks step by step.\n"
                "Available tools:\n"
                f"{actions_desc}"
                f"{skills_section}\n\n"
                "Tool calling format:\n"
                "To invoke a tool, output:\n"
                "<tool_call>call: tool_name{\"param\": \"value\"}</tool_call>\n"
                "Or in JSON format:\n"
                "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>\n\n"
                f"{prompt_content}\n\n"
                f"{closing_guidance}"
            )
            raw_base_inst = getattr(self.context_builder, "system_instruction", None)
            base_instruction = str(raw_base_inst) if isinstance(raw_base_inst, str) else ""
            if (
                base_instruction
                and isinstance(self.backend, DeterministicAssistantBackend)
                and not all_obs
            ):
                system_prompt = base_instruction
            elif base_instruction and "<tool_call>" in base_instruction:
                system_prompt = (
                    f"{base_instruction}{skills_section}\n\n"
                    f"{prompt_content}\n\n"
                    f"{closing_guidance}"
                )
            elif base_instruction:
                system_prompt = f"{base_instruction}\n\n{workflow_instruction}"
            else:
                system_prompt = workflow_instruction

            step_manifest = manifest.model_copy(
                update={"tool_observation_ids": tuple(obs_ids)}
            )
            last_manifest = step_manifest
            step_context = context.model_copy(
                update={
                    "manifest": step_manifest,
                    "content": {
                        **context.content,
                        "tool_observations": tool_obs_records,
                    },
                }
            )

            req = InferenceRequest(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=PRODUCER,
                session_id=turn.session_id,
                turn_id=turn.record_id,
                request_id=_new_id(),
                context=step_context,
                budget=budget,
                strategy="BOUNDED_STEP",
                system_instruction=system_prompt,
                model_name=str(getattr(self.backend, "model_name", "deterministic")),
            )

            model_step_count += 1
            step_num = model_step_count
            trace(
                "assistant.model.start",
                turn_id=turn.record_id,
                controller="bounded_step",
                step=step_num,
                observations=len(all_obs),
            )
            start = time.perf_counter()
            gen_res = await self.backend.generate(req, cancellation)
            latency = (time.perf_counter() - start) * 1000
            trace(
                "assistant.model.end",
                turn_id=turn.record_id,
                controller="bounded_step",
                step=step_num,
                elapsed_ms=round(latency),
                status="failure" if isinstance(gen_res, Failure) else "success",
            )
            if isinstance(gen_res, Failure):
                return Failure(gen_res.failure())

            cand = gen_res.unwrap()
            last_candidate = cand
            if rc := cand.metadata.get("reasoning_content"):
                if isinstance(rc, str) and rc.strip():
                    reasoning_chunks.append(rc.strip())
            tool_call = extract_model_tool_call(cand.text)
            if tool_call:
                preamble, cap_name, tool_args = tool_call
                norm_args = {
                    k: v
                    for k, v in (tool_args or {}).items()
                    if k != "task_id" and v not in ("", None)
                }
                call_key = (cap_name, json.dumps(norm_args, sort_keys=True))
                hit_iteration_limit = (
                    self.limits.max_iterations is not None
                    and model_step_count >= self.limits.max_iterations
                )
                if (call_key in executed_calls and all_obs) or hit_iteration_limit:
                    synth_prompt = (
                        f"{base_instruction}\n\n"
                        f"{prompt_content}\n\n"
                        "Provide the final comprehensive answer to the User Goal directly "
                        "in the user's language based on the Tool Observations above. "
                        "Do NOT emit any <tool_call> tags."
                    ).strip()
                    synth_req = req.model_copy(
                        update={
                            "record_id": _new_id(),
                            "request_id": _new_id(),
                            "system_instruction": synth_prompt,
                        }
                    )
                    synth_mode = (
                        "iteration_limit_synthesis"
                        if hit_iteration_limit and call_key not in executed_calls
                        else "duplicate_tool_synthesis"
                    )
                    trace(
                        "assistant.model.start",
                        turn_id=turn.record_id,
                        controller="bounded_step",
                        step=step_num,
                        observations=len(all_obs),
                        mode=synth_mode,
                    )
                    synth_start = time.perf_counter()
                    synth_res = await self.backend.generate(synth_req, cancellation)
                    synth_lat = (time.perf_counter() - synth_start) * 1000
                    trace(
                        "assistant.model.end",
                        turn_id=turn.record_id,
                        controller="bounded_step",
                        step=step_num,
                        elapsed_ms=round(synth_lat),
                        status="failure" if isinstance(synth_res, Failure) else "success",
                    )
                    if isinstance(synth_res, Failure):
                        return Failure(synth_res.failure())
                    synth_cand = synth_res.unwrap()
                    last_candidate = synth_cand
                    if synth_rc := synth_cand.metadata.get("reasoning_content"):
                        if isinstance(synth_rc, str) and synth_rc.strip():
                            reasoning_chunks.append(synth_rc.strip())
                    return Success({
                        "action_type": "finish",
                        "text": strip_tool_call_tags(synth_cand.text),
                        "tokens_in": cand.tokens_in + synth_cand.tokens_in,
                        "tokens_out": cand.tokens_out + synth_cand.tokens_out,
                        "latency_ms": latency + synth_lat,
                    })

                executed_calls.add(call_key)
                if preamble and not preamble_box:
                    preamble_box.append(preamble)
                pre_executed_observation: Any = None
                if self.actions is not None and hasattr(self.actions, "dispatch_intent"):
                    tool_intent = intent_from_tool_call(turn, cap_name, tool_args)
                    if tool_intent is not None:
                        step_meta = {
                            **turn.metadata,
                            "step_id": str(model_step_count),
                        }
                        if initial_action_metadata:
                            step_meta["in_turn_action"] = dict(initial_action_metadata)
                        step_turn = turn.model_copy(update={"metadata": step_meta})
                        tool_cand = await self.actions.dispatch_intent(
                            step_turn, tool_intent, cancellation
                        )
                        if tool_cand is not None:
                            initial_action_metadata.update(tool_cand.metadata)
                            act_res_dict = tool_cand.metadata.get("action_result") or {}
                            if isinstance(act_res_dict, dict) and "output" in act_res_dict:
                                raw_out = act_res_dict.get("output")
                            elif isinstance(act_res_dict, dict) and "code" in act_res_dict:
                                raw_out = {
                                    "status": "FAILED",
                                    "code": act_res_dict.get("code"),
                                    "message": act_res_dict.get("message") or tool_cand.text,
                                }
                            else:
                                raw_out = {"raw_text": tool_cand.text}
                            pre_executed_observation = _compact_observation_content(
                                raw_out,
                                fallback_text=tool_cand.text,
                                capability=cap_name,
                            )
                    elif (
                        self.capabilities is not None
                        and self.capabilities.registry.descriptor(cap_name) is None
                    ):
                        pre_executed_observation = {
                            "status": "FAILED",
                            "code": "UNKNOWN_TOOL",
                            "message": (
                                f"Tool '{cap_name}' is not registered. "
                                "Use one of the registered Available tools or answer directly."
                            ),
                        }
                return Success({
                    "action_type": "tool",
                    "capability": cap_name,
                    "arguments": tool_args,
                    "pre_executed_observation": pre_executed_observation,
                    "tokens_in": cand.tokens_in,
                    "tokens_out": cand.tokens_out,
                    "latency_ms": latency,
                })

            return Success({
                "action_type": "finish",
                "text": strip_tool_call_tags(cand.text),
                "tokens_in": cand.tokens_in,
                "tokens_out": cand.tokens_out,
                "latency_ms": latency,
            })

        loop_res = await self.run_agentic_loop(
            task_id=task_id,
            session_id=turn.session_id,
            goal=turn.text,
            capabilities=self.capabilities,
            step_decider=step_decider,
            cancellation=token,
            limits=self.limits,
            data_class=turn.data_class,
            actor=turn.producer,
        )
        if isinstance(loop_res, Failure):
            err = loop_res.failure()
            if err.code in {"TASK_CANCELLED", "CANCELLED"} or token.cancelled:
                return Failure(
                    make_assistant_failure(
                        code="CANCELLED",
                        message="Turn processing cancelled by user",
                        request_id=turn.record_id,
                    )
                )
            return Failure(err)

        workflow = loop_res.unwrap()
        total_latency = sum(step.latency_ms for step in workflow.steps)
        total_tokens_in = sum(step.tokens_in for step in workflow.steps)
        total_tokens_out = sum(step.tokens_out for step in workflow.steps)

        proposals = last_candidate.proposals if last_candidate is not None else ()
        cand_meta = dict(last_candidate.metadata) if last_candidate is not None else {}
        if reasoning_chunks:
            cand_meta["reasoning_content"] = "\n\n---\n\n".join(reasoning_chunks)
        delivered_text = strip_tool_call_tags(workflow.final_synthesis or "")
        if not delivered_text and preamble_box:
            delivered_text = preamble_box[0]
        elif (
            preamble_box
            and len(workflow.steps) <= 1
            and len(preamble_box[0].strip()) > 0
            and len(delivered_text) < _MAX_PREAMBLE_ATTACH_CHARS
            and not delivered_text.startswith(preamble_box[0][:30])
        ):
            delivered_text = f"{preamble_box[0]}\n\n{delivered_text}"

        synthesized_meta: dict[str, Any] = {}
        if initial_observations or workflow.tool_calls_executed > 0:
            synthesized_meta["synthesized_from_action"] = True

        candidate = AssistantCandidate(
            text=delivered_text,
            proposals=proposals,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            metadata={
                **initial_action_metadata,
                **cand_meta,
                **synthesized_meta,
                "bounded_workflow_id": workflow.task_id,
                "steps_count": len(workflow.steps),
                "tool_calls_executed": workflow.tool_calls_executed,
                "updated_manifest": last_manifest,
                "latency_ms": total_latency,
            },
        )
        return Success(candidate)

    # -------------------------------------------------------------------------
    # Generic Bounded Workflow Orchestration
    # -------------------------------------------------------------------------

    def create_generic_workflow(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        goal: str,
        limits: BoundedWorkflowLimits | None = None,
        data_class: DataClass = DataClass.PUBLIC,
        actor: ProducerIdentity | None = None,
    ) -> BoundedWorkflow:
        """Initialize a new bounded workflow in the PLANNING state."""
        return BoundedWorkflow(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=actor or PRODUCER,
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            state=AgentWorkflowState.PLANNING,
            limits=limits or BoundedWorkflowLimits(),
            data_class=data_class,
        )

    def transition_generic(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow,
        target_state: AgentWorkflowState,
        action_taken: str,
        capability: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[BoundedWorkflow, str]:
        """Transition generic workflow to a target state, validating against transition rules and limits."""
        allowed = ALLOWED_GENERIC_TRANSITIONS.get(workflow.state, set())
        if target_state not in allowed:
            return Failure(
                f"INVALID_TRANSITION: cannot transition from {_state_str(workflow.state)} to {_state_str(target_state)}"
            )
        if workflow.limits.max_steps is not None and len(workflow.steps) >= workflow.limits.max_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = BoundedWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=target_state,
            action_taken=action_taken,
            capability=capability,
            arguments=arguments or {},
            observation=observation,
            approval_id=approval_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": target_state,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    async def execute_generic_step(  # noqa: PLR0911, PLR0913
        self,
        workflow: BoundedWorkflow,
        capabilities: CapabilityService,
        capability_name: str,
        arguments: dict[str, Any],
        cancellation: CancellationToken,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        pre_executed_observation: object = None,
    ) -> Result[tuple[BoundedWorkflow, ActionResult], ActionFailure]:
        """Invoke a capability under generic state machine supervision: EVALUATING -> EXECUTING -> OBSERVING."""
        step_index = len(workflow.steps) + 1
        req_id = f"{workflow.task_id}:step:{step_index}"

        if (
            workflow.limits.max_tool_calls is not None
            and workflow.tool_calls_executed >= workflow.limits.max_tool_calls
        ):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="TOOL_CALL_LIMIT_EXCEEDED",
                    message="Maximum allowed tool calls limit reached",
                    producer=PRODUCER,
                )
            )

        if workflow.limits.max_steps is not None and len(workflow.steps) >= workflow.limits.max_steps:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="STEP_LIMIT_EXCEEDED",
                    message="Maximum allowed steps limit reached",
                    producer=PRODUCER,
                )
            )

        if approval_id is not None and approval_id in workflow.approvals_used:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="APPROVAL_ALREADY_USED",
                    message=f"Approval token '{approval_id}' has already been consumed",
                    producer=PRODUCER,
                )
            )

        # 1. PLANNING -> EVALUATING
        eval_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.EVALUATING,
            action_taken=f"evaluate:{capability_name}",
            capability=capability_name,
            arguments=arguments,
            approval_id=approval_id,
        )
        if isinstance(eval_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=eval_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = eval_tx.unwrap()

        # 2. EVALUATING -> EXECUTING
        exec_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.EXECUTING,
            action_taken=f"execute:{capability_name}",
            capability=capability_name,
            arguments=arguments,
            approval_id=approval_id,
        )
        if isinstance(exec_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=exec_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = exec_tx.unwrap()

        if pre_executed_observation is not None:
            out_dict = (
                pre_executed_observation
                if isinstance(pre_executed_observation, dict)
                else {"value": pre_executed_observation}
            )
            action_res = ActionResult(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=PRODUCER,
                request_id=req_id,
                capability=capability_name,
                output=out_dict,
                verification={"status": "dispatched"},
            )
        else:
            descriptor = capabilities.registry.descriptor(capability_name)
            if descriptor is None:
                return Failure(
                    ActionFailure(
                        request_id=req_id,
                        capability=capability_name,
                        code="CAPABILITY_NOT_FOUND",
                        message=f"Capability '{capability_name}' is not registered",
                        producer=PRODUCER,
                    )
                )

            call_arguments = dict(arguments)
            trusted_session_id = getattr(workflow, "session_id", None)
            if (
                isinstance(descriptor.input_schema, dict)
                and "session_id" in descriptor.input_schema.get("properties", {})
                and trusted_session_id
            ):
                call_arguments["session_id"] = trusted_session_id

            request = normalize_request(
                descriptor,
                call_arguments,
                actor=workflow.producer,
                request_id=req_id,
                data_class=workflow.data_class,
            )
            _, executed = await capabilities.invoke(request, cancellation)
            if isinstance(executed, Failure):
                _ = self.transition_generic(
                    workflow,
                    AgentWorkflowState.FAILED,
                    action_taken=f"execution_failed:{executed.failure().message}",
                    capability=capability_name,
                )
                return Failure(executed.failure())

            action_res = executed.unwrap()

        # 3. EXECUTING -> OBSERVING
        obs_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.OBSERVING,
            action_taken=f"observed:{capability_name}",
            capability=capability_name,
            observation=action_res.output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(obs_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=obs_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = obs_tx.unwrap()

        approvals_now = (*workflow.approvals_used, approval_id) if approval_id else workflow.approvals_used
        updated_wf = workflow.model_copy(
            update={
                "tool_calls_executed": workflow.tool_calls_executed + 1,
                "observations": (*workflow.observations, action_res.output),
                "approvals_used": approvals_now,
            }
        )

        if self.ledger is not None:
            await self.record_usage(
                updated_wf,
                processor=f"step_{capability_name}",
                backend="capability",
                model=capability_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((updated_wf, action_res))

    async def execute_generic_synthesis(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[BoundedWorkflow, ActionFailure]:
        """Advance workflow to SYNTHESIZING (if needed) and COMPLETE with final text."""
        req_id = f"{workflow.task_id}:synthesis"
        if workflow.state in {AgentWorkflowState.OBSERVING, AgentWorkflowState.PLANNING}:
            synth_tx = self.transition_generic(
                workflow,
                AgentWorkflowState.SYNTHESIZING,
                action_taken="begin_synthesis",
            )
            if isinstance(synth_tx, Failure):
                return Failure(
                    ActionFailure(
                        request_id=req_id,
                        capability="assistant.synthesis",
                        code="WORKFLOW_TRANSITION_FAILED",
                        message=synth_tx.failure(),
                        producer=PRODUCER,
                    )
                )
            workflow = synth_tx.unwrap()

        complete_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.COMPLETE,
            action_taken="synthesis_completed",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(complete_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="assistant.synthesis",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=complete_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = complete_tx.unwrap()

        updated = workflow.model_copy(update={"final_synthesis": final_synthesis})

        if self.ledger is not None:
            await self.record_usage(
                updated,
                processor="assistant_synthesizer",
                backend="assistant_model",
                model="synthesizer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=0,
                latency_ms=latency_ms,
            )

        return Success(updated)

    async def run_agentic_loop(  # noqa: PLR0911, PLR0912, PLR0913
        self,
        task_id: str,
        session_id: str,
        goal: str,
        capabilities: CapabilityService,
        step_decider: Callable[
            [BoundedWorkflow, CancellationToken],
            Awaitable[Result[dict[str, Any], ActionFailure]],
        ],
        cancellation: CancellationToken,
        limits: BoundedWorkflowLimits | None = None,
        data_class: DataClass = DataClass.PUBLIC,
        actor: ProducerIdentity | None = None,
    ) -> Result[BoundedWorkflow, ActionFailure]:
        """Run an end-to-end universal agentic loop governed by BoundedWorkflow state machine."""
        workflow = self.create_generic_workflow(
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            limits=limits,
            data_class=data_class,
            actor=actor,
        )

        while workflow.state not in {
            AgentWorkflowState.COMPLETE,
            AgentWorkflowState.FAILED,
            AgentWorkflowState.CANCELLED,
        }:
            if cancellation.cancelled:
                cancel_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.CANCELLED,
                    action_taken="cancelled_by_token",
                )
                if isinstance(cancel_tx, Success):
                    workflow = cancel_tx.unwrap()
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:cancel",
                        capability="agent_loop",
                        code="TASK_CANCELLED",
                        message="Workflow cancelled by token",
                        producer=PRODUCER,
                    )
                )

            if workflow.limits.max_steps is not None and len(workflow.steps) >= workflow.limits.max_steps:
                fail_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.FAILED,
                    action_taken="step_limit_exceeded",
                )
                if isinstance(fail_tx, Success):
                    workflow = fail_tx.unwrap()
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:limit",
                        capability="agent_loop",
                        code="STEP_LIMIT_EXCEEDED",
                        message=f"Step limit {workflow.limits.max_steps} exceeded",
                        producer=PRODUCER,
                    )
                )

            decision_res = await step_decider(workflow, cancellation)
            if isinstance(decision_res, Failure):
                fail_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.FAILED,
                    action_taken=f"decision_failed:{decision_res.failure().message}",
                )
                if isinstance(fail_tx, Success):
                    workflow = fail_tx.unwrap()
                return Failure(decision_res.failure())

            decision = decision_res.unwrap()
            action_type = decision.get("action_type")

            if action_type == "finish":
                final_text = decision.get("text", "")
                return await self.execute_generic_synthesis(
                    workflow,
                    final_synthesis=final_text,
                    tokens_in=decision.get("tokens_in", 0),
                    tokens_out=decision.get("tokens_out", 0),
                    latency_ms=decision.get("latency_ms", 0.0),
                )

            if action_type == "tool":
                cap_name = decision.get("capability")
                if not cap_name:
                    return Failure(
                        ActionFailure(
                            request_id=f"{workflow.task_id}:step",
                            capability="agent_loop",
                            code="MISSING_CAPABILITY",
                            message="Decision specified 'tool' without capability name",
                            producer=PRODUCER,
                        )
                    )
                step_res = await self.execute_generic_step(
                    workflow=workflow,
                    capabilities=capabilities,
                    capability_name=cap_name,
                    arguments=decision.get("arguments", {}),
                    cancellation=cancellation,
                    approval_id=decision.get("approval_id"),
                    tokens_in=decision.get("tokens_in", 0),
                    tokens_out=decision.get("tokens_out", 0),
                    latency_ms=decision.get("latency_ms", 0.0),
                    pre_executed_observation=decision.get("pre_executed_observation"),
                )
                if isinstance(step_res, Failure):
                    return Failure(step_res.failure())

                workflow, _ = step_res.unwrap()

                # Deliberate next step: OBSERVING -> PLANNING
                plan_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.PLANNING,
                    action_taken=f"deliberate_after:{cap_name}",
                )
                if isinstance(plan_tx, Failure):
                    return Failure(
                        ActionFailure(
                            request_id=f"{workflow.task_id}:loop",
                            capability="agent_loop",
                            code="WORKFLOW_TRANSITION_FAILED",
                            message=plan_tx.failure(),
                            producer=PRODUCER,
                        )
                    )
                workflow = plan_tx.unwrap()
            else:
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:decision",
                        capability="agent_loop",
                        code="INVALID_DECISION_TYPE",
                        message=f"Unknown decision action_type: {action_type}",
                        producer=PRODUCER,
                    )
                )

        return Success(workflow)

    # -------------------------------------------------------------------------
    # Specialized Research Workflow Orchestration (Backwards Compatibility)
    # -------------------------------------------------------------------------

    def create_workflow(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        query: str,
        limits: ResearchWorkflowLimits | None = None,
        data_class: DataClass = DataClass.PUBLIC,
        actor: ProducerIdentity | None = None,
    ) -> ResearchWorkflow:
        """Initialize a new bounded research workflow in the PLANNING state."""
        return ResearchWorkflow(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=actor or PRODUCER,
            task_id=task_id,
            session_id=session_id,
            query=query,
            state=ResearchWorkflowState.PLANNING,
            limits=limits or ResearchWorkflowLimits(),
            data_class=data_class,
        )

    async def record_usage(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow | ResearchWorkflow,
        processor: str,
        backend: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[UsageRecord, ActionFailure]:
        """Record structured usage metrics into the ledger."""
        if self.ledger is None:
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:usage",
                    capability="usage_ledger",
                    code="LEDGER_UNAVAILABLE",
                    message="No UsageLedger configured",
                    producer=PRODUCER,
                )
            )
        record = UsageRecord(
            task_id=workflow.task_id,
            processor=processor,
            backend=backend,
            model=model,
            provider="local",
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            tool_calls=tool_calls,
            images=0,
            audio_seconds=0.0,
            latency_seconds=latency_ms / 1000.0,
            producer=PRODUCER,
        )
        return await self.ledger.append(record)

    def verify_multistage_approval(
        self,
        workflow: ResearchWorkflow,
        stage: Literal["search", "read_page"],
        approval_id: str,
    ) -> Result[bool, str]:
        """Verify that approvals for searching and page reading are distinct."""
        if not approval_id:
            return Failure("MISSING_APPROVAL_ID")

        if stage == "search":
            return Success(True)

        if stage == "read_page":
            if workflow.search_approval_id and approval_id == workflow.search_approval_id:
                return Failure("APPROVAL_REUSE_FORBIDDEN: cannot reuse search approval for page read")
            if approval_id in workflow.read_approval_ids:
                return Failure("APPROVAL_ALREADY_USED: duplicate approval token for page read")
            return Success(True)

        return Failure("INVALID_STAGE")

    def transition_to_search(
        self,
        workflow: ResearchWorkflow,
        approval_id: str | None = None,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from PLANNING to SEARCHING."""
        if workflow.state != ResearchWorkflowState.PLANNING:
            return Failure(f"INVALID_TRANSITION: cannot transition to SEARCHING from {_state_str(workflow.state)}")
        if workflow.searches_executed >= workflow.limits.max_searches:
            return Failure("SEARCH_LIMIT_EXCEEDED")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.SEARCHING,
            action_taken="initiate_search",
            capability="browser.search",
            approval_id=approval_id,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.SEARCHING,
                "search_approval_id": approval_id or workflow.search_approval_id,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def record_search_completed(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        results: tuple[dict[str, Any], ...],
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SEARCHING to SELECTING_SOURCES with recorded results."""
        if workflow.state != ResearchWorkflowState.SEARCHING:
            return Failure(f"INVALID_TRANSITION: cannot complete search from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.SELECTING_SOURCES,
            action_taken=f"search_completed: found {len(results)} items",
            capability="browser.search",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.SELECTING_SOURCES,
                "searches_executed": workflow.searches_executed + 1,
                "found_results": results,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def transition_to_reading_page(
        self,
        workflow: ResearchWorkflow,
        target_url: str,
        approval_id: str,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SELECTING_SOURCES to READING_PAGE with verified page read approval."""
        if workflow.state != ResearchWorkflowState.SELECTING_SOURCES:
            return Failure(f"INVALID_TRANSITION: cannot transition to READING_PAGE from {_state_str(workflow.state)}")
        if workflow.page_reads_executed >= workflow.limits.max_page_reads:
            return Failure("PAGE_READ_LIMIT_EXCEEDED")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        approval_check = self.verify_multistage_approval(workflow, "read_page", approval_id)
        if isinstance(approval_check, Failure):
            return Failure(approval_check.failure())

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.READING_PAGE,
            action_taken=f"read_page_target: {target_url}",
            capability="browser.read_page",
            target_resource=target_url,
            approval_id=approval_id,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.READING_PAGE,
                "read_approval_ids": (*workflow.read_approval_ids, approval_id),
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def record_page_read_completed(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        page_content: dict[str, Any],
        more_sources_needed: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Record completed page read and advance to SELECTING_SOURCES or SYNTHESIZING."""
        if workflow.state != ResearchWorkflowState.READING_PAGE:
            return Failure(f"INVALID_TRANSITION: cannot complete page read from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        reads_now = workflow.page_reads_executed + 1
        can_read_more = reads_now < workflow.limits.max_page_reads and more_sources_needed
        next_state = ResearchWorkflowState.SELECTING_SOURCES if can_read_more else ResearchWorkflowState.SYNTHESIZING

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=next_state,
            action_taken=f"page_read_completed: {page_content.get('title', 'untitled')}",
            capability="browser.read_page",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": next_state,
                "page_reads_executed": reads_now,
                "read_pages": (*workflow.read_pages, page_content),
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def transition_to_complete(
        self,
        workflow: ResearchWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SYNTHESIZING to COMPLETE with final answer."""
        if workflow.state not in {ResearchWorkflowState.SYNTHESIZING, ResearchWorkflowState.SELECTING_SOURCES}:
            return Failure(f"INVALID_TRANSITION: cannot complete from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.COMPLETE,
            action_taken="synthesis_completed",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.COMPLETE,
                "final_synthesis": final_synthesis,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    async def execute_bounded_search(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        capabilities: CapabilityService,
        query: str,
        cancellation: CancellationToken,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[tuple[ResearchWorkflow, ActionResult], ActionFailure]:
        """Orchestrate search transition, capability invocation, and usage ledger entry."""
        tx = self.transition_to_search(workflow, approval_id)
        if isinstance(tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = tx.unwrap()

        descriptor = capabilities.registry.descriptor("browser.search")
        if descriptor is None:
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="CAPABILITY_NOT_FOUND",
                    message="browser.search capability is not registered",
                    producer=PRODUCER,
                )
            )

        request = normalize_request(
            descriptor,
            {"task_id": workflow.task_id, "query": query},
            actor=workflow.producer,
            request_id=f"{workflow.task_id}:search",
            data_class=workflow.data_class,
        )
        _, executed = await capabilities.invoke(request, cancellation)
        if isinstance(executed, Failure):
            return Failure(executed.failure())

        action_res = executed.unwrap()
        results = action_res.output.get("results", ())

        completed = self.record_search_completed(
            workflow,
            tuple(results),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="browser_search",
                backend="browser",
                model="web_search",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((workflow, action_res))

    async def execute_bounded_read(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        capabilities: CapabilityService,
        handle: str,
        target_url: str,
        approval_id: str,
        cancellation: CancellationToken,
        more_sources_needed: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[tuple[ResearchWorkflow, ActionResult], ActionFailure]:
        """Orchestrate page read transition, capability invocation, and usage ledger entry."""
        step_num = workflow.page_reads_executed + 1
        req_id = f"{workflow.task_id}:read_page:{step_num}"
        tx = self.transition_to_reading_page(workflow, target_url=target_url, approval_id=approval_id)
        if isinstance(tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = tx.unwrap()

        descriptor = capabilities.registry.descriptor("browser.read_page")
        if descriptor is None:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="CAPABILITY_NOT_FOUND",
                    message="browser.read_page capability is not registered",
                    producer=PRODUCER,
                )
            )

        request = normalize_request(
            descriptor,
            {"task_id": workflow.task_id, "handle": handle},
            actor=workflow.producer,
            request_id=req_id,
            data_class=workflow.data_class,
        )
        _, executed = await capabilities.invoke(request, cancellation)
        if isinstance(executed, Failure):
            return Failure(executed.failure())

        action_res = executed.unwrap()
        page_content = action_res.output

        completed = self.record_page_read_completed(
            workflow,
            page_content,
            more_sources_needed=more_sources_needed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="browser_read_page",
                backend="browser",
                model="web_reader",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((workflow, action_res))

    async def execute_workflow_synthesis(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, ActionFailure]:
        """Transition workflow to COMPLETE and record synthesis usage metrics."""
        completed = self.transition_to_complete(
            workflow,
            final_synthesis=final_synthesis,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:synthesis",
                    capability="assistant.synthesis",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="assistant_synthesizer",
                backend="assistant_model",
                model="synthesizer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=0,
                latency_ms=latency_ms,
            )

        return Success(workflow)

    async def run_bounded_research(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        query: str,
        capabilities: CapabilityService,
        synthesizer: Callable[[ResearchWorkflow, CancellationToken], Awaitable[Result[str, ActionFailure]]],
        cancellation: CancellationToken,
        search_approval_id: str | None = None,
        read_approval_factory: Callable[[int, str], str] | None = None,
        limits: ResearchWorkflowLimits | None = None,
        actor: ProducerIdentity | None = None,
    ) -> Result[ResearchWorkflow, ActionFailure]:
        """Run an end-to-end bounded research task: search -> read -> synthesize -> complete."""
        workflow = self.create_workflow(
            task_id=task_id,
            session_id=session_id,
            query=query,
            limits=limits,
            actor=actor,
        )

        # 1. Bounded search
        search_res = await self.execute_bounded_search(
            workflow, capabilities, query, cancellation, approval_id=search_approval_id
        )
        if isinstance(search_res, Failure):
            return Failure(search_res.failure())
        workflow, _ = search_res.unwrap()

        # 2. Bounded page reads (up to max_page_reads)
        results = workflow.found_results
        max_reads = min(len(results), workflow.limits.max_page_reads)
        for i in range(max_reads):
            item = results[i]
            target_url = item.get("url", "")
            handle = item.get("handle", "")
            approval_id = (
                read_approval_factory(i + 1, target_url)
                if read_approval_factory
                else f"{task_id}:read:{i + 1}"
            )
            more_needed = (i + 1) < max_reads
            read_res = await self.execute_bounded_read(
                workflow,
                capabilities,
                handle=handle,
                target_url=target_url,
                approval_id=approval_id,
                cancellation=cancellation,
                more_sources_needed=more_needed,
            )
            if isinstance(read_res, Failure):
                return Failure(read_res.failure())
            workflow, _ = read_res.unwrap()

        # 3. Model synthesis
        synth_res = await synthesizer(workflow, cancellation)
        if isinstance(synth_res, Failure):
            return Failure(synth_res.failure())
        final_text = synth_res.unwrap()

        # 4. Complete and record usage
        return await self.execute_workflow_synthesis(workflow, final_synthesis=final_text)


class TaskStepController(BoundedStepController):
    """Backwards-compatible subclass alias for BoundedStepController."""


WorkflowLimits = BoundedWorkflowLimits
WorkflowStep = BoundedWorkflowStep

__all__ = [
    "ALLOWED_GENERIC_TRANSITIONS",
    "AgentWorkflowState",
    "BoundedStepController",
    "BoundedWorkflow",
    "BoundedWorkflowLimits",
    "BoundedWorkflowStep",
    "ResearchWorkflow",
    "ResearchWorkflowLimits",
    "ResearchWorkflowState",
    "ResearchWorkflowStep",
    "TaskStepController",
    "WorkflowLimits",
    "WorkflowStep",
]
