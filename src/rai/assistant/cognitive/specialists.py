"""Specialist agents competing for the Global Workspace broadcast."""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Any, Protocol, runtime_checkable

from returns.result import Failure, Result, Success

from rai.assistant.ports import AssistantModelBackend
from rai.assistant.records import AssistantCandidate, InferenceRequest
from rai.assistant.tool_calling import extract_model_tool_call
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, InferenceBudget, ProducerIdentity, _new_id, _utc_now
from rai.kernel.service import CapabilityService
from rai.skills.models import SkillRegistry

from .models import WorkspaceContribution, WorkspaceContributionKind, WorkspaceState

PRODUCER = ProducerIdentity(producer_id="workspace-specialist", kind="specialist", version="1.0.0")


def _make_specialist_budget(max_input_tokens: int, max_output_tokens: int) -> InferenceBudget:
    now = _utc_now()
    return InferenceBudget(
        record_id=_new_id(),
        timestamp=now,
        producer=PRODUCER,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_agent_turns=1,
        max_tool_calls=1,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=30),
    )


@runtime_checkable
class CognitiveSpecialist(Protocol):
    """Protocol for specialized cognitive agents proposing contributions to the blackboard."""

    @property
    def name(self) -> str:
        """Unique specialist identifier."""
        ...

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        """Inspect current blackboard working memory and propose candidate contribution with salience."""
        ...


class RetrieverSpecialist:
    """Specialist evaluating context sufficiency and proposing memory/knowledge retrieval."""

    def __init__(self, name: str = "retriever") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        del cancellation
        # If memory evidence was already broadcast or observations exist, retrieval urgency is low
        has_retrieval = any(
            c.kind == WorkspaceContributionKind.RETRIEVAL for c in workspace.history
        )
        if has_retrieval or workspace.observations:
            return Success(None)

        evidence_req = bool(workspace.metadata.get("evidence_required", False))
        salience = 0.85 if evidence_req else 0.25
        return Success(
            WorkspaceContribution(
                cycle=workspace.cycle,
                specialist_name=self.name,
                kind=WorkspaceContributionKind.RETRIEVAL,
                content=f"Evaluated retrieval needs for goal: {workspace.goal}",
                payload={"evidence_required": evidence_req},
                salience=salience,
            )
        )


class ActionReasonerSpecialist:
    """Specialist reasoning about required capabilities/skills and proposing tool executions."""

    def __init__(
        self,
        backend: AssistantModelBackend | None = None,
        capabilities: CapabilityService | None = None,
        skills: SkillRegistry | None = None,
        name: str = "action_reasoner",
    ) -> None:
        self._name = name
        self.backend = backend
        self.capabilities = capabilities
        self.skills = skills

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(  # noqa: PLR0911
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        # If we already have observations or tool calls executed, or if no backend/capabilities
        if not self.capabilities or not self.backend:
            return Success(None)

        max_allowed_observations = 2
        max_allowed_tool_calls = 2
        if len(workspace.observations) >= max_allowed_observations or workspace.tool_calls_count >= max_allowed_tool_calls:
            return Success(None)

        skills_prompt = ""
        if self.skills is not None:
            avail_caps = tuple(d.name for d in self.capabilities.registry.descriptors())
            skills_prompt = self.skills.format_skills_instruction(avail_caps)

        prompt = (
            f"You are the Action Reasoning Specialist in Rich AI.\n"
            f"User Goal: {workspace.goal}\n"
            f"{skills_prompt}\n\n"
            f"If an action is needed, emit exactly `<tool_call>call: capability_name{{...}}</tool_call>`.\n"
            f"If no action is needed, reply with 'NO_ACTION'."
        )

        if workspace.context is None:
            return Success(None)

        budget = _make_specialist_budget(1024, 256)
        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=PRODUCER,
            session_id=workspace.session_id,
            turn_id=workspace.turn_id,
            request_id=_new_id(),
            context=workspace.context,
            budget=budget,
            system_instruction=prompt,
        )

        gen_res = await self.backend.generate(inference_req, cancellation)
        if isinstance(gen_res, Failure):
            return Failure(gen_res.failure())

        candidate = gen_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            return Success(None)

        tool_call = extract_model_tool_call(candidate.text)
        if tool_call:
            _, cap_name, tool_args = tool_call
            salience = 0.9 if not workspace.observations else 0.4
            return Success(
                WorkspaceContribution(
                    cycle=workspace.cycle,
                    specialist_name=self.name,
                    kind=WorkspaceContributionKind.TOOL_CALL,
                    content=f"Proposed tool execution: {cap_name}",
                    payload={"capability": cap_name, "arguments": tool_args},
                    salience=salience,
                    tokens_in=candidate.tokens_in,
                    tokens_out=candidate.tokens_out,
                )
            )

        return Success(None)


class AnswerDraftSpecialist:
    """Specialist generating answer drafts from memory and observations."""

    def __init__(
        self,
        backend: AssistantModelBackend | None = None,
        name: str = "answer_draft",
    ) -> None:
        self._name = name
        self.backend = backend

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        if not self.backend:
            return Success(None)

        # Build prompt synthesizing goal and accumulated observations
        obs_text = "None."
        if workspace.observations:
            obs_lines = [json.dumps(obs, ensure_ascii=False) for obs in workspace.observations]
            obs_text = "\n".join(obs_lines)

        prompt = (
            f"Goal: {workspace.goal}\n"
            f"Observations:\n{obs_text}\n\n"
            f"Provide a helpful, concise answer to the user based strictly on the observations."
        )

        if workspace.context is None:
            return Success(None)

        budget = _make_specialist_budget(1500, 512)
        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=PRODUCER,
            session_id=workspace.session_id,
            turn_id=workspace.turn_id,
            request_id=_new_id(),
            context=workspace.context,
            budget=budget,
            system_instruction=prompt,
        )

        gen_res = await self.backend.generate(inference_req, cancellation)
        if isinstance(gen_res, Failure):
            return Failure(gen_res.failure())

        candidate = gen_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            return Success(None)

        # If observations exist or this is the final cycle, answer salience is high
        if workspace.observations or workspace.cycle >= workspace.max_cycles - 1:
            salience = 0.95
        else:
            # Baseline conversational salience
            salience = 0.60

        return Success(
            WorkspaceContribution(
                cycle=workspace.cycle,
                specialist_name=self.name,
                kind=WorkspaceContributionKind.ANSWER_DRAFT,
                content=candidate.text,
                payload={"candidate": candidate.model_dump()},
                salience=salience,
                tokens_in=candidate.tokens_in,
                tokens_out=candidate.tokens_out,
            )
        )


class GroundingEvaluationSpecialist:
    """Specialist verifying that proposed answers are grounded in observations."""

    def __init__(self, name: str = "grounding_eval") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        del cancellation
        # Only evaluate if the current broadcast is an answer draft
        if not workspace.broadcast or workspace.broadcast.kind != WorkspaceContributionKind.ANSWER_DRAFT:
            return Success(None)

        # If answer is ungrounded (e.g. empty when observations exist)
        answer_text = workspace.broadcast.content
        if not answer_text.strip():
            return Success(
                WorkspaceContribution(
                    cycle=workspace.cycle,
                    specialist_name=self.name,
                    kind=WorkspaceContributionKind.CRITIQUE,
                    content="Draft answer is empty.",
                    salience=0.8,
                )
            )

        return Success(None)
