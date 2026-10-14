"""Deterministic AssistantModelBackend for offline conformance and CI tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Literal

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity

from ..memory import (
    extract_memory_proposals,
    grounded_memory_response,
    project_grounded_summary_memories,
    project_recent_turn_memories,
)
from ..records import (
    AssistantCandidate,
    InferenceRequest,
    MemoryProposal,
    make_assistant_failure,
)


class DeterministicAssistantBackend:
    """Deterministic model backend returning predictable answers and proposals."""

    def __init__(
        self,
        fail_mode: Literal[
            "none", "timeout", "cancellation", "error", "invalid"
        ] = "none",
        delay_seconds: float = 0.0,
    ) -> None:
        self.fail_mode = fail_mode
        self.delay_seconds = delay_seconds
        self._state = LifecycleState.CREATED
        self.backend_name = "deterministic"
        self.model_name = "deterministic-conformance"
        self.model_artifact_version = "1.0.0"
        self.prompt_template_version = "deterministic-v1"
        self.is_remote = False
        self.producer = ProducerIdentity(
            producer_id="deterministic-backend", kind="test", version="1.0.0"
        )

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    def _check_simulation(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> ActionFailure | None:
        if cancellation.cancelled or self.fail_mode == "cancellation":
            return make_assistant_failure(
                code="CANCELLED",
                message="inference request was cancelled",
                request_id=request.request_id,
            )
        if self.fail_mode == "timeout":
            return make_assistant_failure(
                code="TIMEOUT",
                message="inference timed out after deadline",
                request_id=request.request_id,
            )
        if self.fail_mode == "error":
            return make_assistant_failure(
                code="BACKEND_ERROR",
                message="internal backend failure",
                request_id=request.request_id,
                retryable=True,
            )
        if self.fail_mode == "invalid":
            return make_assistant_failure(
                code="INVALID_OUTPUT",
                message="invalid output: text length violates min_length",
                request_id=request.request_id,
            )
        return None

    def _resolve_preferences(
        self,
        user_text: str,
        turn_id: str,
        durable_memories: tuple[object, ...] | list[object],
    ) -> tuple[str, list[MemoryProposal]]:
        proposals = list(extract_memory_proposals(user_text, turn_id, self.producer))
        response_text, _ = grounded_memory_response(
            user_text, durable_memories, tuple(proposals), "Rozumiem."
        )
        return response_text, proposals

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        sim_failure = self._check_simulation(request, cancellation)
        if sim_failure is not None:
            return Failure(sim_failure)

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        durable_memories = request.context.content.get("durable_memories", [])
        if not isinstance(durable_memories, (list, tuple)):
            durable_memories = []
        recent_projection = project_recent_turn_memories(
            request.context.content.get("recent_turns", ()), self.producer
        )
        episodic_projection = project_recent_turn_memories(
            request.context.content.get("episodic_evidence", ()), self.producer
        )
        summary_projection = project_grounded_summary_memories(
            request.context.content.get("grounded_summaries", ())
        )
        grounding_memories = (
            *durable_memories,
            *recent_projection,
            *episodic_projection,
            *summary_projection,
        )

        response_text, proposals = self._resolve_preferences(
            user_text=user_text,
            turn_id=request.turn_id,
            durable_memories=grounding_memories,
        )
        tool_observations = request.context.content.get("tool_observations", [])
        if not isinstance(tool_observations, (list, tuple)):
            tool_observations = []
        has_evidence = bool(
            grounding_memories
            or request.context.content.get("episodic_evidence", ())
            or request.context.content.get("external_evidence", ())
            or tool_observations
        )
        if not proposals and tool_observations:
            obs = tool_observations[-1]
            if isinstance(obs, dict):
                cap = str(obs.get("capability", ""))
                obs_summary = str(obs.get("summary", ""))
                obs_content = obs.get("content", {})
                title = (
                    obs_content.get("title")
                    if isinstance(obs_content, dict) and obs_content.get("title")
                    else obs_summary
                )
            else:
                cap = str(getattr(obs, "capability", ""))
                title = getattr(obs, "summary", "")
            title_text = str(title).splitlines()[0] if title else ""
            if cap == "activity.query":
                response_text = f"Podsumowanie aktywności w systemie: {title_text}."
            elif cap == "browser.read_page" or "stron" in title_text.lower():
                response_text = f"Odpowiedź na podstawie odczytanej strony: {title_text or 'strona'}."
            elif cap == "browser.search":
                response_text = f"Odpowiedź na podstawie wyszukiwania: {title_text}."
            else:
                response_text = f"Odpowiedź na podstawie obserwacji ({cap}): {title_text}."
        elif not proposals and not has_evidence and "?" in user_text:
            response_text = (
                "Nie mam wystarczających źródeł, aby odpowiedzieć na to pytanie."
            )

        candidate = AssistantCandidate(
            text=response_text,
            proposals=tuple(proposals),
            tokens_in=len(user_text.split()),
            tokens_out=len(response_text.split()),
            metadata={
                "raw_model_output": response_text if tool_observations else "Rozumiem.",
                "grounding_override": response_text != "Rozumiem.",
            },
        )
        return Success(candidate)

    async def stream(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        res = await self.generate(request, cancellation)
        if isinstance(res, Failure):
            yield Failure(res.failure())
            return

        candidate = res.unwrap()
        words = candidate.text.split()
        for i, word in enumerate(words):
            if cancellation.cancelled:
                yield Failure(
                    make_assistant_failure(
                        code="CANCELLED",
                        message="streaming was cancelled",
                        request_id=request.request_id,
                    )
                )
                return
            yield Success(word + (" " if i < len(words) - 1 else ""))
