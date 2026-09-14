"""Deterministic AssistantModelBackend for offline conformance and CI tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Literal

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity

from ..memory import extract_memory_proposals, grounded_memory_response
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

        response_text, proposals = self._resolve_preferences(
            user_text=user_text,
            turn_id=request.turn_id,
            durable_memories=durable_memories,
        )

        candidate = AssistantCandidate(
            text=response_text,
            proposals=tuple(proposals),
            tokens_in=len(user_text.split()),
            tokens_out=len(response_text.split()),
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
