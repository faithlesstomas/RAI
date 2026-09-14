"""Direct local AssistantModelBackend adapter wrapping LocalTextEngine.

Resolves issue #2 by bypassing broken chat templating paths and using inspectable
plain-text completion prompts.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity, _new_id, _utc_now

from ..context import DEFAULT_SYSTEM_INSTRUCTION
from ..records import (
    AssistantCandidate,
    InferenceRequest,
    MemoryProposal,
    make_assistant_failure,
)


class LocalAssistantBackend:
    """Direct local model backend adapter for AssistantModelBackend protocol."""

    def __init__(  # noqa: PLR0913
        self,
        engine: object | None = None,
        model_name: str | None = None,
        backend_name: str = "ollama",
        max_output_tokens: int = 128,
        temperature: float = 0.2,
        model_artifact_version: str | None = None,
    ) -> None:
        self.engine = engine
        self.model_name = model_name or "local-model"
        self.backend_name = backend_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.model_artifact_version = model_artifact_version
        self.prompt_template_version = "rai-assistant-plain-text-v2"
        self._state = LifecycleState.CREATED
        self.producer = ProducerIdentity(
            producer_id=f"local-assistant-{self.backend_name}",
            kind="model",
            version="1.0.0",
        )

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        if self.engine is None:
            self._state = LifecycleState.FAILED
            return Failure(
                make_assistant_failure(
                    code="ENGINE_NOT_CONFIGURED",
                    message="no local text engine is configured",
                )
            )
        if self.engine is not None and hasattr(self.engine, "load"):
            try:
                load_res = await self.engine.load()
                if isinstance(load_res, Failure):
                    self._state = LifecycleState.FAILED
                    return Failure(
                        make_assistant_failure(
                            code="ENGINE_LOAD_FAILED",
                            message=f"failed to load engine: {load_res.failure()}",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                self._state = LifecycleState.FAILED
                return Failure(
                    make_assistant_failure(
                        code="ENGINE_LOAD_FAILED",
                        message=f"exception loading engine: {exc}",
                    )
                )
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        if self.engine is not None and hasattr(self.engine, "unload"):
            try:
                await self.engine.unload()
            except Exception:  # noqa: BLE001, S110
                pass
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    def _format_prompt(self, request: InferenceRequest) -> str:
        """Format an inspectable plain-text prompt bypassing chat template issues."""
        system_text = request.system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        lines = [
            f"System: {system_text}",
            "Use only relevant conversation and memory evidence below. Durable memory contains "
            "user-stated claims or preferences, not verified world facts. If evidence is missing "
            "or conflicting, say that you do not know. Never invent missing details.",
            "",
        ]

        durable_memories = request.context.content.get("durable_memories", [])
        if isinstance(durable_memories, (list, tuple)) and durable_memories:
            lines.append(
                "Trwała pamięć grafowa (twierdzenia lub preferencje podane przez użytkownika):"
            )
            for mem in durable_memories:
                if isinstance(mem, dict):
                    topic = mem.get("topic", "")
                    content = mem.get("content", {})
                    lines.append(f"- [{topic}] {content}")
            lines.append("")

        recent_turns = request.context.content.get("recent_turns", [])
        if isinstance(recent_turns, (list, tuple)) and recent_turns:
            lines.append("Historia bieżącej rozmowy:")
            for turn in recent_turns:
                if isinstance(turn, dict):
                    role = str(turn.get("role", "")).capitalize()
                    text = str(turn.get("text", ""))
                    lines.append(f"{role}: {text}")
            lines.append("")

        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        lines.append(f"Użytkownik: {user_text}")
        lines.append("Asystent:")
        return "\n".join(lines)

    def _extract_proposals(
        self, user_text: str, turn_id: str
    ) -> tuple[MemoryProposal, ...]:
        """Deterministically extract bounded memory proposals from user statements."""
        lowered = user_text.lower()
        proposals: list[MemoryProposal] = []

        pref_match = None
        if "guile" in lowered:
            pref_match = "Guile"
        elif "python" in lowered or "pythona" in lowered:
            pref_match = "Python"

        if (
            "preferuj" in lowered
            or "preferuję" in lowered
            or "zapamiętaj" in lowered
            or "zmień tę preferencję" in lowered
            or "używaj" in lowered
        ) and pref_match:
            proposals.append(
                MemoryProposal(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=self.producer,
                    source_turn_id=turn_id,
                    kind="preference",
                    topic="code_examples",
                    content={
                        "topic": "code_examples",
                        "preference": pref_match,
                        "raw_statement": user_text,
                    },
                )
            )

        return tuple(proposals)

    @staticmethod
    def _truncate_repetition(text: str) -> tuple[str, bool]:
        """Stop exact sentence loops commonly emitted by very small local models."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        seen: set[str] = set()
        kept: list[str] = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part).strip().casefold()
            if normalized and normalized in seen:
                return " ".join(kept).strip(), True
            if normalized:
                seen.add(normalized)
                kept.append(part.strip())
        return text.strip(), False

    @staticmethod
    def _ground_preference_response(
        user_text: str,
        durable_memories: tuple[object, ...] | list[object],
        proposals: tuple[MemoryProposal, ...],
        model_text: str,
    ) -> tuple[str, bool]:
        """Apply the narrow deterministic grounding gate implemented by this MVP."""
        if proposals:
            preference = proposals[0].content.get("preference")
            if preference:
                return (
                    f"Zapamiętałem: w przykładach kodu będę preferować język {preference}.",
                    True,
                )

        lowered = user_text.lower()
        asks_for_language = "jakim języku" in lowered or "w jakim języku" in lowered
        if not asks_for_language:
            return model_text, False

        for memory in durable_memories:
            if not isinstance(memory, dict) or memory.get("topic") != "code_examples":
                continue
            content = memory.get("content", {})
            if isinstance(content, dict) and content.get("preference"):
                preference = content["preference"]
                return (
                    "Zgodnie z Twoją zapisaną preferencją, powinienem pokazywać "
                    f"przykłady kodu w języku {preference}.",
                    model_text.casefold().find(str(preference).casefold()) < 0,
                )

        return (
            "Nie mam zapisanej preferencji dotyczącej języka w przykładach kodu.",
            True,
        )

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        if cancellation.cancelled:
            return Failure(
                make_assistant_failure(
                    code="CANCELLED",
                    message="inference request was cancelled",
                    request_id=request.request_id,
                )
            )

        prompt = self._format_prompt(request)
        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        durable_memories = request.context.content.get("durable_memories", [])
        if not isinstance(durable_memories, (list, tuple)):
            durable_memories = []

        if self.engine is None or not hasattr(self.engine, "generate"):
            return Failure(
                make_assistant_failure(
                    code="ENGINE_NOT_CONFIGURED",
                    message="no local text engine is configured",
                    request_id=request.request_id,
                )
            )
        try:
            gen_res = await self.engine.generate(
                prompt=prompt,
                stop=["\nUżytkownik:", "\nSystem:", "\nAsystent:"],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            if isinstance(gen_res, Failure):
                return Failure(
                    make_assistant_failure(
                        code="ENGINE_ERROR",
                        message=f"local engine failed: {gen_res.failure()}",
                        request_id=request.request_id,
                        retryable=True,
                    )
                )
            result_obj = gen_res.unwrap()
            response_text = result_obj.text.strip()
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ENGINE_ERROR",
                    message=f"exception during local engine generation: {exc}",
                    request_id=request.request_id,
                    retryable=True,
                )
            )

        if not response_text:
            return Failure(
                make_assistant_failure(
                    code="INVALID_OUTPUT",
                    message="local engine returned an empty response",
                    request_id=request.request_id,
                )
            )

        response_text, repetition_truncated = self._truncate_repetition(response_text)
        proposals = self._extract_proposals(user_text, request.turn_id)
        grounded_text, grounding_override = self._ground_preference_response(
            user_text, durable_memories, proposals, response_text
        )
        candidate = AssistantCandidate(
            text=grounded_text,
            proposals=proposals,
            tokens_in=(
                result_obj.stats.input_tokens
                if result_obj.stats
                else len(prompt.split())
            ),
            tokens_out=(
                result_obj.stats.output_tokens
                if result_obj.stats
                else len(response_text.split())
            ),
            metadata={
                "backend": self.backend_name,
                "model": self.model_name,
                "model_artifact_version": self.model_artifact_version,
                "prompt_template_version": self.prompt_template_version,
                "finish_reason": result_obj.finish_reason,
                "grounding_override": grounding_override,
                "repetition_truncated": repetition_truncated,
                "delivered_words": len(grounded_text.split()),
                "raw_model_output": response_text,
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
        words = re.split(r"(\s+)", candidate.text)
        for word in words:
            if cancellation.cancelled:
                yield Failure(
                    make_assistant_failure(
                        code="CANCELLED",
                        message="streaming was cancelled",
                        request_id=request.request_id,
                    )
                )
                return
            if word:
                yield Success(word)
                await asyncio.sleep(0.005)
