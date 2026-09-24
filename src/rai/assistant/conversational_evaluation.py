"""Multi-turn conversational benchmark measuring dialog coherence and role stability.

Evaluates raw model outputs without deterministic grounding masks to verify:
- absence of prompt echoing ("papugowanie") and role marker leakage,
- absence of phrase repetition loops,
- maintenance of the assistant persona and conversational context across turns.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any, AsyncIterator, Sequence

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity, _new_id, _utc_now

from .judging import _required_phrase_matches
from .ports import AssistantModelBackend
from .records import (
    AssistantCandidate,
    ConversationTurn,
    InferenceRequest,
    make_assistant_failure,
)
from .service import AssistantService
from .store import SQLiteMemoryGraphStore

_PARROT_ROLE_MARKERS = (
    "użytkownik:",
    "uzytkownik:",
    "user:",
    "human:",
    "system:",
    "<|im_start|>",
    "<|im_end|>",
)
_MIN_ECHO_USER_LENGTH = 15
_MIN_REPETITION_PARTS = 3
_MIN_REPETITION_PART_LENGTH = 12


def detect_parroting(user_text: str, model_output: str) -> bool:
    """Detect if the model echoed the user question or emitted role headers."""
    lowered_output = model_output.strip().casefold()
    lowered_user = user_text.strip().casefold()
    if not lowered_output:
        return False
    for marker in _PARROT_ROLE_MARKERS:
        if lowered_output.startswith(marker):
            return True
    first_line = lowered_output.split("\n", 1)[0].strip()
    if first_line == lowered_user:
        return True
    if len(lowered_user) > _MIN_ECHO_USER_LENGTH and lowered_user in first_line:
        return True
    return False


def detect_repetition(model_output: str) -> bool:
    """Detect exact phrase or sentence repetition loops commonly seen in small models."""
    parts = [
        p.strip().casefold()
        for p in re.split(r"(?<=[.!?])\s+", model_output.strip())
        if p.strip()
    ]
    if len(parts) < _MIN_REPETITION_PARTS:
        return False
    seen: set[str] = set()
    for part in parts:
        if len(part) > _MIN_REPETITION_PART_LENGTH and part in seen:
            return True
        seen.add(part)
    return False


def detect_role_confusion(model_output: str) -> bool:
    """Detect if the model generated user turns or role transitions inside its response."""
    lowered = model_output.casefold()
    return any(
        marker in lowered
        for marker in (
            "\nużytkownik:",
            "\nuzytkownik:",
            "\nuser:",
            "\nhuman:",
            "<|im_start|>user",
        )
    )


class ConversationalTurnCase(BaseModel):
    """One user turn in a conversational evaluation scenario."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    user_text: str
    required_phrases: tuple[str, ...] = Field(default_factory=tuple)
    accepted_any_phrases: tuple[str, ...] = Field(default_factory=tuple)
    forbidden_phrases: tuple[str, ...] = Field(default_factory=tuple)
    forbidden_parroting: bool = True


class ConversationalScenario(BaseModel):
    """A multi-turn conversation scenario."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    description: str
    turns: tuple[ConversationalTurnCase, ...]


class ConversationalCorpus(BaseModel):
    """The complete multi-turn conversation test suite."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    description: str
    scenarios: tuple[ConversationalScenario, ...]


class ConversationalTurnMeasurement(BaseModel):
    """Recorded observation for one evaluated dialog turn."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    turn_id: str
    turn_index: int
    user_text: str
    delivered_text: str
    raw_model_output: str
    parroting_detected: bool
    repetition_detected: bool
    role_confusion_detected: bool
    required_phrases_passed: bool
    accepted_phrase_passed: bool
    coherence_passed: bool
    turn_latency_ms: float
    tokens_in: int
    tokens_out: int


class ConversationalEvaluationReport(BaseModel):
    """Aggregate benchmark report for multi-turn conversational quality."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    judge_version: str = "conversational-phrase-v3"
    backend_name: str
    model_name: str
    model_artifact_version: str | None
    prompt_template_version: str | None
    scenario_count: int
    total_turns: int
    mean_coherence: float
    parroting_rate: float
    repetition_rate: float
    role_confusion_rate: float
    mean_latency_ms: float
    measurements: tuple[ConversationalTurnMeasurement, ...]


class _CandidateRecordingBackend:
    """Benchmark-only decorator capturing raw candidates by input turn ID."""

    def __init__(self, delegate: AssistantModelBackend) -> None:
        self._delegate = delegate
        self._candidates: dict[str, AssistantCandidate] = {}

    @property
    def state(self) -> LifecycleState:
        return self._delegate.state

    @property
    def backend_name(self) -> str:
        return str(getattr(self._delegate, "backend_name", "unknown"))

    @property
    def model_name(self) -> str:
        return str(getattr(self._delegate, "model_name", "unknown"))

    @property
    def model_artifact_version(self) -> str | None:
        value = getattr(self._delegate, "model_artifact_version", None)
        return str(value) if value is not None else None

    @property
    def prompt_template_version(self) -> str:
        return str(
            getattr(self._delegate, "prompt_template_version", "unknown-template")
        )

    @property
    def max_output_tokens(self) -> int:
        return int(getattr(self._delegate, "max_output_tokens", 512))

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        return await self._delegate.start()

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        return await self._delegate.stop()

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        result = await self._delegate.generate(request, cancellation)
        if isinstance(result, Success):
            self._candidates[request.turn_id] = result.unwrap()
        return result

    async def stream(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        async for chunk in self._delegate.stream(request, cancellation):
            yield chunk

    def pop_candidate(self, turn_id: str) -> AssistantCandidate | None:
        return self._candidates.pop(turn_id, None)


def _phrase_requirements_pass(
    turn_case: ConversationalTurnCase, model_output: str
) -> tuple[bool, bool]:
    """Evaluate conjunctive requirements separately from accepted alternatives."""
    required_passed = all(
        _required_phrase_matches(phrase, model_output)
        for phrase in turn_case.required_phrases
    )
    accepted_passed = not turn_case.accepted_any_phrases or any(
        _required_phrase_matches(phrase, model_output)
        for phrase in turn_case.accepted_any_phrases
    )
    return required_passed, accepted_passed


def load_conversational_corpus(path: Path) -> ConversationalCorpus:
    """Load and validate the conversational benchmark corpus."""
    with open(path, "rb") as file:
        data = json.load(file)
    scenarios = []
    for s in data.get("scenarios", []):
        turns = [
            ConversationalTurnCase(
                turn_id=t["turn_id"],
                user_text=t["user_text"],
                required_phrases=tuple(t.get("required_phrases", ())),
                accepted_any_phrases=tuple(
                    t.get("accepted_any_phrases", t.get("expected_phrases", ()))
                ),
                forbidden_phrases=tuple(t.get("forbidden_phrases", ())),
                forbidden_parroting=t.get("forbidden_parroting", True),
            )
            for t in s.get("turns", [])
        ]
        scenarios.append(
            ConversationalScenario(
                scenario_id=s["scenario_id"],
                description=s.get("description", ""),
                turns=tuple(turns),
            )
        )
    return ConversationalCorpus(
        corpus_version=data.get("corpus_version", "unknown"),
        description=data.get("description", ""),
        scenarios=tuple(scenarios),
    )


async def run_conversational_benchmark(  # noqa: PLR0912, PLR0915
    backend: AssistantModelBackend,
    *,
    corpus_path: Path,
    output_path: Path | None = None,
) -> Result[ConversationalEvaluationReport, ActionFailure]:
    """Execute all multi-turn scenarios against isolated storage and produce report."""
    try:
        corpus = load_conversational_corpus(corpus_path)
    except Exception as exc:  # noqa: BLE001
        return Failure(
            make_assistant_failure(
                code="CORPUS_LOAD_ERROR",
                message=f"Failed to load conversational corpus: {exc}",
            )
        )

    with TemporaryDirectory(prefix="rai-conv-benchmark-") as directory:
        store = SQLiteMemoryGraphStore(Path(directory) / "memory.sqlite3")
        recording_backend = _CandidateRecordingBackend(backend)
        service = AssistantService(store=store, backend=recording_backend)
        service_res = await service.start()
        if isinstance(service_res, Failure):
            return Failure(service_res.failure())

        measurements: list[ConversationalTurnMeasurement] = []
        producer = ProducerIdentity(
            producer_id="benchmark-dialog",
            kind="evaluation",
            version="1.0.0",
        )

        try:
            for scenario in corpus.scenarios:
                session_id = f"session-conv-{scenario.scenario_id}"
                reply_to_turn_id: str | None = None

                for turn_idx, turn_case in enumerate(scenario.turns):
                    turn = ConversationTurn(
                        record_id=_new_id(),
                        producer=producer,
                        session_id=session_id,
                        role="user",
                        text=turn_case.user_text,
                        reply_to_turn_id=reply_to_turn_id,
                    )

                    start_time = time.monotonic()
                    turn_res = await service.accept_turn(turn)
                    latency_ms = (time.monotonic() - start_time) * 1000.0

                    if isinstance(turn_res, Failure):
                        return Failure(turn_res.failure())

                    response = turn_res.unwrap()
                    reply_to_turn_id = response.turn_id
                    delivered_text = response.text

                    raw_model_output = delivered_text
                    tokens_in = len(turn_case.user_text.split())
                    tokens_out = len(delivered_text.split())

                    candidate = recording_backend.pop_candidate(turn.record_id)
                    if candidate is not None:
                        raw_model_output = str(
                            candidate.metadata.get("raw_model_output", candidate.text)
                        )
                        tokens_in = candidate.tokens_in
                        tokens_out = candidate.tokens_out

                    # Evaluation checks on raw model output
                    parroting = detect_parroting(turn_case.user_text, raw_model_output)
                    repetition = detect_repetition(raw_model_output)
                    role_confusion = detect_role_confusion(raw_model_output)

                    # Coherence check
                    (
                        required_phrases_passed,
                        accepted_phrase_passed,
                    ) = _phrase_requirements_pass(
                        turn_case,
                        raw_model_output,
                    )

                    contains_forbidden = False
                    for phrase in turn_case.forbidden_phrases:
                        if phrase.casefold() in raw_model_output.casefold():
                            contains_forbidden = True
                            break

                    coherence_passed = (
                        (not parroting or not turn_case.forbidden_parroting)
                        and not repetition
                        and not role_confusion
                        and required_phrases_passed
                        and accepted_phrase_passed
                        and not contains_forbidden
                        and bool(raw_model_output.strip())
                    )

                    measurements.append(
                        ConversationalTurnMeasurement(
                            scenario_id=scenario.scenario_id,
                            turn_id=turn_case.turn_id,
                            turn_index=turn_idx,
                            user_text=turn_case.user_text,
                            delivered_text=delivered_text,
                            raw_model_output=raw_model_output,
                            parroting_detected=parroting,
                            repetition_detected=repetition,
                            role_confusion_detected=role_confusion,
                            required_phrases_passed=required_phrases_passed,
                            accepted_phrase_passed=accepted_phrase_passed,
                            coherence_passed=coherence_passed,
                            turn_latency_ms=latency_ms,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                        )
                    )
        finally:
            await service.stop()

    total_turns = len(measurements)
    mean_coherence = (
        sum(1.0 for m in measurements if m.coherence_passed) / total_turns
        if total_turns
        else 0.0
    )
    parroting_rate = (
        sum(1.0 for m in measurements if m.parroting_detected) / total_turns
        if total_turns
        else 0.0
    )
    repetition_rate = (
        sum(1.0 for m in measurements if m.repetition_detected) / total_turns
        if total_turns
        else 0.0
    )
    role_confusion_rate = (
        sum(1.0 for m in measurements if m.role_confusion_detected) / total_turns
        if total_turns
        else 0.0
    )
    mean_latency = (
        sum(m.turn_latency_ms for m in measurements) / total_turns
        if total_turns
        else 0.0
    )

    report = ConversationalEvaluationReport(
        corpus_version=corpus.corpus_version,
        backend_name=getattr(backend, "backend_name", "unknown"),
        model_name=getattr(backend, "model_name", "unknown"),
        model_artifact_version=getattr(backend, "model_artifact_version", None),
        prompt_template_version=getattr(backend, "prompt_template_version", None),
        scenario_count=len(corpus.scenarios),
        total_turns=total_turns,
        mean_coherence=mean_coherence,
        parroting_rate=parroting_rate,
        repetition_rate=repetition_rate,
        role_confusion_rate=role_confusion_rate,
        mean_latency_ms=mean_latency,
        measurements=tuple(measurements),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
            )

    return Success(report)
