"""Unit tests for the multi-turn conversational benchmark and detector heuristics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from returns.result import Result, Success

from rai.assistant import conversational_evaluation
from rai.assistant import store as assistant_store
from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.conversational_evaluation import (
    ConversationalCorpus,
    ConversationalTurnCase,
    _phrase_requirements_pass,
    detect_parroting,
    detect_repetition,
    detect_role_confusion,
    load_conversational_corpus,
    run_conversational_benchmark,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.ports import LifecycleState
from rai.kernel.records import ActionFailure

FIXTURE_PATH = Path("tests/fixtures/assistant/v2/conversational-dialog.corpus.json")
V1_FIXTURE_PATH = Path("tests/fixtures/assistant/v1/conversational-dialog.corpus.json")


def test_detect_parroting() -> None:
    """Verify parroting detection on user echo and role marker prefixes."""
    user = "Jaka jest dziś pogoda w Warszawie?"

    # Role markers
    assert detect_parroting(user, "użytkownik: Jaka jest dziś pogoda?")
    assert detect_parroting(user, "User: Jaka jest dziś pogoda?")
    assert detect_parroting(user, "<|im_start|>user\nJaka...")

    # Exact echo
    assert detect_parroting(user, "Jaka jest dziś pogoda w Warszawie?")
    assert detect_parroting(user, "Jaka jest dziś pogoda w Warszawie?\nOdpowiedź...")

    # Normal non-parroting response
    assert not detect_parroting(user, "Dziś w Warszawie jest słonecznie, 18 stopni.")
    assert not detect_parroting(user, "")


def test_detect_repetition() -> None:
    """Verify repetition loop detection on repeated sentences."""
    repeating = (
        "Dziś w Warszawie jest słonecznie. "
        "Dziś w Warszawie jest słonecznie. "
        "Dziś w Warszawie jest słonecznie."
    )
    assert detect_repetition(repeating)

    normal = (
        "Cześć! W czym mogę Ci dzisiaj pomóc? Mogę odpalić skrypt lub wyszukać pliki."
    )
    assert not detect_repetition(normal)
    assert not detect_repetition("Krótki tekst.")


def test_detect_role_confusion() -> None:
    """Verify role confusion detection when model hallucinates turn boundaries."""
    confused = "Oto odpowiedź.\n\nUżytkownik: A co dalej?\nAsystent: Dalej robimy..."
    assert detect_role_confusion(confused)

    confused_im = "Oto kod.<|im_start|>user\nDzięki!"
    assert detect_role_confusion(confused_im)

    clean = "Oto prosty skrypt w bashu:\n```bash\necho hello\n```"
    assert not detect_role_confusion(clean)


MIN_SCENARIOS = 5
MIN_TURNS = 10
FIRST_SCENARIO_TURNS = 2


class _CountingDeterministicBackend(DeterministicAssistantBackend):
    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self.start_calls += 1
        return await super().start()

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self.stop_calls += 1
        return await super().stop()


class _CountingStore(SQLiteMemoryGraphStore):
    start_calls = 0
    stop_calls = 0

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        type(self).start_calls += 1
        return await super().start()

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        type(self).stop_calls += 1
        return await super().stop()


def test_load_conversational_corpus() -> None:
    """Verify that the conversational corpus fixture is correctly loaded and structured."""
    corpus = load_conversational_corpus(FIXTURE_PATH)
    assert isinstance(corpus, ConversationalCorpus)
    assert corpus.corpus_version == "rai-assistant-conversational-dialog-v2"
    assert len(corpus.scenarios) >= MIN_SCENARIOS

    total_turns = sum(len(s.turns) for s in corpus.scenarios)
    assert total_turns >= MIN_TURNS

    first_scenario = corpus.scenarios[0]
    assert first_scenario.scenario_id == "chitchat-greeting"
    assert len(first_scenario.turns) == FIRST_SCENARIO_TURNS
    assert first_scenario.turns[0].user_text == "Cześć, jak się masz?"


def test_load_v1_corpus_maps_expected_phrases_to_accepted_alternatives() -> None:
    """Keep historical v1 fixtures readable without preserving their weak judge."""
    corpus = load_conversational_corpus(V1_FIXTURE_PATH)
    first_turn = corpus.scenarios[0].turns[0]

    assert first_turn.required_phrases == ()
    assert "cześć" in first_turn.accepted_any_phrases


def test_required_phrases_are_conjunctive() -> None:
    turn = ConversationalTurnCase(
        turn_id="required-all",
        user_text="Napisz skrypt",
        required_phrases=("set -euo pipefail", "hello world", "echo"),
    )

    required_passed, accepted_passed = _phrase_requirements_pass(
        turn, "#!/bin/bash\necho hello world"
    )

    assert not required_passed
    assert accepted_passed


def test_required_phrases_accept_short_polish_inflection() -> None:
    turn = ConversationalTurnCase(
        turn_id="polish-vocative",
        user_text="Jak ma na imię pies?",
        required_phrases=("Burek",),
    )

    required_passed, _accepted_passed = _phrase_requirements_pass(
        turn, "Cześć Bureku!"
    )

    assert required_passed


@pytest.mark.asyncio
async def test_run_conversational_benchmark_with_deterministic_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run conversational benchmark against DeterministicAssistantBackend."""
    backend = _CountingDeterministicBackend()
    output_path = tmp_path / "conv-report.json"
    benchmark_corpus_path = tmp_path / "conv-corpus.json"
    benchmark_corpus_path.write_text(
        json.dumps(
            {
                "corpus_version": "test-dialog-v2",
                "scenarios": [
                    {
                        "scenario_id": "lifecycle",
                        "turns": [
                            {
                                "turn_id": "turn-1",
                                "user_text": "Przywitaj się.",
                                "accepted_any_phrases": ["rozumiem"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _CountingStore.start_calls = 0
    _CountingStore.stop_calls = 0
    monkeypatch.setattr(
        conversational_evaluation, "SQLiteMemoryGraphStore", _CountingStore
    )

    async def run_inline(function: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return function(*args, **kwargs)

    monkeypatch.setattr(assistant_store.asyncio, "to_thread", run_inline)

    result = await run_conversational_benchmark(
        backend,
        corpus_path=benchmark_corpus_path,
        output_path=output_path,
    )

    assert isinstance(result, Success)
    report = result.unwrap()

    assert report.backend_name == "deterministic"
    assert report.scenario_count == 1
    assert report.total_turns == 1
    assert 0.0 <= report.parroting_rate <= 1.0
    assert 0.0 <= report.repetition_rate <= 1.0
    assert 0.0 <= report.role_confusion_rate <= 1.0
    assert output_path.exists()
    assert backend.state.value == "STOPPED"
    assert backend.start_calls == 1
    assert backend.stop_calls == 1
    assert _CountingStore.start_calls == 1
    assert _CountingStore.stop_calls == 1
    assert report.judge_version == "conversational-phrase-v3"
    assert report.prompt_template_version == "deterministic-v1"
    assert all(m.raw_model_output == "Rozumiem." for m in report.measurements)

    with open(output_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["backend_name"] == "deterministic"
    assert len(saved_data["measurements"]) == 1
