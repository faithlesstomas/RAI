"""Unit tests for the multi-turn conversational benchmark and detector heuristics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.conversational_evaluation import (
    ConversationalCorpus,
    detect_parroting,
    detect_repetition,
    detect_role_confusion,
    load_conversational_corpus,
    run_conversational_benchmark,
)

FIXTURE_PATH = Path("tests/fixtures/assistant/v1/conversational-dialog.corpus.json")


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

    normal = "Cześć! W czym mogę Ci dzisiaj pomóc? Mogę odpalić skrypt lub wyszukać pliki."
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
EXPECTED_SCENARIO_COUNT = 5
EXPECTED_TURN_COUNT = 12


def test_load_conversational_corpus() -> None:
    """Verify that the conversational corpus fixture is correctly loaded and structured."""
    corpus = load_conversational_corpus(FIXTURE_PATH)
    assert isinstance(corpus, ConversationalCorpus)
    assert corpus.corpus_version == "rai-assistant-conversational-dialog-v1"
    assert len(corpus.scenarios) >= MIN_SCENARIOS

    total_turns = sum(len(s.turns) for s in corpus.scenarios)
    assert total_turns >= MIN_TURNS

    first_scenario = corpus.scenarios[0]
    assert first_scenario.scenario_id == "chitchat-greeting"
    assert len(first_scenario.turns) == FIRST_SCENARIO_TURNS
    assert first_scenario.turns[0].user_text == "Cześć, jak się masz?"


@pytest.mark.asyncio
async def test_run_conversational_benchmark_with_deterministic_backend(tmp_path: Path) -> None:
    """Run conversational benchmark against DeterministicAssistantBackend."""
    backend = DeterministicAssistantBackend()
    output_path = tmp_path / "conv-report.json"

    result = await run_conversational_benchmark(
        backend,
        corpus_path=FIXTURE_PATH,
        output_path=output_path,
    )

    assert isinstance(result, Success)
    report = result.unwrap()

    assert report.backend_name == "deterministic"
    assert report.scenario_count == EXPECTED_SCENARIO_COUNT
    assert report.total_turns == EXPECTED_TURN_COUNT
    assert 0.0 <= report.parroting_rate <= 1.0
    assert 0.0 <= report.repetition_rate <= 1.0
    assert 0.0 <= report.role_confusion_rate <= 1.0
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["backend_name"] == "deterministic"
    assert len(saved_data["measurements"]) == EXPECTED_TURN_COUNT
