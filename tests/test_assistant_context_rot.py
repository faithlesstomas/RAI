"""Unit tests for the context rot and long-context evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.context_rot_evaluation import (
    ContextRotCorpus,
    estimate_tokens,
    evaluate_context_rot,
    format_context_rot_summary_table,
    generate_haystack,
    load_context_rot_corpus,
    save_context_rot_report,
)

_FIXTURE_PATH = Path("tests/fixtures/assistant/v1/context-rot.corpus.json")


def test_load_context_rot_corpus() -> None:
    """Ensure the versioned context-rot fixture loads cleanly and validates."""
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    assert isinstance(corpus, ContextRotCorpus)
    assert corpus.corpus_version == "rai-assistant-context-rot-v1"
    assert len(corpus.scenarios) >= 3
    assert len(corpus.distractors) >= 10

    scenario_ids = [s.scenario_id for s in corpus.scenarios]
    assert "test-db-port" in scenario_ids
    assert "shfmt-indent" in scenario_ids
    assert "code-language" in scenario_ids


def test_generate_haystack_needle_depths() -> None:
    """Verify haystack generator correctly places the needle at start, middle, and end."""
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    needle_user = "TEST_NEEDLE_USER_FACT_42"
    needle_assistant = "TEST_NEEDLE_ASSISTANT_ACK"

    for depth in ("start", "middle", "end"):
        turns = generate_haystack(
            distractors=corpus.distractors,
            needle_user=needle_user,
            needle_assistant=needle_assistant,
            target_tokens=2000,
            depth=depth,  # type: ignore[arg-type]
        )
        assert len(turns) >= 4

        # Find the index of the needle user turn
        needle_idx = next(
            i for i, turn in enumerate(turns) if turn["text"] == needle_user
        )
        total_turns = len(turns)
        relative_pos = needle_idx / total_turns

        if depth == "start":
            assert relative_pos <= 0.25, f"Expected start <= 0.25, got {relative_pos}"
        elif depth == "middle":
            assert 0.35 <= relative_pos <= 0.65, f"Expected middle ~0.50, got {relative_pos}"
        elif depth == "end":
            assert relative_pos >= 0.75, f"Expected end >= 0.75, got {relative_pos}"


def test_estimate_tokens_monotonic() -> None:
    """Ensure token estimation scales reasonably with text length."""
    short_text = "Cześć, jak się masz?"
    long_text = short_text * 50
    assert estimate_tokens(short_text) < estimate_tokens(long_text)
    assert estimate_tokens(short_text) >= 1


@pytest.mark.asyncio
async def test_evaluate_context_rot_deterministic() -> None:
    """Run hermetic evaluation with DeterministicAssistantBackend."""
    corpus = load_context_rot_corpus(_FIXTURE_PATH)
    backend = DeterministicAssistantBackend()

    # Use a small subset of scenarios and small token steps for fast CI test
    test_corpus = ContextRotCorpus(
        corpus_version=corpus.corpus_version,
        description=corpus.description,
        scenarios=corpus.scenarios[:1],
        distractors=corpus.distractors[:3],
    )

    result = await evaluate_context_rot(
        backend,
        corpus=test_corpus,
        token_steps=(200, 500),
        depths=("start", "end"),
        trials=1,
        max_output_tokens=64,
        max_latency_seconds=10.0,
    )

    assert isinstance(result, Success)
    report = result.unwrap()
    assert report.total_cases > 0
    assert len(report.measurements) == report.total_cases
    assert len(report.aggregates) > 0

    table = format_context_rot_summary_table(report)
    assert "CONTEXT ROT BENCHMARK REPORT" in table
    assert "raw_context" in table
    assert "graph_memory" in table
    assert "Tokens Out" in table

    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.json"
        save_context_rot_report(report, out_path)
        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["artifact_version"] == "rai-assistant-context-rot-v1"
        assert len(data["measurements"]) == report.total_cases
