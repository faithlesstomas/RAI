"""Unit tests for Chain of Thought extraction and ReasoningBudgetLogitsProcessor."""

from __future__ import annotations

import numpy as np

from rai.inference.cot_utils import extract_reasoning_and_content
from rai.inference.engines.llama import ReasoningBudgetLogitsProcessor


def test_extract_reasoning_standard_think_tags() -> None:
    text = "<think>Calculating 2 + 2 * 2 = 2 + 4 = 6</think>The result is 6."
    clean, reasoning = extract_reasoning_and_content(text)
    assert clean == "The result is 6."
    assert reasoning == "Calculating 2 + 2 * 2 = 2 + 4 = 6"


def test_extract_reasoning_pipe_think_tags() -> None:
    text = "<|think|>Analyzing query.<|/think|>Here is the answer."
    clean, reasoning = extract_reasoning_and_content(text)
    assert clean == "Here is the answer."
    assert reasoning == "Analyzing query."


def test_extract_reasoning_thought_tags() -> None:
    text = "<thought>Thinking...</thought>Final answer."
    clean, reasoning = extract_reasoning_and_content(text)
    assert clean == "Final answer."
    assert reasoning == "Thinking..."


def test_extract_reasoning_unclosed_tags() -> None:
    text = "Hello! <think>I should start thinking but generation stopped here"
    clean, reasoning = extract_reasoning_and_content(text)
    assert clean == "Hello!"
    assert reasoning == "I should start thinking but generation stopped here"


def test_extract_reasoning_explicit_reasoning() -> None:
    text = "Clean text without tags"
    explicit = "Explicit reasoning chain"
    clean, reasoning = extract_reasoning_and_content(text, explicit_reasoning=explicit)
    assert clean == "Clean text without tags"
    assert reasoning == "Explicit reasoning chain"


def test_extract_reasoning_empty() -> None:
    clean, reasoning = extract_reasoning_and_content("")
    assert clean == ""
    assert reasoning is None


def test_extract_reasoning_no_tags() -> None:
    clean, reasoning = extract_reasoning_and_content("Just a normal response.")
    assert clean == "Just a normal response."
    assert reasoning is None


def test_reasoning_budget_logits_processor() -> None:
    open_tokens = {100, 101}
    close_token = 102
    processor = ReasoningBudgetLogitsProcessor(
        open_token_ids=open_tokens,
        close_token_id=close_token,
        budget=5,
    )

    logits = np.zeros(200, dtype=np.float32)

    # Sequence without thinking: [1, 2, 3]
    out = processor(np.array([1, 2, 3]), logits)
    assert np.array_equal(out, logits)

    # Thinking opened, 3 tokens generated (within budget of 5)
    seq_within_budget = np.array([1, 100, 10, 11, 12])
    out = processor(seq_within_budget, logits)
    assert np.array_equal(out, logits)

    # Thinking opened, 5 tokens generated (budget reached!)
    seq_budget_reached = np.array([1, 100, 10, 11, 12, 13, 14])
    forced = processor(seq_budget_reached, logits)
    forced_val = 1e9
    assert forced[close_token] == forced_val
    assert forced[10] == -forced_val

    # Thinking already closed: [1, 100, 10, 11, 102, 20, 21, 22]
    seq_closed = np.array([1, 100, 10, 11, 102, 20, 21, 22])
    out = processor(seq_closed, logits)
    assert np.array_equal(out, logits)
