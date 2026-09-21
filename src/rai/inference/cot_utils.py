"""
Utilities for Chain of Thought (CoT) / reasoning content extraction and formatting.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Patterns matching closed thinking blocks across architectures
THINK_BLOCK_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", flags=re.DOTALL),
    re.compile(r"<\|think\|>(.*?)(?:<\|/think\|>|<\|end_of_thought\|>|</think>)", flags=re.DOTALL),
    re.compile(r"<thought>(.*?)</thought>", flags=re.DOTALL),
]

# Patterns matching unclosed thinking blocks (truncated generation)
UNCLOSED_THINK_PATTERNS = [
    re.compile(r"<think>(.*)$", flags=re.DOTALL),
    re.compile(r"<\|think\|>(.*)$", flags=re.DOTALL),
    re.compile(r"<thought>(.*)$", flags=re.DOTALL),
]


def extract_reasoning_and_content(
    text: str,
    explicit_reasoning: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Extracts reasoning/thinking tokens from text output or merges explicit reasoning.

    Returns:
        tuple of (clean_text, reasoning_content or None)
    """
    if explicit_reasoning and explicit_reasoning.strip():
        # Clean tags from text if present
        clean_text = text
        for pattern in THINK_BLOCK_PATTERNS:
            clean_text = pattern.sub("", clean_text)
        return clean_text.strip(), explicit_reasoning.strip()

    if not text:
        return "", None

    # Check for closed thinking blocks
    extracted_reasoning: list[str] = []
    clean_text = text

    for pattern in THINK_BLOCK_PATTERNS:
        matches = pattern.findall(clean_text)
        if matches:
            for m in matches:
                if m.strip():
                    extracted_reasoning.append(m.strip())
            clean_text = pattern.sub("", clean_text)

    # Check for unclosed thinking block at the end (truncated generation)
    if not extracted_reasoning:
        for pattern in UNCLOSED_THINK_PATTERNS:
            match = pattern.search(clean_text)
            if match:
                unclosed_content = match.group(1).strip()
                if unclosed_content:
                    extracted_reasoning.append(unclosed_content)
                clean_text = pattern.sub("", clean_text)
                break

    reasoning_str = "\n\n".join(extracted_reasoning).strip() if extracted_reasoning else None
    return clean_text.strip(), reasoning_str
