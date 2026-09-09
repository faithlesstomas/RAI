"""Text normalization for speech synthesis to strip Markdown and formatting."""

from __future__ import annotations

import re


def normalize_speech_text(text: str) -> str:
    """Normalize input text for speech synthesis by stripping Markdown and symbols."""
    if not text:
        return ""

    normalized = text

    # Remove code blocks: replace with content or pause
    normalized = re.sub(r"```[a-zA-Z0-9_-]*\n(.*?)```", r" \1 ", normalized, flags=re.DOTALL)
    normalized = re.sub(r"```(.*?)```", r" \1 ", normalized, flags=re.DOTALL)

    # Remove inline code marks: `code` -> code
    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)

    # Remove image links: ![alt](url) -> alt
    normalized = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", normalized)

    # Convert links: [text](url) -> text
    normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)

    # Remove raw URLs
    normalized = re.sub(r"https?://\S+", "", normalized)

    # Remove HTML tags: <tag> -> empty
    normalized = re.sub(r"<[^>]+>", "", normalized)

    # Remove bold and italic markers: ***text***, **text**, *text*, ___text___, __text__, _text_
    normalized = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", normalized)
    normalized = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", normalized)

    # Remove strikethrough: ~~text~~ -> text
    normalized = re.sub(r"~~([^~]+)~~", r"\1", normalized)

    # Strip header markers (#, ##, ###...) at line beginnings
    normalized = re.sub(r"^#{1,6}\s+", "", normalized, flags=re.MULTILINE)

    # Strip blockquotes (> ) at line beginnings
    normalized = re.sub(r"^>\s+", "", normalized, flags=re.MULTILINE)

    # Strip bullet points and list markers (-, *, +, 1.) at line beginnings
    normalized = re.sub(r"^[-*+]\s+", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\d+\.\s+", "", normalized, flags=re.MULTILINE)

    # Collapse multiple whitespaces / newlines to single space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized
