"""
Jupyter Notebook (.ipynb v4) exporter for Rich Assistant sessions.
Converts conversational turns and REPL code blocks into interactive notebook format.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from rai.assistant.records import ConversationTurn


def _extract_code_blocks(text: str) -> List[tuple[str, str]]:
    """Extract code blocks with language from markdown text. Returns [(lang, code)]."""
    pattern = r"```([a-zA-Z0-9_-]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(m[0].lower() or "python", m[1].strip()) for m in matches]


def export_turns_to_notebook(
    session_id: str,
    turns: list[ConversationTurn],
    profile_scope: str = "default",
) -> Dict[str, Any]:
    """
    Export a sequence of conversation turns into standard nbformat v4 dictionary.
    """
    cells: List[Dict[str, Any]] = []

    # Title header cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# RAI Assistant Session: `{session_id}`\n\n",
            f"**Profile:** `{profile_scope}`  \n",
            "This interactive notebook was automatically exported from Rich AI (RAI).\n",
        ],
    })

    execution_count = 1

    for turn in turns:
        role_label = "**User**" if turn.role == "user" else "**Assistant**"
        text = turn.text.strip()
        code_blocks = _extract_code_blocks(text)

        if not code_blocks:
            # Simple text dialog turn -> Markdown cell
            cells.append({
                "cell_type": "markdown",
                "metadata": {"rai_role": turn.role, "turn_id": turn.record_id},
                "source": [f"### {role_label}\n\n", text + "\n"],
            })
        else:
            # Turn with embedded code blocks
            # 1. Preceding text as markdown
            remaining_text = re.sub(r"```[a-zA-Z0-9_-]*\n.*?```", "", text, flags=re.DOTALL).strip()
            if remaining_text:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {"rai_role": turn.role, "turn_id": turn.record_id},
                    "source": [f"### {role_label}\n\n", remaining_text + "\n"],
                })

            # 2. Each code block as a code cell
            for lang, code in code_blocks:
                cell_type = "code" if lang in ("python", "py", "") else "markdown"
                if cell_type == "code":
                    cells.append({
                        "cell_type": "code",
                        "execution_count": execution_count,
                        "metadata": {"language": lang, "turn_id": turn.record_id},
                        "source": [line + "\n" for line in code.splitlines()],
                        "outputs": [],
                    })
                    execution_count += 1
                else:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {"turn_id": turn.record_id},
                        "source": [f"```{lang}\n{code}\n```\n"],
                    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (RAI Sandbox)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14",
            },
            "rai": {
                "session_id": session_id,
                "profile_scope": profile_scope,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
