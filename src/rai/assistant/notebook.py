"""
Jupyter Notebook (.ipynb v4) exporter for Rich Assistant sessions.
Converts conversational turns and REPL code blocks into interactive notebook format.
"""
from __future__ import annotations

import re
import sys
import uuid
from typing import Any, Dict, List

from rai.assistant.records import ConversationTurn


def _generate_cell_id() -> str:
    """Generate a unique 8-character cell ID compliant with nbformat 4.5+."""
    return uuid.uuid4().hex[:8]


def _extract_code_blocks(text: str) -> List[tuple[str, str]]:
    """Extract code blocks with language from markdown text. Returns [(lang, code)]."""
    pattern = r"```([a-zA-Z0-9_-]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(m[0].lower(), m[1].strip()) for m in matches]


def export_turns_to_notebook(
    session_id: str,
    turns: list[ConversationTurn],
    profile_scope: str = "default",
) -> Dict[str, Any]:
    """
    Export a sequence of conversation turns into standard nbformat v4 dictionary.
    Fully compliant with Jupyter Notebook v4.5 schema.
    """
    cells: List[Dict[str, Any]] = []

    # Title header cell
    cells.append({
        "cell_type": "markdown",
        "id": _generate_cell_id(),
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
                "id": _generate_cell_id(),
                "metadata": {"rai_role": turn.role, "turn_id": turn.record_id},
                "source": [f"### {role_label}\n\n", text + "\n"],
            })
        else:
            # Turn with embedded code blocks
            # 1. Preceding text as markdown (excluding Output: label if present)
            remaining_text = re.sub(r"```[a-zA-Z0-9_-]*\n.*?```", "", text, flags=re.DOTALL)
            clean_text = "\n".join(
                line for line in remaining_text.splitlines() if line.strip() != "Output:"
            ).strip()
            if clean_text:
                cells.append({
                    "cell_type": "markdown",
                    "id": _generate_cell_id(),
                    "metadata": {"rai_role": turn.role, "turn_id": turn.record_id},
                    "source": [f"### {role_label}\n\n", clean_text + "\n"],
                })

            # Check if there is an associated output block (unlabeled code block following python)
            py_blocks: List[str] = [code for lang, code in code_blocks if lang in ("python", "py")]
            out_blocks: List[str] = [code for lang, code in code_blocks if lang == ""]

            if py_blocks:
                for idx, code in enumerate(py_blocks):
                    outputs: List[Dict[str, Any]] = []
                    if idx == len(py_blocks) - 1 and out_blocks:
                        out_text = "\n\n".join(out_blocks)
                        outputs.append({
                            "output_type": "stream",
                            "name": "stdout",
                            "text": [line + "\n" for line in out_text.splitlines()],
                        })

                    cells.append({
                        "cell_type": "code",
                        "id": _generate_cell_id(),
                        "execution_count": execution_count,
                        "metadata": {"language": "python", "turn_id": turn.record_id},
                        "source": [line + "\n" for line in code.splitlines()],
                        "outputs": outputs,
                    })
                    execution_count += 1
            else:
                for lang, code in code_blocks:
                    cells.append({
                        "cell_type": "markdown",
                        "id": _generate_cell_id(),
                        "metadata": {"turn_id": turn.record_id},
                        "source": [f"```{lang}\n{code}\n```\n"],
                    })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
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
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "rai": {
                "session_id": session_id,
                "profile_scope": profile_scope,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
