"""
Secure Python execution tool utilizing tiered sandboxing:
- Tier-1: Pure Monty Nano-Sandbox (<1 ms, zero-IO, persistent session state).
- Tier-2: Bubblewrap OS namespace container (system imports, network, physical files).
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, Optional, Tuple

from returns.result import Success

from rai.paths import cache_dir
from rai.tools.security.hitl import get_approval_manager
from rai.tools.security.monty import get_monty_runner
from rai.tools.security.sandbox import get_sandbox_runner

logger = logging.getLogger(__name__)

# Modules supported natively by the pure Monty sandbox
_MONTY_SAFE_MODULES = {
    "math", "re", "json", "datetime", "hashlib", "itertools",
    "collections", "random", "pathlib", "base64", "typing",
    "copy", "dataclasses", "functools", "asyncio", "unicodedata",
    "posixpath", "binascii",
}

# Keywords indicating need for full OS environment and HITL consent
_RISKY_KEYWORDS = [
    "import os", "import subprocess", "import sys", "import socket",
    "import urllib", "import requests", "import shutil", "eval(", "exec(", "open(",
]


def can_run_in_monty(code: str) -> Tuple[bool, str]:  # noqa: PLR0911, PLR0912
    """
    Analyzes Python AST to determine if code can execute within pure Monty sandbox.
    Returns (is_compatible, reason).
    """
    clean_code = code.strip()
    if not clean_code:
        return False, "Empty code."

    # Immediate rejection if risky system keywords are present
    if any(kw in clean_code for kw in _RISKY_KEYWORDS):
        return False, "Code requests system-level operations or sensitive builtins."

    try:
        tree = ast.parse(clean_code)
    except SyntaxError:
        return False, "Code failed AST parsing."

    for node in ast.walk(tree):
        # 1. Module imports check
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in _MONTY_SAFE_MODULES:
                    return False, f"Module '{alias.name}' requires full OS environment."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in _MONTY_SAFE_MODULES:
                    return False, f"Module '{node.module}' requires full OS environment."

        # 2. Unsupported syntax constructs in Monty
        elif isinstance(node, ast.ClassDef) and node.bases:
            return False, "Class inheritance requires full CPython."
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            return False, "Generators require full CPython."
        elif isinstance(node, ast.Match):
            return False, "Pattern matching requires full CPython."

    return True, "Compatible with pure Monty sandbox."


async def get_session_blob(
    session_id: str, store: Optional[Any] = None
) -> Optional[bytes]:
    """Retrieve stored Monty session heap snapshot from persistent storage."""
    if store is None:
        from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
        store = SQLiteMemoryGraphStore()
    res = await store.get_repl_session(session_id)
    if isinstance(res, Success) and res.unwrap():
        blob, _ = res.unwrap()
        return blob
    return None


async def store_session_blob(
    session_id: str,
    blob: bytes,
    variables: Optional[dict[str, str]] = None,
    store: Optional[Any] = None,
) -> None:
    """Persist Monty session heap snapshot into persistent storage."""
    if store is None:
        from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
        store = SQLiteMemoryGraphStore()
    await store.save_repl_session(session_id, blob, variables or {})


async def clear_session_blob(
    session_id: str, store: Optional[Any] = None
) -> None:
    """Clear stored Monty session heap snapshot from persistent storage."""
    if store is None:
        from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
        store = SQLiteMemoryGraphStore()
    await store.clear_repl_session(session_id)


async def run_secure_python_code(  # noqa: PLR0912, PLR0913, PLR0915
    code: str,
    allow_network: bool = False,
    rw_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    force_os_sandbox: bool = False,
    store: Optional[Any] = None,
    monty_runner: Optional[Any] = None,
) -> str:
    """
    Executes Python code securely via Tiered Sandboxing.
    Pure/algorithmic code executes immediately in Monty without HITL interruption.
    OS-level actions escalate to Bubblewrap under HITL authorization.
    """
    logger.info(
        "Agent requested python execution (allow_network=%s, session_id=%s, force_os_sandbox=%s)",
        allow_network, session_id, force_os_sandbox,
    )

    if monty_runner is None:
        monty_runner = get_monty_runner()
    monty_eligible, monty_reason = can_run_in_monty(code)

    # 1. Tier-1 Fast-Path: Pure Monty execution (No HITL needed)
    if not force_os_sandbox and not allow_network and monty_eligible and monty_runner.is_available():
        logger.debug("Executing code via Tier-1 Monty nano-sandbox (reason: %s)", monty_reason)
        prior_blob: Optional[bytes] = None
        if session_id:
            prior_blob = await get_session_blob(session_id, store=store)
        res = monty_runner.run(code, session_blob=prior_blob)

        if isinstance(res, Success):
            monty_result = res.unwrap()
            if session_id and monty_result.session_blob:
                await store_session_blob(
                    session_id,
                    monty_result.session_blob,
                    variables=monty_result.variables,
                    store=store,
                )

            output = monty_result.output if monty_result.output else "[Success: Code executed and returned no output]"

            # State-Aware Feedback: append concise variable state
            if monty_result.variables:
                var_summary = ", ".join(f"{k} ({v})" for k, v in sorted(monty_result.variables.items()))
                output += f"\n\n[Active REPL session state: {var_summary}]"

            return output
        else:
            return f"Execution Failure:\n{res.failure()}"

    # 2. Tier-2 Escalation: Full OS execution in Bubblewrap
    logger.info("Escalating Python execution to Tier-2 Bubblewrap runner.")

    # Human-in-the-Loop policy check
    requires_approval = allow_network or any(kw in code for kw in _RISKY_KEYWORDS)
    if requires_approval:
        logger.info("Python script requires explicit human authorization. Suspending coroutine...")
        approval_manager = get_approval_manager()
        req = approval_manager.register_request(code, "PythonTools")

        approved = await approval_manager.wait_for_approval(req)
        if not approved:
            logger.warning("Execution denied by user for Python code.")
            return "Execution Error: Action denied by the user."
        logger.info("Execution approved by user. Resuming...")

    # Setup the sandbox read-write working directory
    user_cache = str(cache_dir() / "sandbox")
    active_rw_dir = rw_dir or os.path.join(user_cache, "workspace_outputs")
    os.makedirs(active_rw_dir, exist_ok=True)

    script_host_path = os.path.join(active_rw_dir, "script.py")
    try:
        with open(script_host_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        logger.error("Failed to write python script to host path '%s': %s", script_host_path, e)
        return f"Execution Error: Failed to setup script environment. {e}"

    runner = get_sandbox_runner()
    cmd_list = ["python3", "/output/script.py"]

    result = runner.run(
        cmd_list,
        allow_network=allow_network,
        rw_dir=active_rw_dir,
    )

    if os.path.exists(script_host_path):
        try:
            os.remove(script_host_path)
        except Exception as e:
            logger.warning("Failed to clean up script file '%s': %s", script_host_path, e)

    if result.is_success():
        output = result.stdout if result.stdout else "[Success: Code executed and returned no output]"
        if result.stderr:
            output += f"\n\n[Stderr:\n{result.stderr}]"
        return output
    else:
        return f"Execution Failure (exit code {result.returncode}):\n{result.stderr or result.stdout}"
