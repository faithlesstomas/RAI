"""
Secure Python execution tool utilizing bubblewrap namespaces and HITL guardrails.
"""
import os
import logging
from typing import Optional

from rai.tools.security.sandbox import get_sandbox_runner
from rai.tools.security.hitl import get_approval_manager

logger = logging.getLogger(__name__)

async def run_secure_python_code(
    code: str,
    allow_network: bool = False,
    rw_dir: Optional[str] = None
) -> str:
    """
    Executes Python code securely within an isolated and sandboxed workspace.

    Args:
        code (str): The Python code snippet to execute.
        allow_network (bool): Set True to grant the sandbox network/internet access (requires explicit Human-in-the-Loop consent).
        rw_dir (str, optional): Path to a host directory to bind as read-write at '/workspace/output'.
    """
    logger.info("Agent requested python execution (allow_network=%s)", allow_network)

    # 1. Human-in-the-Loop policy
    # If the agent requests network access or attempts high-risk actions, it must be approved
    # We scan for standard library imports or keywords that indicate system interaction
    risky_keywords = [
        "import os", "import subprocess", "import sys", "import socket",
        "import urllib", "import requests", "import shutil", "eval(", "exec(", "open("
    ]
    requires_approval = allow_network or any(kw in code for kw in risky_keywords)

    if requires_approval:
        logger.info("Python script requires explicit human authorization. Suspending coroutine...")
        approval_manager = get_approval_manager()
        req = approval_manager.register_request(code, "PythonTools")
        
        # Suspend executing thread and wait for user consent
        approved = await approval_manager.wait_for_approval(req)
        if not approved:
            logger.warning("Execution denied by user for Python code.")
            return "Execution Error: Action denied by the user."
        logger.info("Execution approved by user. Resuming...")

    # 2. Setup the sandbox read-write working directory
    user_cache = os.path.expanduser("~/.cache/rai/sandbox")
    active_rw_dir = rw_dir or os.path.join(user_cache, "workspace_outputs")
    os.makedirs(active_rw_dir, exist_ok=True)

    # Write the Python code to a file inside the read-write workspace
    script_host_path = os.path.join(active_rw_dir, "script.py")
    try:
        with open(script_host_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        logger.error("Failed to write python script to host path '%s': %s", script_host_path, e)
        return f"Execution Error: Failed to setup script environment. {e}"

    # 3. Spawn sandboxed runner
    runner = get_sandbox_runner()
    # The script is mounted under /output/script.py
    cmd_list = ["python3", "/output/script.py"]

    result = runner.run(
        cmd_list,
        allow_network=allow_network,
        rw_dir=active_rw_dir
    )

    # Clean up the script file on the host after execution
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
