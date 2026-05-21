"""
Secure Shell command execution tool utilizing bubblewrap namespaces and HITL guardrails.
"""
import os
import logging
from typing import Optional

from rai.tools.security.sandbox import get_sandbox_runner
from rai.tools.security.guardrails import validate_command
from rai.tools.security.hitl import get_approval_manager

logger = logging.getLogger(__name__)

async def run_secure_shell_command(
    command: str,
    allow_network: bool = False,
    rw_dir: Optional[str] = None
) -> str:
    """
    Executes a shell command securely within an isolated and sandboxed workspace.

    Args:
        command (str): The shell command to execute (e.g., 'ls -la', 'echo hello').
        allow_network (bool): Set True to grant the sandbox network/internet access (requires explicit Human-in-the-Loop consent).
        rw_dir (str, optional): Path to a host directory to bind as read-write at '/workspace/output'.
    """
    logger.info("Agent requested command execution: '%s' (allow_network=%s)", command, allow_network)

    # 1. Static analysis guardrails
    is_safe, error_msg = validate_command(command)
    if not is_safe:
        logger.warning("Command rejected by static guardrails: '%s' - Reason: %s", command, error_msg)
        return f"Execution Error: Command was blocked by system safety guardrails. {error_msg}"

    # 2. Human-in-the-Loop policy
    # If the agent requests network access or attempts any high-risk action, it must be approved
    requires_approval = allow_network or ("sudo " in command) or ("chmod " in command)
    if requires_approval:
        logger.info("Command requires explicit human authorization. Suspending coroutine...")
        approval_manager = get_approval_manager()
        req = approval_manager.register_request(command, "ShellTools")
        
        # Suspend executing thread and wait for user consent
        approved = await approval_manager.wait_for_approval(req)
        if not approved:
            logger.warning("Execution denied by user for command: '%s'", command)
            return "Execution Error: Action denied by the user."
        logger.info("Execution approved by user. Resuming...")

    # 3. Spawn sandboxed runner
    # Resolve a clean sandbox working cache dir for outputs
    user_cache = os.path.expanduser("~/.cache/rai/sandbox")
    active_rw_dir = rw_dir or os.path.join(user_cache, "workspace_outputs")
    os.makedirs(active_rw_dir, exist_ok=True)

    runner = get_sandbox_runner()
    # Convert command string to args list for safe subprocess execution without shell expansion on host
    cmd_list = ["/bin/bash", "-c", command]

    result = runner.run(
        cmd_list,
        allow_network=allow_network,
        rw_dir=active_rw_dir
    )

    if result.is_success():
        output = result.stdout if result.stdout else "[Success: Command returned no output]"
        if result.stderr:
            output += f"\n\n[Stderr:\n{result.stderr}]"
        return output
    else:
        return f"Execution Failure (exit code {result.returncode}):\n{result.stderr or result.stdout}"
