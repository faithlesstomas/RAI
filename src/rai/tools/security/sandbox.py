"""
Sandboxing runner implementations for executing shell commands and scripts in isolated environments.
"""
import os
import shutil
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SandboxResult:
    """Represents the outcome of a sandboxed execution."""
    def __init__(self, returncode: int, stdout: str, stderr: str, sandbox_type: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.sandbox_type = sandbox_type

    def is_success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "sandbox_type": self.sandbox_type,
            "success": self.is_success()
        }


class SandboxRunner(ABC):
    """Abstract base class for executing commands inside an isolated sandbox."""

    @abstractmethod
    def run(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        allow_network: bool = False,
        rw_dir: Optional[str] = None
    ) -> SandboxResult:
        """Executes a command inside the sandbox."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this sandbox implementation can be used."""
        pass


class UnavailableSandboxRunner(SandboxRunner):
    """Fail-closed runner used when no supported sandbox is available."""

    def run(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        allow_network: bool = False,
        rw_dir: Optional[str] = None,
    ) -> SandboxResult:
        del command, env, allow_network, rw_dir
        return SandboxResult(
            -1,
            "",
            "No supported sandbox is available; host execution was refused.",
            "unavailable",
        )

    def is_available(self) -> bool:
        return False


class BubblewrapRunner(SandboxRunner):
    """
    Executes commands inside a bubblewrap (bwrap) secure namespace container.
    """

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        self.bwrap_path = shutil.which("bwrap")
        self.workspace_dir = workspace_dir or os.getcwd()
        self._usable: Optional[bool] = None

    def is_available(self) -> bool:
        if self.bwrap_path is None:
            return False
        if self._usable is None:
            try:
                probe = subprocess.run(
                    [
                        self.bwrap_path,
                        "--unshare-user",
                        "--ro-bind",
                        "/",
                        "/",
                        "--",
                        "/bin/true",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3,
                )
                self._usable = probe.returncode == 0
            except (OSError, subprocess.SubprocessError):
                self._usable = False
        return self._usable

    def run(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        allow_network: bool = False,
        rw_dir: Optional[str] = None
    ) -> SandboxResult:
        if not self.is_available():
            return UnavailableSandboxRunner().run(command, env, allow_network, rw_dir)

        bwrap_cmd = [self.bwrap_path]

        # 1. Unshare namespaces for isolation
        # --unshare-all unshares user, IPC, PID, UTS, and mount namespaces.
        # It also unshares network by default unless --share-net is supplied.
        bwrap_cmd.append("--unshare-all")
        if allow_network:
            bwrap_cmd.append("--share-net")

        # 2. Mount proc and dev filesystem
        bwrap_cmd.extend(["--proc", "/proc"])
        bwrap_cmd.extend(["--dev", "/dev"])

        # 3. Mount core system directories read-only dynamically
        # Different distros handle usr-merge or symlinks differently.
        core_mounts = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"]
        for m in core_mounts:
            if os.path.exists(m):
                bwrap_cmd.extend(["--ro-bind", m, m])

        # 4. Mount isolated volatile storage for /tmp and /run
        bwrap_cmd.extend(["--tmpfs", "/tmp"])
        bwrap_cmd.extend(["--tmpfs", "/run"])

        # 5. Bind workspace directory
        # Mount host workspace as read-only to avoid unauthorized modifications
        sandbox_workspace = "/workspace"
        bwrap_cmd.extend(["--ro-bind", self.workspace_dir, sandbox_workspace])

        # 6. Mount dedicated read-write directory if provided
        # If the tool needs to write files, we isolate the writes to a clean dir.
        if rw_dir:
            os.makedirs(rw_dir, exist_ok=True)
            # Bind mount the host read-write dir to /output
            bwrap_cmd.extend(["--dir", "/output"])
            bwrap_cmd.extend(["--bind", rw_dir, "/output"])
        else:
            # Fallback output tmpfs inside container
            bwrap_cmd.extend(["--dir", "/output"])
            bwrap_cmd.extend(["--tmpfs", "/output"])

        # 7. Set working directory inside sandbox
        bwrap_cmd.extend(["--chdir", sandbox_workspace])

        # 8. Setup basic environment variables
        bwrap_env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/tmp",
            "TERM": "xterm-256color"
        }
        if env:
            bwrap_env.update(env)

        # 9. Format execution args to bwrap
        # Command must be executed via shell inside container if it needs shell expansion,
        # or executed directly.
        bwrap_cmd.extend(command)

        logger.debug("Executing bubblewrap command: %s", " ".join(bwrap_cmd))
        try:
            res = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                env=bwrap_env,
                check=False
            )
            return SandboxResult(res.returncode, res.stdout, res.stderr, "bubblewrap")
        except Exception as e:
            logger.error("Bubblewrap sandbox execution crashed: %s", e)
            return SandboxResult(-1, "", f"Sandbox Error: {e}", "bubblewrap")


class GuixContainerRunner(SandboxRunner):
    """
    Executes commands inside a reproducible GNU Guix container.
    """

    def __init__(self) -> None:
        self.guix_path = shutil.which("guix")

    def is_available(self) -> bool:
        return self.guix_path is not None

    def run(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        allow_network: bool = False,
        rw_dir: Optional[str] = None
    ) -> SandboxResult:
        if not self.is_available():
            return UnavailableSandboxRunner().run(command, env, allow_network, rw_dir)

        guix_cmd = [self.guix_path, "shell", "--container"]
        
        # Include minimal core packages for container utilities
        guix_cmd.extend(["coreutils", "bash"])

        if allow_network:
            guix_cmd.append("--network")

        # Bind read-write directory
        if rw_dir:
            os.makedirs(rw_dir, exist_ok=True)
            guix_cmd.extend([f"--share={rw_dir}=/output"])

        # Match the Bubblewrap contract: source workspace is visible read-only.
        workspace_dir = os.getcwd()
        guix_cmd.extend([f"--expose={workspace_dir}=/workspace"])

        # command execution
        guix_cmd.append("--")
        guix_cmd.extend(command)

        logger.debug("Executing Guix container command: %s", " ".join(guix_cmd))
        try:
            res = subprocess.run(
                guix_cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False
            )
            return SandboxResult(res.returncode, res.stdout, res.stderr, "guix")
        except Exception as e:
            logger.error("Guix container sandbox execution crashed: %s", e)
            return SandboxResult(-1, "", f"Sandbox Error: {e}", "guix")


def get_sandbox_runner(workspace_dir: Optional[str] = None) -> SandboxRunner:
    """
    Factory function returning the best available sandbox runner.
    Prioritizes bubblewrap, then guix, then returns a dummy fallback if none are found.
    """
    bwrap = BubblewrapRunner(workspace_dir)
    if bwrap.is_available():
        return bwrap

    guix = GuixContainerRunner()
    if guix.is_available():
        return guix

    logger.error("No sandboxing tools (bubblewrap or Guix) are available; execution is disabled.")
    return UnavailableSandboxRunner()
