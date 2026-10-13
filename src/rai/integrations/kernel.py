"""
Jupyter Kernel implementation backed by Pydantic Monty runner.
Executes notebook cells inside the safe persistent nano-sandbox.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional
from returns.result import Success

from rai.tools.security.monty import MontyRunner, get_monty_runner

try:
    from ipykernel.kernelbase import Kernel
except ImportError:  # pragma: no cover
    # Minimal fallback protocol if ipykernel is not installed
    class Kernel:  # type: ignore[no-redef]
        implementation = "MontyKernel"
        implementation_version = "1.0"
        language = "python"
        language_version = "3.12"
        language_info = {
            "name": "python",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        }
        banner = "Monty Python REPL Kernel"

        def send_response(self, *args: object, **kwargs: object) -> None:
            pass


class MontyJupyterKernel(Kernel):
    """Jupyter kernel executing Python code via persistent Monty sandboxing."""

    implementation = "MontyKernel"
    implementation_version = "0.1.0"
    language = "python"
    language_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    language_info = {
        "name": "python",
        "mimetype": "text/x-python",
        "file_extension": ".py",
        "pygments_lexer": "ipython3",
    }
    banner = "Rich AI - Monty Persistent Sandbox Kernel"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.runner: MontyRunner = get_monty_runner()
        self.session_blob: Optional[bytes] = None
        self.variables: Dict[str, str] = {}
        self.iopub_socket: Any = None

    def do_execute(  # noqa: PLR0913
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: bool = False,
    ) -> Dict[str, Any]:
        """Execute notebook cell inside Monty persistent session."""
        if not code.strip():
            return {
                "status": "ok",
                "execution_count": getattr(self, "execution_count", 1),
                "payload": [],
                "user_expressions": {},
            }

        res = self.runner.run(code, session_blob=self.session_blob)

        if isinstance(res, Success):
            exec_res = res.unwrap()
            self.session_blob = exec_res.session_blob
            self.variables = exec_res.variables

            if not silent and exec_res.output:
                stream_content = {"name": "stdout", "text": exec_res.output + "\n"}
                self.send_response(self.iopub_socket, "stream", stream_content)

            return {
                "status": "ok",
                "execution_count": getattr(self, "execution_count", 1),
                "payload": [],
                "user_expressions": {},
            }
        else:
            err_msg = res.failure()
            if not silent:
                err_content = {
                    "ename": "MontyExecutionError",
                    "evalue": err_msg,
                    "traceback": [f"MontyExecutionError: {err_msg}"],
                }
                self.send_response(self.iopub_socket, "error", err_content)

            return {
                "status": "error",
                "execution_count": getattr(self, "execution_count", 1),
                "ename": "MontyExecutionError",
                "evalue": err_msg,
                "traceback": [f"MontyExecutionError: {err_msg}"],
            }


if __name__ == "__main__":
    try:
        from ipykernel.kernelapp import IPKernelApp

        IPKernelApp.launch_instance(kernel_class=MontyJupyterKernel)
    except ImportError:
        print(
            "ipykernel is required to launch MontyJupyterKernel standalone. "
            "Install with: uv add --extra jupyter ipykernel"
        )
