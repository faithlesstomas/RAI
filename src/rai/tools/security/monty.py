"""
Monty-based pure execution runner and REPL session manager.
Provides sub-millisecond, memory-safe, isolated Python execution with heap snapshot persistence.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from returns.result import Failure, Result, Success

logger = logging.getLogger(__name__)

try:
    from pydantic_monty import Monty, MountDir, OSAccess, MemoryFile
    _MONTY_AVAILABLE = True
except (ImportError, OSError) as _exc:  # pragma: no cover
    logger.warning("Pydantic Monty is not available: %s", _exc)
    _MONTY_AVAILABLE = False
    Monty = None  # type: ignore[misc, assignment]
    MountDir = None  # type: ignore[misc, assignment]
    OSAccess = None  # type: ignore[misc, assignment]
    MemoryFile = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class MontyExecutionResult:
    """Represents the terminal outcome of a Monty execution step."""
    output: str
    session_blob: bytes
    variables: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    return_value: Any = None


class MontyRunner:
    """
    Executes sandboxed Python snippets inside a lightweight Monty interpreter worker.
    Manages session pooling, memory quotas, execution timeouts, and state snapshots.
    """

    def __init__(
        self,
        max_memory_bytes: int = 32 * 1024 * 1024,
        max_duration_secs: float = 2.0,
    ) -> None:
        self.max_memory_bytes = max_memory_bytes
        self.max_duration_secs = max_duration_secs
        self._pool: Optional[Any] = None

    def is_available(self) -> bool:
        """Check if Pydantic Monty is installed and usable."""
        return _MONTY_AVAILABLE

    def _get_pool(self) -> Any:  # noqa: ANN401
        if not self.is_available():
            raise RuntimeError("Pydantic Monty runtime is not available.")
        if self._pool is None:
            pool = Monty()
            pool.__enter__()
            self._pool = pool
        return self._pool

    def close(self) -> None:
        """Release worker pool resources."""
        if self._pool is not None:
            try:
                self._pool.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                logger.debug("Error while closing Monty pool: %s", exc)
            self._pool = None

    def __enter__(self) -> MontyRunner:
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        self.close()

    def run(
        self,
        code: str,
        session_blob: Optional[bytes] = None,
        inputs: Optional[dict[str, Any]] = None,
    ) -> Result[MontyExecutionResult, str]:
        """
        Executes a Python snippet in an isolated Monty worker.
        If `session_blob` is supplied, restores prior heap state before evaluation.
        Dumps the updated heap state into `session_blob` upon completion.
        """
        if not self.is_available():
            return Failure("MONTY_UNAVAILABLE")

        clean_code = code.strip()
        if not clean_code:
            return Failure("EMPTY_CODE")

        start_time = time.perf_counter()
        pool = self._get_pool()

        limits = {
            "max_memory": self.max_memory_bytes,
            "max_feed_duration_secs": self.max_duration_secs,
        }
        try:
            with pool.checkout(
                limits=limits,
            ) as session:
                if session_blob:
                    session.load_session(session_blob)

                raw_output = session.feed_run(clean_code, inputs=inputs or {})
                new_blob = session.dump()

                # Extract variable types for active session state feedback
                variables_dict: dict[str, str] = {}
                try:
                    inspect_code = (
                        "{k: type(v).__name__ for k, v in locals().items() "
                        "if not k.startswith('_')}"
                    )
                    inspected = session.feed_run(inspect_code)
                    if isinstance(inspected, dict):
                        variables_dict = {str(k): str(v) for k, v in inspected.items()}
                        # The inspection snippet itself shouldn't leak into stored snapshot
                        new_blob = session.dump()
                except Exception as inspect_exc:
                    logger.debug("Failed to inspect variables in Monty session: %s", inspect_exc)

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                out_str = str(raw_output) if raw_output is not None else ""

                return Success(
                    MontyExecutionResult(
                        output=out_str,
                        session_blob=new_blob,
                        variables=variables_dict,
                        duration_ms=round(elapsed_ms, 2),
                        return_value=raw_output,
                    )
                )
        except Exception as exc:
            err_msg = str(exc)
            logger.warning("Monty execution error: %s", err_msg)
            return Failure(f"MONTY_EXECUTION_ERROR: {err_msg}")

    def inspect_state(self, session_blob: bytes) -> Result[dict[str, str], str]:
        """Inspect variables and functions stored in a session snapshot without mutating it."""
        if not self.is_available():
            return Failure("MONTY_UNAVAILABLE")
        if not session_blob:
            return Success({})
        pool = self._get_pool()
        try:
            with pool.checkout(
                limits={
                    "max_memory": self.max_memory_bytes,
                    "max_feed_duration_secs": self.max_duration_secs,
                }
            ) as session:
                session.load_session(session_blob)
                inspect_code = (
                    "{k: type(v).__name__ for k, v in locals().items() "
                    "if not k.startswith('_')}"
                )
                inspected = session.feed_run(inspect_code)
                if isinstance(inspected, dict):
                    return Success({str(k): str(v) for k, v in inspected.items()})
                return Success({})
        except Exception as exc:
            return Failure(f"INSPECT_ERROR: {exc}")


_RUNNER_INSTANCE: Optional[MontyRunner] = None


def get_monty_runner() -> MontyRunner:
    """Return the singleton instance of MontyRunner."""
    global _RUNNER_INSTANCE  # noqa: PLW0603
    if _RUNNER_INSTANCE is None:
        _RUNNER_INSTANCE = MontyRunner()
    return _RUNNER_INSTANCE
