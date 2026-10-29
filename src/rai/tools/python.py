"""
Secure Python execution tool utilizing tiered sandboxing:
- Tier-1: Pure Monty Nano-Sandbox (<1 ms, zero-IO, persistent session state).
- Tier-2: Bubblewrap OS namespace container (system imports, network, physical files).
"""
from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from returns.result import Failure, Result, Success

from rai.assistant.records import ObservationArtifact, summarize_observation_schema
from rai.diagnostics import trace, trace_payload
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    _new_id,
    max_data_class,
)
from rai.paths import cache_dir
from rai.tools.security.hitl import get_approval_manager
from rai.tools.security.monty import get_monty_runner
from rai.tools.security.sandbox import get_sandbox_runner

REPL_PRODUCER = ProducerIdentity(
    producer_id="rai.tools.python.repl",
    kind="repl",
    version="1.0.0",
)

if TYPE_CHECKING:
    from rai.assistant.ports import MemoryGraphStore, ObservationStore
    from rai.tools.security.monty import MontyRunner

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
    "__import__",
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

        # 2. Builtin calls and identifiers requiring CPython
        elif isinstance(node, ast.Name):
            if node.id in {
                "__import__", "eval", "exec", "open", "compile",
                "breakpoint", "globals", "locals", "vars",
            }:
                return False, f"Builtin identifier '{node.id}' requires full OS environment."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__", "eval", "exec", "open", "compile",
                "breakpoint", "globals", "locals", "vars",
            }:
                return False, f"Builtin call '{node.func.id}' requires full OS environment."
        elif isinstance(node, ast.Attribute):
            if node.attr in {
                "__subclasses__", "__globals__", "__code__", "__builtins__",
                "__class__", "__base__", "__bases__", "__loader__", "__spec__",
            }:
                return False, f"Reflection attribute '{node.attr}' requires full OS environment."

        # 3. Unsupported syntax constructs in Monty
        elif isinstance(node, ast.ClassDef) and node.bases:
            return False, "Class inheritance requires full CPython."
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            return False, "Generators require full CPython."
        elif isinstance(node, ast.Match):
            return False, "Pattern matching requires full CPython."

    return True, "Compatible with pure Monty sandbox."


_repl_session_locks_guard = threading.Lock()



async def get_session_repl_lock(profile_scope: str, session_id: str) -> asyncio.Lock:
    """
    Acquire or create an asyncio lock for serializing REPL operations per (profile_scope, session_id).
    Locks are attached directly to the active event loop to ensure proper loop-isolation without
    retaining references to closed loops in global registries.
    """
    loop = asyncio.get_running_loop()
    key = (profile_scope, session_id)
    with _repl_session_locks_guard:
        if not hasattr(loop, "_rai_repl_locks"):
            loop._rai_repl_locks = {}
        locks: dict[tuple[str, str], asyncio.Lock] = loop._rai_repl_locks

        if key not in locks:
            locks[key] = asyncio.Lock()
        return locks[key]


async def get_session_blob(
    session_id: str,
    store: Optional[MemoryGraphStore] = None,
    profile_scope: str = "default",
) -> Result[Optional[bytes], ActionFailure]:
    """Retrieve stored Monty session heap snapshot from persistent storage."""
    if store is None:
        from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
        store = SQLiteMemoryGraphStore()
    res = await store.get_repl_session(session_id, profile_scope=profile_scope)
    if isinstance(res, Failure):
        return Failure(res.failure())
    unwrapped = res.unwrap()
    blob = unwrapped[0] if unwrapped is not None else None
    return Success(blob)


async def store_session_blob(
    session_id: str,
    blob: bytes,
    variables: Optional[dict[str, str]] = None,
    store: Optional[MemoryGraphStore] = None,
    profile_scope: str = "default",
) -> Result[None, ActionFailure]:
    """Persist Monty session heap snapshot into persistent storage."""
    if store is None:
        from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
        store = SQLiteMemoryGraphStore()
    return await store.save_repl_session(
        session_id, blob, variables or {}, profile_scope=profile_scope
    )


async def clear_session_blob(
    session_id: str,
    store: Optional[MemoryGraphStore] = None,
    profile_scope: str = "default",
) -> Result[None, ActionFailure]:
    """Clear stored Monty session heap snapshot from persistent storage."""
    lock = await get_session_repl_lock(profile_scope, session_id)
    async with lock:
        if store is None:
            from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415
            store = SQLiteMemoryGraphStore()
        return await store.clear_repl_session(session_id, profile_scope=profile_scope)


@dataclass
class MontyObservationContext:
    """Manages session observation artifact binding, publishing, and data-class propagation for Monty REPL."""

    session_id: Optional[str]
    _current_session_dc: list[DataClass]
    bound_artifacts: dict[str, ObservationArtifact]
    inputs: dict[str, Any]
    external_lookup: dict[str, Any]
    published_artifacts: list[ObservationArtifact]

    @property
    def session_data_class(self) -> DataClass:
        return self._current_session_dc[0]

    def format_feedback(self, variables: Optional[dict[str, str]] = None) -> str:
        parts: list[str] = []
        if variables:
            var_summary = ", ".join(f"{k} ({v})" for k, v in sorted(variables.items()))
            parts.append(f"[Active REPL session state: {var_summary}]")

        available_arts = list(self.bound_artifacts.values())
        if available_arts:
            obs_lines: list[str] = []
            for art in available_arts:
                schema = art.schema_summary
                dc_str = (
                    art.data_class.value
                    if isinstance(art.data_class, DataClass)
                    else str(art.data_class)
                )
                details: list[str] = [
                    f"capability={art.capability}",
                    f"data_class={dc_str}",
                ]
                if schema.total_count is not None:
                    details.append(f"total_count={schema.total_count}")
                if schema.top_level_keys:
                    details.append(f"keys={list(schema.top_level_keys)}")
                if schema.list_fields:
                    details.append(f"lists={dict(schema.list_fields)}")
                obs_lines.append(f"- {art.artifact_id}: {', '.join(details)}")
            obs_summary = "\n".join(obs_lines)
            parts.append(f"[Available session observations:\n{obs_summary}]")

        return "\n\n".join(parts)

    async def persist_published(
        self, store: Optional[Any]
    ) -> Result[None, ActionFailure]:
        if store is None or not self.published_artifacts:
            return Success(None)
        put_fn = getattr(store, "put", None) or getattr(
            store, "put_observation_artifact", None
        )
        if not callable(put_fn):
            return Success(None)

        for art in self.published_artifacts:
            res = put_fn(art)  # pylint: disable=not-callable
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                res = await res
            if isinstance(res, Failure):
                logger.error(
                    "Failed to persist published artifact %s: %s",
                    art.artifact_id,
                    res.failure().message,
                )
                return res
        return Success(None)


async def prepare_monty_observation_context(
    session_id: Optional[str],
    store: Optional[Any] = None,
    default_data_class: DataClass = DataClass.PUBLIC,
    turn_id: Optional[str] = None,
) -> MontyObservationContext:
    """
    Prepares scoped observation bindings, publish callback, and data-class tracking for Monty REPL.
    """
    bound_artifacts: dict[str, ObservationArtifact] = {}
    obs_mapping: dict[Any, Any] = {}
    inputs: dict[str, Any] = {}
    existing_obs_nums: set[int] = set()

    if session_id and store:
        list_fn = getattr(store, "list_for_session", None) or getattr(
            store, "list_observation_artifacts_for_session", None
        )
        if callable(list_fn):
            try:
                res = list_fn(session_id)  # pylint: disable=not-callable
                if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                    list_res = await res
                else:
                    list_res = res
                if isinstance(list_res, Success):
                    for art in list_res.unwrap():
                        alias = art.artifact_id
                        bound_artifacts[alias] = art
                        obs_mapping[alias] = art.content
                        if alias.startswith("obs_"):
                            try:
                                num = int(alias.split("_", 1)[1])
                                existing_obs_nums.add(num)
                                obs_mapping[num] = art.content
                            except (IndexError, ValueError):
                                pass
                        if alias.isidentifier():
                            inputs[alias] = art.content
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to list observations for session %s: %s", session_id, exc)

    inputs["obs"] = obs_mapping

    initial_session_dc = max_data_class(
        default_data_class, *(art.data_class for art in bound_artifacts.values())
    )
    current_session_dc = [initial_session_dc]

    curr_num = 1
    while curr_num in existing_obs_nums:
        curr_num += 1
    next_index = [curr_num]

    published_artifacts: list[ObservationArtifact] = []

    def publish(
        value: Any,
        name: Optional[str] = None,
        data_class: Optional[Union[DataClass, str]] = None,
    ) -> str:
        if name and str(name).strip():
            art_id = str(name).strip()
        else:
            while next_index[0] in existing_obs_nums:
                next_index[0] += 1
            art_id = f"obs_{next_index[0]}"
            existing_obs_nums.add(next_index[0])
            next_index[0] += 1

        if isinstance(value, dict):
            content = value
        elif isinstance(value, (list, tuple)):
            content = {"items": list(value), "total_count": len(value)}
        else:
            content = {"value": value}

        effective_dc = max_data_class(current_session_dc[0], data_class)
        current_session_dc[0] = effective_dc

        schema = summarize_observation_schema(content)
        try:
            size_chars = len(json.dumps(content, ensure_ascii=False))
        except Exception:
            size_chars = len(str(content))

        derived_ids = tuple(sorted(bound_artifacts.keys()))

        new_art = ObservationArtifact(
            record_id=_new_id(),
            artifact_id=art_id,
            session_id=session_id or "anonymous",
            turn_id=turn_id or "repl",
            capability="python_repl",
            request_id=f"repl_{session_id or 'anon'}_{_new_id()[:8]}",
            data_class=effective_dc,
            content=content,
            schema_summary=schema,
            size_chars=size_chars,
            derived_from=derived_ids,
            producer=REPL_PRODUCER,
        )

        published_artifacts.append(new_art)
        bound_artifacts[art_id] = new_art
        obs_mapping[art_id] = content
        if art_id.startswith("obs_"):
            try:
                num = int(art_id.split("_", 1)[1])
                obs_mapping[num] = content
            except (IndexError, ValueError):
                pass
        if art_id.isidentifier():
            inputs[art_id] = content

        return art_id

    external_lookup = {"publish": publish}

    return MontyObservationContext(
        session_id=session_id,
        _current_session_dc=current_session_dc,
        bound_artifacts=bound_artifacts,
        inputs=inputs,
        external_lookup=external_lookup,
        published_artifacts=published_artifacts,
    )


async def run_secure_python_code(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915
    code: str,
    allow_network: bool = False,
    rw_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    force_os_sandbox: bool = False,
    store: Optional[Union[MemoryGraphStore, ObservationStore]] = None,
    monty_runner: Optional[MontyRunner] = None,
    profile_scope: str = "default",
    require_hitl: bool = True,
    turn_id: Optional[str] = None,
    data_class: Optional[DataClass] = None,
) -> str:
    """
    Executes Python code securely via Tiered Sandboxing.
    Pure/algorithmic code executes immediately in Monty without HITL interruption.
    OS-level actions escalate to Bubblewrap under HITL authorization.
    """
    logger.info(
        "Agent requested python execution (allow_network=%s, session_id=%s, force_os_sandbox=%s, profile_scope=%s)",
        allow_network, session_id, force_os_sandbox, profile_scope,
    )

    if monty_runner is None:
        monty_runner = get_monty_runner()
    monty_eligible, monty_reason = can_run_in_monty(code)

    ctx = await prepare_monty_observation_context(
        session_id=session_id,
        store=store,
        default_data_class=data_class or DataClass.PUBLIC,
        turn_id=turn_id,
    )

    # 1. Tier-1 Fast-Path: Pure Monty execution (No HITL needed)
    if not force_os_sandbox and not allow_network and monty_eligible and monty_runner.is_available():
        logger.debug("Executing code via Tier-1 Monty nano-sandbox (reason: %s)", monty_reason)

        async def _run_monty_step() -> str:
            prior_blob: Optional[bytes] = None
            if session_id:
                blob_res = await get_session_blob(
                    session_id, store=store, profile_scope=profile_scope
                )
                if isinstance(blob_res, Failure):
                    err = blob_res.failure()
                    logger.error(
                        "Failed to retrieve session blob for %s: %s", session_id, err.message
                    )
                    return f"Execution Failure:\nFailed to load REPL session state: {err.message}"
                prior_blob = blob_res.unwrap()

            res = await asyncio.to_thread(
                monty_runner.run,
                code,
                session_blob=prior_blob,
                inputs=ctx.inputs,
                external_lookup=ctx.external_lookup,
            )

            if isinstance(res, Success):
                monty_result = res.unwrap()
                if session_id and monty_result.session_blob:
                    save_res = await store_session_blob(
                        session_id,
                        monty_result.session_blob,
                        variables=monty_result.variables,
                        store=store,
                        profile_scope=profile_scope,
                    )
                    if isinstance(save_res, Failure):
                        err = save_res.failure()
                        logger.error(
                            "Failed to persist session blob for %s: %s", session_id, err.message
                        )
                        return f"Execution Failure:\nFailed to persist REPL session state: {err.message}"

                if ctx.published_artifacts and store:
                    persist_res = await ctx.persist_published(store)
                    if isinstance(persist_res, Failure):
                        err = persist_res.failure()
                        logger.error(
                            "Failed to persist published artifacts: %s", err.message
                        )

                output = monty_result.output if monty_result.output else "[Success: Code executed and returned no output]"

                # State-Aware Feedback: append concise variable state and observation inventory
                feedback = ctx.format_feedback(monty_result.variables)
                if monty_result.variables:
                    var_summary = ", ".join(f"{k} ({v})" for k, v in sorted(monty_result.variables.items()))
                    trace("capability.repl_state", session_id=session_id or "none", variables=var_summary)
                    trace_payload("capability.repl_state", monty_result.variables, session_id=session_id or "none")
                else:
                    trace("capability.repl_state", session_id=session_id or "none", variables="empty")

                if feedback:
                    output += f"\n\n{feedback}"

                return output
            else:
                return f"Execution Failure:\n{res.failure()}"

        if session_id:
            lock = await get_session_repl_lock(profile_scope, session_id)
            async with lock:
                return await _run_monty_step()
        return await _run_monty_step()

    # 2. Tier-2 Escalation: Full OS execution in Bubblewrap
    logger.info("Escalating Python execution to Tier-2 Bubblewrap runner.")

    if allow_network and ctx.session_data_class == DataClass.LOCAL:
        return "Execution Error: Network access is prohibited for LOCAL data class."

    if ctx.session_data_class == DataClass.PRIVATE:
        if allow_network and not require_hitl:
            return "Execution Error: Egress of PRIVATE data requires explicit user approval."
        require_hitl = True

    if require_hitl:
        # Human-in-the-Loop policy check: all CPython executions require explicit human authorization
        logger.info("Python script in Tier-2 OS environment requires explicit human authorization. Suspending coroutine...")
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

    result = await asyncio.to_thread(
        runner.run,
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
