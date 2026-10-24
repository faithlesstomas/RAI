"""
Unit and integration tests for secure shell and python tools.
"""
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Failure, Success

from rai.assistant.records import make_assistant_failure
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.records import PolicyOutcome, RiskClass
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.tools.python import clear_session_blob, run_secure_python_code
from rai.tools.security.sandbox import SandboxResult
from rai.tools.shell import run_secure_shell_command


@pytest.mark.asyncio
async def test_run_secure_shell_command_success() -> None:
    """Test that a simple safe shell command executes successfully."""
    # We patch the runner to return a successful mock result so we don't rely on bwrap in this unit test
    mock_result = SandboxResult(returncode=0, stdout="hello world\n", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.shell.get_sandbox_runner") as mock_get_runner:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner
        
        output = await run_secure_shell_command("echo 'hello world'", allow_network=False)
        
        assert "hello world" in output
        mock_runner.run.assert_called_once()
        # Verify bwrap execution arguments
        cmd_args = mock_runner.run.call_args[0][0]
        assert cmd_args == ["/bin/bash", "-c", "echo 'hello world'"]


@pytest.mark.asyncio
async def test_run_secure_shell_command_blocked_guardrails() -> None:
    """Test that a dangerous command is blocked by the guardrails before running."""
    # Attempt a fork bomb
    output = await run_secure_shell_command(":(){ :|:& };:")
    assert "blocked by system safety guardrails" in output


@pytest.mark.asyncio
async def test_run_secure_shell_command_hitl_approved() -> None:
    """Test that a shell command requiring HITL consent executes if approved."""
    mock_result = SandboxResult(returncode=0, stdout="root access verified", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.shell.get_sandbox_runner") as mock_get_runner, \
         patch("rai.tools.shell.get_approval_manager") as mock_get_approval:
        
        # Mock runner
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner
        
        # Mock approval manager to approve the request
        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=True)
        mock_get_approval.return_value = mock_approval
        
        output = await run_secure_shell_command("sudo whoami")
        
        assert "root access verified" in output
        mock_approval.register_request.assert_called_once_with("sudo whoami", "ShellTools")
        mock_approval.wait_for_approval.assert_called_once()
        mock_runner.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_secure_shell_command_hitl_denied() -> None:
    """Test that a shell command requiring HITL consent fails if denied."""
    with patch("rai.tools.shell.get_approval_manager") as mock_get_approval:
        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=False)
        mock_get_approval.return_value = mock_approval
        
        output = await run_secure_shell_command("sudo whoami")
        
        assert "Action denied by the user" in output
        mock_approval.register_request.assert_called_once_with("sudo whoami", "ShellTools")
        mock_approval.wait_for_approval.assert_called_once()


@pytest.mark.asyncio
async def test_run_secure_python_code_success() -> None:
    """Test that python code executed via OS sandbox uses bubblewrap."""
    mock_result = SandboxResult(returncode=0, stdout="4\n", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner
        
        output = await run_secure_python_code("print(2+2)", force_os_sandbox=True)
        
        assert "4" in output
        mock_runner.run.assert_called_once()
        # The script is run via python3 from /output/script.py
        cmd_args = mock_runner.run.call_args[0][0]
        assert cmd_args == ["python3", "/output/script.py"]


@pytest.mark.asyncio
async def test_run_secure_python_code_monty_fast_path() -> None:
    """Test that pure python code executes automatically via Tier-1 Monty without HITL."""
    output = await run_secure_python_code("x = 10; y = 20; x + y")
    assert "30" in output
    assert "[Active REPL session state:" in output
    assert "x (int)" in output
    assert "y (int)" in output


@pytest.mark.asyncio
async def test_run_secure_python_code_hitl_approved() -> None:
    """Test that a python script with risky imports runs if approved."""
    mock_result = SandboxResult(returncode=0, stdout="sys version info", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner, \
         patch("rai.tools.python.get_approval_manager") as mock_get_approval:
        
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner
        
        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=True)
        mock_get_approval.return_value = mock_approval
        
        code = "import sys\nprint(sys.version)"
        output = await run_secure_python_code(code)
        
        assert "sys version info" in output
        mock_approval.register_request.assert_called_once_with(code, "PythonTools")
        mock_approval.wait_for_approval.assert_called_once()
        mock_runner.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_secure_python_code_hitl_denied() -> None:
    """Test that a python script with risky imports is blocked if denied."""
    with patch("rai.tools.python.get_approval_manager") as mock_get_approval:
        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=False)
        mock_get_approval.return_value = mock_approval
        
        code = "import os\nos.system('clear')"
        output = await run_secure_python_code(code)
        
        assert "Action denied by the user" in output
        mock_approval.register_request.assert_called_once_with(code, "PythonTools")
        mock_approval.wait_for_approval.assert_called_once()


@pytest.mark.asyncio
async def test_run_secure_python_code_session_persistence_with_store(tmp_path: os.PathLike[str]) -> None:
    """Test that python session state persists into MemoryGraphStore across execution calls."""
    store = SQLiteMemoryGraphStore(Path(tmp_path) / "test_memory.sqlite3")
    init_res = await store.start()
    assert isinstance(init_res, Success)

    session_id = "test-session-persistence"

    # Turn 1: define variables
    out1 = await run_secure_python_code("x = 42\ny = 'hello'", session_id=session_id, store=store)
    assert "[Active REPL session state:" in out1
    assert "x (int)" in out1

    # Verify state was saved to SQLite store
    saved = await store.get_repl_session(session_id)
    assert saved.unwrap() is not None
    blob, vars_dict = saved.unwrap()
    assert len(blob) > 0
    assert vars_dict["x"] == "int"

    # Turn 2: use variables from previous turn
    out2 = await run_secure_python_code("x * 2", session_id=session_id, store=store)
    assert "84" in out2

    # Clear session
    await clear_session_blob(session_id, store=store)
    cleared = await store.get_repl_session(session_id)
    assert cleared.unwrap() is None


@pytest.mark.asyncio
async def test_run_python_code_capability_service_fast_path_and_hitl() -> None:
    """Test that pure python executes seamlessly through CapabilityService without HITL, while risky code demands approval."""
    registry = create_default_capability_registry()
    service = CapabilityService(
        registry=registry,
        policy=PolicyEngine(isolation_available=lambda _iso: True),
        audit=InMemoryAuditLedger(),
        approvals=None,  # No approval broker configured
    )

    # 1. Pure Python code fast path (Tier-1 Monty) through CapabilityService
    descriptor = registry.descriptor("run_python_code")
    assert descriptor is not None
    req_pure = normalize_request(
        descriptor,
        {"code": "x = 10; y = 20; x + y", "session_id": "cap-pure-session"},
    )
    decision_pure, result_pure = await service.invoke(req_pure)
    assert decision_pure is not None
    assert decision_pure.outcome == PolicyOutcome.ALLOW
    assert decision_pure.risk_class == RiskClass.LOW
    assert isinstance(result_pure, Success)
    action_res = result_pure.unwrap()
    assert "30" in action_res.output["text"]
    assert "[Active REPL session state:" in action_res.output["text"]

    # 2. Risky code requiring OS sandbox escalates to HIGH risk and fails with APPROVAL_UNAVAILABLE
    req_risky = normalize_request(
        descriptor,
        {"code": "import os\nos.listdir('.')"},
    )
    decision_risky, result_risky = await service.invoke(req_risky)
    assert decision_risky is not None
    assert decision_risky.outcome == PolicyOutcome.ASK
    assert decision_risky.risk_class == RiskClass.HIGH
    assert isinstance(result_risky, Failure)
    assert result_risky.failure().code == "APPROVAL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_run_secure_python_code_storage_failure_handling() -> None:
    """Test that persistence failures in store_session_blob result in an Execution Failure."""
    mock_store = MagicMock()
    mock_store.get_repl_session = AsyncMock(return_value=Success(None))
    mock_store.save_repl_session = AsyncMock(
        return_value=Failure(
            make_assistant_failure(
                code="SAVE_REPL_SESSION_FAILED",
                message="Disk quota exceeded",
                request_id="test_fail",
            )
        )
    )

    output = await run_secure_python_code(
        "x = 99", session_id="failing-session", store=mock_store
    )
    assert output.startswith("Execution Failure:")
    assert "Failed to persist REPL session state: Disk quota exceeded" in output

