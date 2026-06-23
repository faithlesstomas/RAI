"""
Unit and integration tests for secure shell and python tools.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rai.tools.shell import run_secure_shell_command
from rai.tools.python import run_secure_python_code
from rai.tools.security.sandbox import SandboxResult


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
    """Test that simple python code executes successfully."""
    mock_result = SandboxResult(returncode=0, stdout="4\n", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner
        
        output = await run_secure_python_code("print(2+2)")
        
        assert "4" in output
        mock_runner.run.assert_called_once()
        # The script is run via python3 from /output/script.py
        cmd_args = mock_runner.run.call_args[0][0]
        assert cmd_args == ["python3", "/output/script.py"]


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
