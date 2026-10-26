"""
Unit and integration tests for secure shell and python tools.
"""
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Failure, Result, Success

from rai.assistant.records import make_assistant_failure
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.commands.approval import TerminalApprovalBroker
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.defaults import HitlApprovalBroker, create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.tools.python import clear_session_blob, get_session_repl_lock, run_secure_python_code
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
    """Test that python code executed via OS sandbox uses bubblewrap and requires approval."""
    mock_result = SandboxResult(returncode=0, stdout="4\n", stderr="", sandbox_type="bubblewrap")
    
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner, \
         patch("rai.tools.python.get_approval_manager") as mock_get_approval:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner

        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=True)
        mock_get_approval.return_value = mock_approval

        output = await run_secure_python_code("print(2+2)", force_os_sandbox=True)
        
        assert "4" in output
        mock_approval.register_request.assert_called_once_with("print(2+2)", "PythonTools")
        mock_approval.wait_for_approval.assert_called_once()
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
async def test_run_secure_python_code_monty_unavailable_fallback_requires_hitl() -> None:
    """When Monty runner is unavailable, execution must fallback to Tier-2 and require HITL approval."""
    mock_result = SandboxResult(returncode=0, stdout="2\n", stderr="", sandbox_type="bubblewrap")
    mock_monty = MagicMock()
    mock_monty.is_available.return_value = False

    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner, \
         patch("rai.tools.python.get_approval_manager") as mock_get_approval, \
         patch("rai.tools.python.get_monty_runner", return_value=mock_monty):

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner

        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=True)
        mock_get_approval.return_value = mock_approval

        code = "x = 1 + 1; print(x)"
        output = await run_secure_python_code(code, monty_runner=mock_monty)

        assert "2" in output
        mock_approval.register_request.assert_called_once_with(code, "PythonTools")
        mock_approval.wait_for_approval.assert_called_once()
        mock_runner.run.assert_called_once()


def test_can_run_in_monty_rejections() -> None:
    from rai.tools.python import can_run_in_monty  # noqa: PLC0415

    ok, reason = can_run_in_monty("__import__('os').system('id')")
    assert not ok
    assert "sensitive builtins" in reason or "Builtin" in reason

    ok, reason = can_run_in_monty("eval('2 + 2')")
    assert not ok

    ok, reason = can_run_in_monty("exec('x = 1')")
    assert not ok

    ok, reason = can_run_in_monty("open('/etc/passwd')")
    assert not ok

    ok, reason = can_run_in_monty("().__class__.__base__.__subclasses__()")
    assert not ok
    assert "Reflection attribute" in reason

    ok, _ = can_run_in_monty("x = 42\ny = [i * 2 for i in range(10)]")
    assert ok


def test_python_risk_evaluator_policy() -> None:
    from rai.kernel.defaults import _python_risk_evaluator  # noqa: PLC0415

    # Safe pure python with Monty available -> LOW
    assert _python_risk_evaluator({"code": "x = 1 + 1"}) == RiskClass.LOW

    # Dynamic imports or OS operations -> HIGH
    assert _python_risk_evaluator({"code": "__import__('os').system('id')"}) == RiskClass.HIGH
    assert _python_risk_evaluator({"code": "import os"}) == RiskClass.HIGH
    assert _python_risk_evaluator({"code": "open('test.txt')"}) == RiskClass.HIGH

    # force_os_sandbox or allow_network -> HIGH
    assert _python_risk_evaluator({"code": "x = 1", "force_os_sandbox": True}) == RiskClass.HIGH
    assert _python_risk_evaluator({"code": "x = 1", "allow_network": True}) == RiskClass.HIGH

    # When Monty runner is unavailable -> unconditionally HIGH
    mock_runner = MagicMock()
    mock_runner.is_available.return_value = False
    with patch("rai.tools.security.monty.get_monty_runner", return_value=mock_runner):
        assert _python_risk_evaluator({"code": "x = 1 + 1"}) == RiskClass.HIGH


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


@pytest.mark.asyncio
async def test_run_secure_python_code_load_failure_handling() -> None:
    """Test that failure to retrieve session blob results in an Execution Failure and aborts execution."""
    mock_store = MagicMock()
    mock_store.get_repl_session = AsyncMock(
        return_value=Failure(
            make_assistant_failure(
                code="GET_REPL_SESSION_FAILED",
                message="Corrupted session database",
                request_id="test_fail_load",
            )
        )
    )
    mock_store.save_repl_session = AsyncMock()

    output = await run_secure_python_code(
        "x = 123", session_id="corrupt-session", store=mock_store
    )
    assert output.startswith("Execution Failure:")
    assert "Failed to load REPL session state: Corrupted session database" in output
    mock_store.save_repl_session.assert_not_called()


@pytest.mark.asyncio
async def test_container_python_propagates_profile_scope(tmp_path: Path) -> None:
    """Test that ApplicationContainer's run_python_code wrapper propagates active profile_scope."""
    container = ApplicationContainer(
        config={"active_agent": "researcher"},
        testing=True,
    )
    assert container.active_profile == "researcher"

    with patch("rai.tools.python.run_secure_python_code", new=AsyncMock(return_value="[Success]")) as mock_run:
        handler = container.capability_registry._capabilities["run_python_code"]._handler
        assert handler is not None
        res = await handler({"code": "x = 42", "session_id": "test-session"})
        assert res == {"text": "[Success]"}
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["profile_scope"] == "researcher"
        assert mock_run.call_args.kwargs["require_hitl"] is False


@pytest.mark.asyncio
async def test_session_repl_lock_shared_serialization() -> None:
    """Test that get_session_repl_lock returns the same lock for identical (profile_scope, session_id)."""
    lock1 = await get_session_repl_lock("profile_a", "session_1")
    lock2 = await get_session_repl_lock("profile_a", "session_1")
    lock3 = await get_session_repl_lock("profile_b", "session_1")
    lock4 = await get_session_repl_lock("profile_a", "session_2")

    assert lock1 is lock2
    assert lock1 is not lock3
    assert lock1 is not lock4


@pytest.mark.asyncio
async def test_run_python_code_capability_hitl_single_prompt() -> None:
    """Test that CapabilityService asks for HITL approval once and does not double-prompt in handler."""
    registry = create_default_capability_registry()
    mock_broker = MagicMock()
    mock_broker.request = AsyncMock(return_value=Success("approval-123"))

    service = CapabilityService(
        registry=registry,
        policy=PolicyEngine(isolation_available=lambda _iso: True),
        audit=InMemoryAuditLedger(),
        approvals=mock_broker,
    )

    mock_result = SandboxResult(returncode=0, stdout="hello from os\n", stderr="", sandbox_type="bubblewrap")
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner, \
         patch("rai.tools.python.get_approval_manager") as mock_get_approval:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner

        mock_approval = MagicMock()
        mock_approval.wait_for_approval = AsyncMock(return_value=True)
        mock_get_approval.return_value = mock_approval

        descriptor = registry.descriptor("run_python_code")
        assert descriptor is not None
        req = normalize_request(
            descriptor,
            {"code": "import os\nos.listdir('.')"},
        )
        decision, result = await service.invoke(req)

        assert decision is not None
        assert decision.outcome == PolicyOutcome.ASK
        mock_broker.request.assert_called_once()
        mock_approval.register_request.assert_not_called()
        assert isinstance(result, Success)
        assert "hello from os" in result.unwrap().output["text"]


@pytest.mark.asyncio
async def test_hitl_approval_broker_registers_exact_python_script() -> None:
    """Test that HitlApprovalBroker passes the exact Python script and details for Tier-2 approval."""
    broker = HitlApprovalBroker()
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    decision = PolicyDecision(
        producer=producer,
        request_id="req-py-1",
        outcome=PolicyOutcome.ASK,
        risk_class=RiskClass.HIGH,
        policy_version="1.0.0",
        reason_codes=("PYTHON_HOST_EXECUTION",),
        actor=producer,
        data_class=DataClass.PUBLIC,
        target_resource="python:workspace",
        requested_side_effects=("process",),
        isolation="host-process",
        verification_plan=("observed",),
    )
    registry = create_default_capability_registry()
    descriptor = registry.descriptor("run_python_code")
    assert descriptor is not None
    test_script = "import os\nprint(os.getpid())"
    req = normalize_request(descriptor, {"code": test_script})

    with patch("rai.kernel.defaults.get_approval_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.register_request.return_value = MagicMock()
        mock_manager.wait_for_approval = AsyncMock(return_value=True)
        mock_get_manager.return_value = mock_manager

        res = await broker.request_action(decision, req, CancellationToken())
        assert isinstance(res, Success)
        mock_manager.register_request.assert_called_once()
        call_args, call_kwargs = mock_manager.register_request.call_args
        assert call_args[0] == test_script
        assert call_args[1] == "PythonTools"
        assert call_kwargs["details"]["code"] == test_script
        assert call_kwargs["details"]["target_resource"] == "capability://run_python_code"


@pytest.mark.asyncio
async def test_terminal_approval_broker_shows_exact_python_script(tmp_path: Path) -> None:
    """Test that TerminalApprovalBroker prints the exact Python script for Tier-2 approval."""
    config_file = tmp_path / "approvals.json"
    config_file.write_text("{}", encoding="utf-8")
    broker = TerminalApprovalBroker(profile="default", path=config_file)
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    decision = PolicyDecision(
        producer=producer,
        request_id="req-py-2",
        outcome=PolicyOutcome.ASK,
        risk_class=RiskClass.HIGH,
        policy_version="1.0.0",
        reason_codes=("PYTHON_HOST_EXECUTION",),
        actor=producer,
        data_class=DataClass.PUBLIC,
        target_resource="python:workspace",
        requested_side_effects=("process",),
        isolation="host-process",
        verification_plan=("observed",),
    )
    registry = create_default_capability_registry()
    descriptor = registry.descriptor("run_python_code")
    assert descriptor is not None
    test_script = "import sys\nprint('running tier-2')"
    req = normalize_request(descriptor, {"code": test_script})

    with patch("rai.commands.approval.typer.echo") as mock_echo, \
         patch("rai.commands.approval.choose", new=AsyncMock(return_value="y")):
        res = await broker.request_action(decision, req, CancellationToken())
        assert isinstance(res, Success)
        mock_echo.assert_called_once()
        echo_output = mock_echo.call_args[0][0]
        assert "Script:" in echo_output
        assert test_script in echo_output
        assert "Tier-2 Bubblewrap OS execution" in echo_output


@pytest.mark.asyncio
async def test_capability_service_delegates_to_request_action() -> None:
    """Test that CapabilityService invokes broker.request_action with request arguments."""
    registry = create_default_capability_registry()
    mock_broker = MagicMock()
    mock_broker.request_action = AsyncMock(return_value=Success("approval-py-123"))

    service = CapabilityService(
        registry=registry,
        policy=PolicyEngine(isolation_available=lambda _iso: True),
        audit=InMemoryAuditLedger(),
        approvals=mock_broker,
    )

    mock_result = SandboxResult(returncode=0, stdout="done\n", stderr="", sandbox_type="bubblewrap")
    with patch("rai.tools.python.get_sandbox_runner") as mock_get_runner:
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_get_runner.return_value = mock_runner

        descriptor = registry.descriptor("run_python_code")
        assert descriptor is not None
        test_code = "import os\nos.listdir('.')"
        req = normalize_request(descriptor, {"code": test_code})
        decision, result = await service.invoke(req)

        assert decision is not None
        assert decision.outcome == PolicyOutcome.ASK
        mock_broker.request_action.assert_called_once()
        call_decision, call_req, _ = mock_broker.request_action.call_args[0]
        assert call_req.arguments["code"] == test_code
        assert isinstance(result, Success)


@pytest.mark.asyncio
async def test_capability_service_rejects_missing_code_before_hitl_prompt() -> None:
    """Malformed run_python_code request missing 'code' must fail with INVALID_ARGUMENT without prompting HITL."""
    registry = create_default_capability_registry()
    mock_broker = MagicMock()
    mock_broker.request_action = AsyncMock(return_value=Success("approval-py-123"))

    service = CapabilityService(
        registry=registry,
        policy=PolicyEngine(isolation_available=lambda _iso: True),
        audit=InMemoryAuditLedger(),
        approvals=mock_broker,
    )

    descriptor = registry.descriptor("run_python_code")
    assert descriptor is not None
    req = normalize_request(descriptor, {"session_id": "sess-1"})
    decision, result = await service.invoke(req)

    assert decision is not None
    assert decision.outcome == PolicyOutcome.ASK
    mock_broker.request_action.assert_not_called()
    assert isinstance(result, Failure)
    assert result.failure().code == "INVALID_ARGUMENT"


def test_extract_model_tool_call_gemma_multiline_and_quotes() -> None:
    """Gemma tool call syntax with <|"|> delimiters, newlines, quotes, and dicts must preserve code."""
    from rai.assistant.tool_calling import extract_model_tool_call  # noqa: PLC0415

    code_body = 'from scipy.special import zeta\nd = {"s": 2}\nprint(f"zeta(2) = {zeta(d[\'s\'])}")'
    raw = f'<|tool_call>call:run_python_code{{code:<|"|>{code_body}<|"|>}}<tool_call|>'
    extracted = extract_model_tool_call(raw)
    assert extracted is not None
    _, cap, args = extracted
    assert cap == "run_python_code"
    assert args.get("code") == code_body


def test_format_action_summary_sanitizes_escape_sequences() -> None:
    """Terminal approval summary must strip ANSI escape sequences and non-printable control chars."""
    from rai.commands.approval import _format_action_summary  # noqa: PLC0415

    registry = create_default_capability_registry()
    descriptor = registry.descriptor("run_python_code")
    assert descriptor is not None

    malicious_code = "import os\n# \x1b[2K\rPrompt hijacked!\nprint('real code')"
    req = normalize_request(descriptor, {"code": malicious_code})
    policy = PolicyEngine(isolation_available=lambda _: True)
    decision = policy.evaluate(req, descriptor)
    summary = _format_action_summary(req, decision)
    assert "\x1b" not in summary
    assert "Prompt hijacked!" in summary
    assert "print('real code')" in summary


def test_sanitize_terminal_text_neutralizes_c1_controls() -> None:
    """Both 7-bit ANSI and 8-bit C1 controls (e.g. U+009B) must be stripped/neutralized."""
    from rai.tools.security.hitl import sanitize_terminal_text  # noqa: PLC0415

    # \x9b is 8-bit CSI (U+009B)
    malicious = "echo 'test'\x9b2K\r\x80\x9fmalicious\u202esverted"
    sanitized = sanitize_terminal_text(malicious)
    assert "\x9b" not in sanitized
    assert "\x80" not in sanitized
    assert "\x9f" not in sanitized
    assert "\u202e" not in sanitized
    assert "\\r" in sanitized or "\r" not in sanitized


@pytest.mark.asyncio
async def test_workflow_binds_trusted_session_id_overriding_model_input() -> None:
    """Workflow controller must always bind trusted session_id even if model passed a different one."""
    from rai.actions.workflow import BoundedStepController  # noqa: PLC0415
    from rai.kernel.records import _utc_now  # noqa: PLC0415

    registry = create_default_capability_registry()
    mock_service = MagicMock()
    mock_service.registry = registry

    captured_requests: list[CapabilityRequest] = []

    async def mock_invoke(
        req: CapabilityRequest, cancellation: CancellationToken
    ) -> tuple[object, Result[ActionResult, ActionFailure]]:
        captured_requests.append(req)
        return None, Success(
            ActionResult(
                record_id="res-1",
                timestamp=_utc_now(),
                producer=ProducerIdentity(producer_id="p-1", kind="test", version="1.0"),
                request_id=req.record_id,
                capability=req.capability,
                output={"result": "ok"},
                verification={"status": "verified"},
            )
        )

    mock_service.invoke = mock_invoke

    controller = BoundedStepController(capabilities=mock_service)
    wf = controller.create_generic_workflow(
        task_id="task-step-1", session_id="trusted-session-uuid", goal="Run code"
    )

    # Model attempts to target a different session
    malicious_args = {"code": "x = 1", "session_id": "malicious-victim-session"}

    res = await controller.execute_generic_step(
        wf, mock_service, "run_python_code", malicious_args, CancellationToken()
    )
    assert isinstance(res, Success)
    assert len(captured_requests) == 1
    assert captured_requests[0].arguments["session_id"] == "trusted-session-uuid"
