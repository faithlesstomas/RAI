"""Tests for Jupyter and IPython integrations with Monty REPL."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from returns.result import Success

from rai.integrations.kernel import MontyJupyterKernel
from rai.integrations.ipython import RAIMagics, load_ipython_extension


def test_monty_jupyter_kernel_execution_and_persistence() -> None:
    kernel = MontyJupyterKernel()
    mock_send = MagicMock()
    kernel.send_response = mock_send

    # Cell 1: Define variables
    res1 = kernel.do_execute("x = 10\ny = 20\nx + y", silent=False)
    assert res1["status"] == "ok"
    assert kernel.session_blob is not None
    assert kernel.variables.get("x") == "int"
    assert kernel.variables.get("y") == "int"

    # Cell 2: Access variable defined in Cell 1
    res2 = kernel.do_execute("z = x * 3\nz", silent=False)
    assert res2["status"] == "ok"
    assert kernel.variables.get("z") == "int"

    # Verify stdout was emitted
    assert mock_send.called
    stream_calls = [call for call in mock_send.call_args_list if call.args[1] == "stream"]
    min_stream_calls = 2
    assert len(stream_calls) >= min_stream_calls


def test_monty_jupyter_kernel_empty_and_error() -> None:
    kernel = MontyJupyterKernel()
    mock_send = MagicMock()
    kernel.send_response = mock_send

    # Empty cell
    res_empty = kernel.do_execute("   \n", silent=True)
    assert res_empty["status"] == "ok"

    # Blocked execution
    res_err = kernel.do_execute("import os\nos.system('id')", silent=False)
    assert res_err["status"] == "error"
    assert res_err["ename"] == "MontyExecutionError"


def test_rai_magics_session_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    magics = RAIMagics(session_id="initial-session")
    magics.rai("help")
    captured = capsys.readouterr()
    assert "RAI IPython Magic Usage" in captured.out

    magics.rai("session custom-session")
    assert magics.session_id == "custom-session"


def test_rai_magics_ask_and_cell() -> None:
    magics = RAIMagics(session_id="magic-test-session")
    mock_service = MagicMock()
    mock_service.start = AsyncMock(return_value=Success(None))
    mock_candidate = MagicMock(text="AI response text")
    mock_service.accept_turn = AsyncMock(return_value=Success(mock_candidate))

    with patch.object(magics, "_get_service", return_value=mock_service):
        magics.rai("ask How do I sort a list?")
        assert mock_service.accept_turn.called
        turn = mock_service.accept_turn.call_args[0][0]
        assert turn.text == "How do I sort a list?"
        assert turn.session_id == "magic-test-session"

        # Cell magic
        mock_service.accept_turn.reset_mock()
        magics.rai_cell("", "Write a quicksort implementation")
        assert mock_service.accept_turn.called
        turn_cell = mock_service.accept_turn.call_args[0][0]
        assert "quicksort" in turn_cell.text


def test_load_ipython_extension() -> None:
    mock_ipython = MagicMock()
    load_ipython_extension(mock_ipython)
    assert mock_ipython.register_magics.called


def test_rai_magics_unique_session_ids() -> None:
    """Test that distinct instances of RAIMagics get unique session IDs by default, preventing notebook variable leaks."""
    magic1 = RAIMagics()
    magic2 = RAIMagics()
    assert magic1.session_id != magic2.session_id
    assert magic1.session_id.startswith("ipython-")
    assert magic2.session_id.startswith("ipython-")

    magic_explicit = RAIMagics(session_id="explicit-id")
    assert magic_explicit.session_id == "explicit-id"


@pytest.mark.asyncio
async def test_rai_magics_inside_active_asyncio_loop(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that RAIMagics executes in an active asyncio event loop (e.g. Jupyter kernel) without RuntimeError."""
    magics = RAIMagics(session_id="jupyter-loop-session")
    mock_service = MagicMock()
    mock_service.start = AsyncMock(return_value=Success(None))
    mock_candidate = MagicMock(text="Response generated inside loop")
    mock_service.accept_turn = AsyncMock(return_value=Success(mock_candidate))
    mock_service.store = MagicMock()
    mock_service.store.get_repl_session = AsyncMock(
        return_value=Success((b"mock_blob", {"ans": "int", "msg": "str"}))
    )

    with patch.object(magics, "_get_service", return_value=mock_service):
        # 1. Line magic in active loop
        magics.rai("ask Test prompt in active loop")
        captured = capsys.readouterr()
        assert "Response generated inside loop" in captured.out

        # 2. Cell magic in active loop
        magics.rai_cell("", "x = 42\nx")
        captured = capsys.readouterr()
        assert "Response generated inside loop" in captured.out

        # 3. REPL state display in active loop
        magics.rai("repl")
        captured = capsys.readouterr()
        assert "Active REPL variables (jupyter-loop-session):" in captured.out
        assert "ans: int" in captured.out
        assert "msg: str" in captured.out
