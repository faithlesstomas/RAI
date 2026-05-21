import pytest
import asyncio
from unittest.mock import patch, MagicMock
from rai.tools.security.hitl import get_approval_manager

@pytest.mark.asyncio
async def test_approval_manager_lifecycle():
    manager = get_approval_manager()
    # Reset pending for isolation
    manager._pending.clear()

    # 1. Register a request
    req = manager.register_request("rm -rf /workspace/output/test.txt", "ShellTools")
    assert req.id.startswith("appr-")
    assert req.command == "rm -rf /workspace/output/test.txt"
    assert req.tool_name == "ShellTools"
    assert req.status == "pending"

    # 2. Get the request
    retrieved = manager.get_request(req.id)
    assert retrieved is req

    # 3. List pending requests
    pending = manager.list_pending()
    assert req.id in pending
    assert pending[req.id]["command"] == req.command

    # 4. Resolve the request as approved
    resolved = manager.resolve_request(req.id, approved=True)
    assert resolved is True
    assert req.status == "approved"
    assert req.event.is_set()

    # 5. List pending should no longer contain it
    pending_after = manager.list_pending()
    assert req.id not in pending_after

@pytest.mark.asyncio
async def test_wait_for_approval_flow():
    manager = get_approval_manager()
    manager._pending.clear()

    req = manager.register_request("cat /etc/passwd", "ShellTools")

    # Mock the desktop prompt to do nothing so it doesn't try to spawn Zenity
    with patch.object(manager, "prompt_desktop_async", return_value=None) as mock_prompt:
        # Create execution task that calls wait_for_approval
        task = asyncio.create_task(manager.wait_for_approval(req))

        # Yield execution to allow task to run and block on event.wait()
        await asyncio.sleep(0.05)
        
        # Verify prompt_desktop_async was scheduled
        mock_prompt.assert_called_once_with(req)
        assert not req.event.is_set()

        # Resolve request as approved
        manager.resolve_request(req.id, approved=True)

        # Wait for the task to complete
        result = await task
        assert result is True
        assert req.status == "approved"

@pytest.mark.asyncio
async def test_wait_for_approval_denied_flow():
    manager = get_approval_manager()
    manager._pending.clear()

    req = manager.register_request("sudo chmod 777 /", "ShellTools")

    with patch.object(manager, "prompt_desktop_async", return_value=None):
        task = asyncio.create_task(manager.wait_for_approval(req))
        await asyncio.sleep(0.05)

        # Resolve request as rejected/denied
        manager.resolve_request(req.id, approved=False)

        result = await task
        assert result is False
        assert req.status == "rejected"
