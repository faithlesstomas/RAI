"""
Human-in-the-Loop (HITL) consent mechanism and transaction authorization layer.
"""
import uuid
import shutil
import logging
import asyncio
import sys
import os
from asyncio.subprocess import Process
from typing import Dict, Any, Optional
from rich.panel import Panel

logger = logging.getLogger(__name__)

class ApprovalRequest:
    """Represents a pending or resolved system tool execution authorization request."""

    def __init__(self, request_id: str, command: str, tool_name: str) -> None:
        self.id = request_id
        self.command = command
        self.tool_name = tool_name
        self.status = "pending"  # 'pending', 'approved', 'rejected'
        self.event = asyncio.Event()
        self.zenity_process: Optional[Process] = None


class ApprovalManager:
    """
    Centralized singleton to coordinate human authorization requests.
    Supports asynchronous suspension and desktop dialog fallbacks.
    """
    _instance: Optional["ApprovalManager"] = None
    _pending: Dict[str, ApprovalRequest] = {}

    def __new__(cls) -> "ApprovalManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pending = {}
        return cls._instance

    def register_request(self, command: str, tool_name: str) -> ApprovalRequest:
        """Registers a new approval request."""
        request_id = f"appr-{uuid.uuid4().hex[:8]}"
        req = ApprovalRequest(request_id, command, tool_name)
        self._pending[request_id] = req
        logger.info("Registered approval request %s for command: %s", request_id, command)
        return req

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Retrieves a registered approval request."""
        return self._pending.get(request_id)

    def list_pending(self) -> Dict[str, Dict[str, Any]]:
        """Lists all pending approval requests."""
        return {
            req_id: {
                "id": req.id,
                "command": req.command,
                "tool_name": req.tool_name,
                "status": req.status,
            }
            for req_id, req in self._pending.items()
            if req.status == "pending"
        }

    def resolve_request(self, request_id: str, approved: bool) -> bool:
        """Resolves a pending request, setting its event and releasing suspended coroutines."""
        req = self._pending.get(request_id)
        if not req or req.status != "pending":
            return False

        req.status = "approved" if approved else "rejected"
        logger.info("Resolved approval request %s: %s", request_id, req.status)

        # Terminate any active desktop dialog if resolved via API
        if req.zenity_process:
            try:
                req.zenity_process.terminate()
            except ProcessLookupError:
                pass  # already closed

        # Set the event to resume execution
        req.event.set()
        return True

    async def prompt_desktop_async(self, req: ApprovalRequest) -> None:
        """
        Attempts to display a system GUI prompt to the user using Zenity on a desktop environment.
        Runs asynchronously without blocking the event loop.
        """
        zenity_path = shutil.which("zenity")
        if not zenity_path:
            logger.debug("Zenity not available. Skipping GUI popup.")
            return

        # Prepare a highly visible security warning
        dialog_text = (
            "🚨 RAI SECURITY GATEWAY DIALOG 🚨\n\n"
            f"An AI agent wants to execute a command outside the sandbox.\n"
            f"Tool: [bold]{req.tool_name}[/bold]\n\n"
            "Command:\n"
            f"👉 {req.command}\n\n"
            "Do you authorize this system execution?"
        )

        try:
            # Spawn zenity subprocess asynchronously
            proc = await asyncio.create_subprocess_exec(
                zenity_path,
                "--question",
                "--text",
                dialog_text,
                "--title",
                "RAI Security Consent Required",
                "--ok-label",
                "Authorize",
                "--cancel-label",
                "Deny",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            req.zenity_process = proc

            # Wait for user input or process termination
            returncode = await proc.wait()
            
            # Zenity returns 0 for OK, and 1 for Cancel/Close
            if req.status == "pending":
                is_approved = (returncode == 0)
                self.resolve_request(req.id, is_approved)

        except Exception as e:
            logger.error("Failed to spawn or handle Zenity desktop dialog: %s", e)

    async def wait_for_approval(self, req: ApprovalRequest) -> bool:
        """
        Suspends the executing thread, spawns standard UI prompts,
        and waits until approval is resolved.
        """
        # 1. Spawn Zenity desktop prompt in the background (concurrently)
        asyncio.create_task(self.prompt_desktop_async(req))

        # 2. If running standalone and stdin is a TTY, also run a console approval prompt
        async def console_prompt() -> None:
            if os.environ.get("RAI_SERVE") != "1" and sys.stdin.isatty():
                try:
                    from ...core import console
                    print("\n")
                    console.print(Panel(
                        f"[bold yellow]Command:[/bold yellow] {req.command}\n"
                        f"[bold yellow]Tool:[/bold yellow] {req.tool_name}",
                        title="🚨 RAI STANDALONE GATEWAY - AUTHORIZATION REQUIRED 🚨",
                        border_style="red"
                    ))
                    loop = asyncio.get_running_loop()
                    user_val = await loop.run_in_executor(None, input, "Authorize this execution? (y/n): ")
                    approved = user_val.strip().lower() in ("y", "yes")
                    if req.status == "pending":
                        self.resolve_request(req.id, approved)
                except Exception as e:
                    logger.error("Error in console approval prompt: %s", e)

        asyncio.create_task(console_prompt())

        # 3. Block until the request event is set (either by Zenity or by REST/WebSocket APIs)
        await req.event.wait()

        # 4. Clean up registry
        if req.id in self._pending:
            # We keep resolved requests in case clients query them later, but can pop them or expire them
            pass

        return req.status == "approved"


def get_approval_manager() -> ApprovalManager:
    """Returns the global singleton approval manager."""
    return ApprovalManager()
