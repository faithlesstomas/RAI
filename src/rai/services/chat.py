"""
Chat Service for handling agent interactions using the Google Antigravity SDK.
"""
import os
import uuid
import logging
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from returns.result import Failure, Result, Success

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import CapabilitiesConfig, BuiltinTools
from rai.core import setup_tools
from rai.config_manager import (
    load_config,
    load_agents,
    TRAJECTORY_DIR,
    get_conversation_id_for_session,
    set_conversation_id_for_session,
    clear_conversation_id_for_session,
)
from rai.exceptions import ChainExecutionError
from rai.services.history import HistoryService

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for managing chat interactions and agent lifecycles via google-antigravity.
    """

    def __init__(self) -> None:
        self._history_service = HistoryService()

    def _resolve_agent_config(
        self,
        chain_configs: Optional[List[Dict[str, Any]]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolves agent configuration from either explicit agent_id, overrides, or system config."""
        config = {}
        if agent_id:
            agents = load_agents()
            if agent_id in agents:
                config = agents[agent_id].copy()
        else:
            app_config = load_config()
            active_agent = app_config.get("active_agent") or app_config.get("active_session", "default")
            config = (app_config.get("agents") or app_config.get("sessions", {})).get(active_agent, {}).copy()

        if chain_configs and len(chain_configs) > 0:
            overrides = chain_configs[0]
            for key, val in overrides.items():
                if val is not None:
                    config[key] = val

        return config

    async def run_chain(
        self,
        chain_input: str,
        chain_configs: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        """
        Runs a stateful agent execution turn with the Antigravity SDK.
        """
        if not chain_input:
            return Failure(ValueError("Missing input."))

        final_session_id = session_id or str(uuid.uuid4())
        agent_config = self._resolve_agent_config(chain_configs, agent_id)

        # Record user query in history
        await self._history_service.add_message(final_session_id, "user", chain_input)

        # Set up active agent tools
        enabled_tool_names = agent_config.get("tools")
        agent_tools, _ = setup_tools(
            enable_tools=True,
            quiet=True,
            enabled_tool_names=enabled_tool_names,
        )

        # Resolve persistent conversation ID
        conv_id = get_conversation_id_for_session(final_session_id)
        traj_file = os.path.join(TRAJECTORY_DIR, f"traj-{conv_id}") if conv_id else ""

        if conv_id and os.path.exists(traj_file):
            logger.info("Resuming conversation %s from %s", conv_id, traj_file)
            actual_conv_id = conv_id
        else:
            logger.info("Starting a new conversation for session %s", final_session_id)
            actual_conv_id = None

        try:
            # Construct LocalAgentConfig
            config = LocalAgentConfig(
                system_instructions=agent_config.get("system") or agent_config.get("system_instructions"),
                model=agent_config.get("model", "gemini-2.5-flash"),
                tools=agent_tools,
                conversation_id=actual_conv_id,
                save_dir=TRAJECTORY_DIR,
                capabilities=CapabilitiesConfig(
                    disabled_tools=[BuiltinTools.RUN_COMMAND]
                )
            )

            # Start agent session
            async with Agent(config) as ag:
                # If resuming, wait a fraction of a second and drain history steps from connection queue
                if actual_conv_id and hasattr(ag, "conversation"):
                    await asyncio.sleep(0.15)
                    queue = ag.conversation.connection._step_queue
                    while not queue.empty():
                        try:
                            step = queue.get_nowait()
                            ag.conversation._steps.append(step)
                        except asyncio.QueueEmpty:
                            break

                response = await ag.chat(prompt=chain_input)
                content = await response.text()

                # Capture the conversation ID AFTER ag.chat() — the cascade_id
                # is populated by the WebSocket reader loop during chat().
                new_conv_id = ag.conversation_id
                if new_conv_id:
                    set_conversation_id_for_session(final_session_id, new_conv_id)

                # Extract tool calls safely
                tool_calls = []
                async for tc in response.tool_calls:
                    tool_calls.append({
                        "name": tc.name,
                        "arguments": tc.args,
                    })

                # Save turn response to history DB
                await self._history_service.add_message(
                    final_session_id,
                    "assistant",
                    content,
                    tool_calls=tool_calls if tool_calls else None
                )

                payload = {
                    "content": content,
                    "tool_calls": tool_calls if tool_calls else None,
                    "session_id": final_session_id,
                }
                return Success(payload)

        except Exception as e:
            logger.error("Error during agent execution: %s", e, exc_info=True)
            return Failure(ChainExecutionError(f"Agent execution failed: {e}"))

    async def stream_chain(
        self,
        chain_input: str,
        chain_configs: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """
        Streams agent execution token deltas.
        """
        if not chain_input:
            yield Failure(ValueError("Missing input."))
            return

        final_session_id = session_id or str(uuid.uuid4())
        agent_config = self._resolve_agent_config(chain_configs, agent_id)

        # Record user query in history
        await self._history_service.add_message(final_session_id, "user", chain_input)

        enabled_tool_names = agent_config.get("tools")
        agent_tools, _ = setup_tools(
            enable_tools=True,
            quiet=True,
            enabled_tool_names=enabled_tool_names,
        )

        # Resolve persistent conversation ID
        conv_id = get_conversation_id_for_session(final_session_id)
        traj_file = os.path.join(TRAJECTORY_DIR, f"traj-{conv_id}") if conv_id else ""

        if conv_id and os.path.exists(traj_file):
            logger.info("Resuming conversation %s from %s", conv_id, traj_file)
            actual_conv_id = conv_id
        else:
            logger.info("Starting a new conversation for session %s", final_session_id)
            actual_conv_id = None

        try:
            config = LocalAgentConfig(
                system_instructions=agent_config.get("system") or agent_config.get("system_instructions"),
                model=agent_config.get("model", "gemini-2.5-flash"),
                tools=agent_tools,
                conversation_id=actual_conv_id,
                save_dir=TRAJECTORY_DIR,
            )

            accumulated_response = ""
            async with Agent(config) as ag:
                # If resuming, wait a fraction of a second and drain history steps from connection queue
                if actual_conv_id and hasattr(ag, "conversation"):
                    await asyncio.sleep(0.15)
                    queue = ag.conversation.connection._step_queue
                    while not queue.empty():
                        try:
                            step = queue.get_nowait()
                            ag.conversation._steps.append(step)
                        except asyncio.QueueEmpty:
                            break

                response = await ag.chat(prompt=chain_input)
                async for chunk in response:
                    accumulated_response += chunk
                    yield chunk

                # Capture the conversation ID AFTER ag.chat() — the cascade_id
                # is populated by the WebSocket reader loop during chat().
                new_conv_id = ag.conversation_id
                if new_conv_id:
                    set_conversation_id_for_session(final_session_id, new_conv_id)

                # Collect tool calls at the end of the turn
                tool_calls = []
                async for tc in response.tool_calls:
                    tool_calls.append({
                        "name": tc.name,
                        "arguments": tc.args,
                    })

                # Save streamed assistant response to history
                await self._history_service.add_message(
                    final_session_id,
                    "assistant",
                    accumulated_response,
                    tool_calls=tool_calls if tool_calls else None
                )

        except Exception as e:
            logger.error("Error during streaming execution: %s", e, exc_info=True)
            yield Failure(ChainExecutionError(f"Error during streaming: {e}"))

    async def get_session_history(self, session_id: str) -> Result[List[Dict[str, Any]], Exception]:
        """Retrieves history for a session."""
        return await self._history_service.get_session_history(session_id)

    async def clear_session_history(self, session_id: str) -> Result[None, Exception]:
        """Clears history for a specific session and resets the conversation trajectory."""
        db_res = await self._history_service.clear_history(session_id)
        if isinstance(db_res, Failure):
            return db_res
        try:
            clear_conversation_id_for_session(session_id)
            return Success(None)
        except Exception as e:
            logger.error(f"Failed to clear conversation id mapping: {e}")
            return Failure(e)

    async def add_message_to_history(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Result[None, Exception]:
        """Adds a message to the session history."""
        return await self._history_service.add_message(session_id, role, content, tool_calls)
