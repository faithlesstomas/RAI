"""Quarantined Google Antigravity compatibility backend."""
import os
import uuid
import logging
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from returns.result import Failure, Result, Success

from google.antigravity import Agent, LocalAgentConfig, LocalOpenAIAgentConfig
from google.antigravity.types import (
    BuiltinTools,
    CapabilitiesConfig,
    CustomSystemInstructions,
    SubagentConfig,
)
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
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.audit import JsonlAuditLedger
from rai.kernel.compatibility import UNTRUSTED_CONTENT_INSTRUCTION
from rai.kernel.defaults import HitlApprovalBroker, create_default_capability_registry, isolation_available
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    ContextPackage,
    InferenceBudget,
    ProducerIdentity,
    Task,
)
from rai.services.history import HistoryService

logger = logging.getLogger(__name__)
BACKEND_PRODUCER = ProducerIdentity(
    producer_id="rai.backend.antigravity",
    kind="agent-backend",
    version="1.0.0",
)

DEFAULT_ANTIGRAVITY_MODEL = "gemini-2.5-flash"
DEFAULT_LEMONADE_URL = "http://127.0.0.1:13305/api/v1"


async def resolve_lemonade_model(
    base_url: str = DEFAULT_LEMONADE_URL,
    explicit_model: Optional[str] = None,
) -> str:
    """Dynamically resolve the active model on Lemonade daemon, never hardcoding."""
    if explicit_model:
        return explicit_model
    env_model = os.environ.get("RAI_LEMONADE_WORKER_MODEL") or os.environ.get("LEMONADE_MODEL")
    if env_model:
        return env_model
    try:
        import httpx

        host = base_url.replace("/api/v1", "").rstrip("/")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{host}/api/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models and isinstance(models, list):
                    first_model = models[0].get("id") or models[0].get("checkpoint")
                    if first_model:
                        return str(first_model)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to query Lemonade for active models: %s", exc)
    return "default"


class AntigravityBackend:
    """
    Compatibility backend for chat interactions via google-antigravity.
    """

    def __init__(self, capability_service: CapabilityService | None = None) -> None:
        self._history_service = HistoryService()
        self._state = LifecycleState.CREATED
        self._capability_service = capability_service or CapabilityService(
            create_default_capability_registry(),
            PolicyEngine(isolation_available=isolation_available),
            JsonlAuditLedger(),
            HitlApprovalBroker(),
        )

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    async def execute(
        self,
        task: Task,
        context: ContextPackage,
        capabilities: tuple[str, ...],
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]:
        """Implement the public AgentBackend port without leaking SDK objects."""
        del budget
        if cancellation.cancelled:
            return Failure(self._terminal_failure(task, "CANCELLED", "execution cancelled"))
        if self._state != LifecycleState.RUNNING:
            return Failure(
                self._terminal_failure(task, "BACKEND_NOT_RUNNING", "backend is not running")
            )
        result = await self.run_chain(
            chain_input=task.objective,
            chain_configs=[{"tools": list(capabilities)}],
            session_id=task.correlation_id or task.record_id,
            context=context.content,
        )
        if cancellation.cancelled:
            return Failure(self._terminal_failure(task, "CANCELLED", "execution cancelled"))
        if isinstance(result, Failure):
            return Failure(self._terminal_failure(task, "BACKEND_FAILED", str(result.failure())))
        payload = result.unwrap()
        return Success(
            ActionResult(
                record_id=f"result:{task.record_id}",
                timestamp=task.timestamp,
                producer=BACKEND_PRODUCER,
                correlation_id=task.correlation_id,
                request_id=task.record_id,
                capability="backend.antigravity",
                output={
                    "content": payload.get("content", ""),
                    "tool_calls": payload.get("tool_calls"),
                    "session_id": payload.get("session_id"),
                },
                verification={"sdk-response-normalized": True},
            )
        )

    @staticmethod
    def _terminal_failure(task: Task, code: str, message: str) -> ActionFailure:
        return ActionFailure(
            record_id=f"failure:{task.record_id}:{code}",
            timestamp=task.timestamp,
            producer=BACKEND_PRODUCER,
            correlation_id=task.correlation_id,
            request_id=task.record_id,
            capability="backend.antigravity",
            code=code,
            message=message,
        )

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

    async def _drain_replay(self, ag: Agent) -> None:
        """Robustly drains replayed steps from the connection queue before starting the turn."""
        if not hasattr(ag, "conversation"):
            return
        connection = ag.conversation.connection
        queue = connection._step_queue

        max_timeout = 1.0
        poll_interval = 0.01
        elapsed = 0.0
        last_size = -1
        stable_count = 0
        stability_threshold = 5  # 50ms of stability (5 * 10ms poll interval)

        while elapsed < max_timeout:
            current_size = queue.qsize()
            if current_size == last_size:
                stable_count += 1
                if stable_count >= stability_threshold:
                    break
            else:
                stable_count = 0
                last_size = current_size
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        while not queue.empty():
            try:
                step = queue.get_nowait()
                if step is not None and not isinstance(step, Exception):
                    ag.conversation._steps.append(step)
            except asyncio.QueueEmpty:
                break

    async def _build_agent_config(
        self,
        agent_config: Dict[str, Any],
        agent_tools: List[Any],
        custom_sys_inst: Optional[CustomSystemInstructions],
        sys_inst: str,
        actual_conv_id: Optional[str],
    ) -> Any:
        use_lemonade = (
            agent_config.get("backend") == "lemonade"
            or agent_config.get("use_lemonade")
            or os.environ.get("RAI_ANTIGRAVITY_WORKER_BACKEND") == "lemonade"
        )
        lemonade_base_url = (
            agent_config.get("lemonade_url")
            or os.environ.get("LEMONADE_BASE_URL")
            or DEFAULT_LEMONADE_URL
        )

        if use_lemonade:
            resolved_worker_model = await resolve_lemonade_model(
                base_url=lemonade_base_url,
                explicit_model=agent_config.get("worker_model") or agent_config.get("model"),
            )
            return LocalOpenAIAgentConfig(
                base_url=lemonade_base_url,
                model=resolved_worker_model,
                system_instructions=custom_sys_inst,
                tools=agent_tools,
                conversation_id=actual_conv_id,
                save_dir=TRAJECTORY_DIR,
                capabilities=CapabilitiesConfig(
                    disabled_tools=[BuiltinTools.RUN_COMMAND]
                ),
            )

        main_model = (
            agent_config.get("model")
            or os.environ.get("RAI_ANTIGRAVITY_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_ANTIGRAVITY_MODEL
        )

        subagents = []
        if agent_config.get("enable_local_worker", False):
            subagents.append(
                SubagentConfig(
                    name="local_worker",
                    description="Local subagent worker running on local Lemonade daemon",
                    system_instructions=sys_inst,
                    capabilities=CapabilitiesConfig(disabled_tools=[BuiltinTools.RUN_COMMAND]),
                    tools=agent_tools,
                )
            )

        return LocalAgentConfig(
            system_instructions=custom_sys_inst,
            model=main_model,
            tools=agent_tools,
            conversation_id=actual_conv_id,
            save_dir=TRAJECTORY_DIR,
            subagents=subagents or None,
            capabilities=CapabilitiesConfig(
                disabled_tools=[BuiltinTools.RUN_COMMAND]
            ),
        )

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
            capability_service=self._capability_service,
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
            sys_inst = (
                agent_config.get("system")
                or agent_config.get("system_instructions")
                or agent_config.get("system_prompt")
                or ""
            )
            if UNTRUSTED_CONTENT_INSTRUCTION not in sys_inst:
                sys_inst = (
                    f"{sys_inst}\n\n{UNTRUSTED_CONTENT_INSTRUCTION}".strip()
                    if sys_inst
                    else UNTRUSTED_CONTENT_INSTRUCTION
                )
            custom_sys_inst = CustomSystemInstructions(text=sys_inst) if sys_inst else None

            # Construct agent config
            config = await self._build_agent_config(
                agent_config, agent_tools, custom_sys_inst, sys_inst, actual_conv_id
            )

            # Start agent session
            async with Agent(config) as ag:
                # If resuming, wait a fraction of a second and drain history steps from connection queue
                if actual_conv_id:
                    await self._drain_replay(ag)

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
            capability_service=self._capability_service,
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
            sys_inst = (
                agent_config.get("system")
                or agent_config.get("system_instructions")
                or agent_config.get("system_prompt")
                or ""
            )
            if UNTRUSTED_CONTENT_INSTRUCTION not in sys_inst:
                sys_inst = (
                    f"{sys_inst}\n\n{UNTRUSTED_CONTENT_INSTRUCTION}".strip()
                    if sys_inst
                    else UNTRUSTED_CONTENT_INSTRUCTION
                )
            custom_sys_inst = CustomSystemInstructions(text=sys_inst) if sys_inst else None

            config = await self._build_agent_config(
                agent_config, agent_tools, custom_sys_inst, sys_inst, actual_conv_id
            )

            accumulated_response = ""
            async with Agent(config) as ag:
                # If resuming, wait a fraction of a second and drain history steps from connection queue
                if actual_conv_id:
                    await self._drain_replay(ag)

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
