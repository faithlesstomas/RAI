"""
Chat Service for handling agent interactions.
"""
import functools
import importlib
import inspect
import pkgutil
import uuid
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from returns.result import Failure, Result, Success

from .. import adapters
from ..adapters.base import Processor
from ..exceptions import AdapterNotFoundError, ChainExecutionError, AgentConfigError
from ..adapters.pydantic_ai import PydanticAIAdapter
from ..adapters.agno import AgnoAdapter

logger = logging.getLogger(__name__)

from .history import HistoryService

class ChatService:
    """
    Service for managing chat interactions and agent lifecycles.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Type[Processor]] = self._discover_adapters()
        self._history_service = HistoryService()

    @functools.lru_cache(maxsize=None)
    def _discover_adapters(self) -> Dict[str, Type[Processor]]:
        """Dynamically discovers and loads adapter classes."""
        discovered: Dict[str, Type[Processor]] = {}
        for module_info in pkgutil.iter_modules(adapters.__path__, adapters.__name__ + "."):
            if module_info.name.endswith(".base") or module_info.name.endswith(".__init__"):
                continue

            try:
                module = importlib.import_module(module_info.name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Processor) and not inspect.isabstract(obj):
                        # Use framework mapping logic or name convention
                        # Here we assume module name maps to framework or class name
                        adapter_name = module_info.name.split(".")[-1]
                        discovered[adapter_name] = obj
            except ImportError:
                # Ignore modules that cannot be imported (missing deps)
                pass
        
        # Hardcode known ones to ensure they are available even if discovery fails or logic differs
        # This acts as a registry
        discovered["agno"] = AgnoAdapter
        discovered["pydantic_ai"] = PydanticAIAdapter
        
        return discovered

    def _get_adapter_class(self, framework: str) -> Result[Type[Processor], Exception]:
        """Retrieves the adapter class for the given framework."""
        adapter_class = self._adapters.get(framework)
        if not adapter_class:
            return Failure(AdapterNotFoundError(f"Framework '{framework}' not supported."))
        return Success(adapter_class)

    def _infer_backend(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Infers the backend based on the model name if not explicitly provided."""
        if "backend" not in agent_config:
            model = agent_config.get("model", "")
            if model.startswith("gemini"):
                agent_config["backend"] = "gemini"
            elif model.startswith("claude"):
                agent_config["backend"] = "anthropic"
            elif model.startswith("gpt"):
                agent_config["backend"] = "openai"
            else:
                agent_config["backend"] = "ollama"
        return agent_config

    def create_processor(self, config: Dict[str, Any], session_id: str) -> Result[Processor, Exception]:
        """
        Creates a processor instance for the given configuration and session.
        This is used for stateful/interactive sessions (CLI).
        """
        framework = config.get("framework", "agno")
        adapter_class_result = self._get_adapter_class(framework)
        
        if isinstance(adapter_class_result, Failure):
            return adapter_class_result

        adapter_class = adapter_class_result.unwrap()
        
        # Prepare adapter config
        adapter_config = self._infer_backend(config.copy())
        adapter_config["session_id"] = session_id

        try:
            return Success(adapter_class(agent_config=adapter_config))
        except Exception as e:
            return Failure(ChainExecutionError(f"Failed to instantiate processor: {e}"))

    async def run_chain(
        self,
        chain_input: str,
        chain_configs: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Result[Dict[str, Any], Exception]:
        """
        Runs a stateless chain of agents. 
        Compatible with `engine.run_chain` logic but moved here.
        """
        if not chain_input:
            return Failure(ValueError("Missing input."))
        if not chain_configs:
            return Failure(ValueError("Missing chain configuration."))

        final_session_id = session_id or str(uuid.uuid4())
        current_input = chain_input
        final_payload: Dict[str, Any] = {}

        # Fetch history for the session
        history_result = await self._history_service.get_session_history(final_session_id)
        history = history_result.unwrap() if isinstance(history_result, Success) else []

        # Save user input to history
        await self._history_service.add_message(final_session_id, "user", chain_input)

        for agent_config in chain_configs:
            # Determine framework for this step
            framework = agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
            if framework == "agno": # Normalize default
                framework = "agno"

            # Create Processor
            # Note: We create a fresh processor for each step in the chain (stateless execution)
            # The session_id allows hydration from DB if supported
            proc_config = agent_config.copy()
            proc_config["framework"] = framework
            # context is passed via config in some adapters
            if context:
                proc_config["context"] = context

            processor_result = self.create_processor(proc_config, final_session_id)
            if isinstance(processor_result, Failure):
                return processor_result
            
            processor = processor_result.unwrap()
            
            # Execute
            try:
                # Pass history to arun (if supported by adapter, but we will make base support it)
                # We need to update Processor protocol to accept history
                result = await processor.arun(prompt=current_input, history=history)
                if isinstance(result, Failure):
                    return result
                
                final_payload = result.unwrap()
                current_input = final_payload.get("content", "")
                
                # Save assistant response to history
                await self._history_service.add_message(
                    final_session_id, 
                    "assistant", 
                    current_input,
                    tool_calls=final_payload.get("tool_calls")
                )
                
                # Update local history for next step in chain ?? 
                # Chains usually pass output as input to next, not full history?
                # But for chat usage, we want full context. 
                # If it's a chain of different agents, maybe disjoint history?
                # For now, let's append response to 'history' list so next agent sees it too
                history.append({
                    "role": "assistant",
                    "content": current_input,
                    "tool_calls": final_payload.get("tool_calls")
                })

            finally:
                await processor.close()

        return Success(final_payload)

    async def stream_chain(
        self,
        chain_input: str,
        chain_configs: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """
        Streams the execution of a chain of agents.
        Only the last agent's response is streamed.
        """
        if not chain_input:
            yield Failure(ValueError("Missing input."))
            return
        if not chain_configs:
            yield Failure(ValueError("Missing chain configuration."))

        final_session_id = session_id or str(uuid.uuid4())
        current_input = chain_input

        # Fetch history for the session
        history_result = await self._history_service.get_session_history(final_session_id)
        history = history_result.unwrap() if isinstance(history_result, Success) else []

        # Save user input to history
        await self._history_service.add_message(final_session_id, "user", chain_input)

        # Execute all previous steps
        for agent_config in chain_configs[:-1]:
            # Use run_chain logic for intermediate steps (we could optimize this to avoid re-creation logic if we had private methods)
            # but simplest is just to run them one by one.
            # Or better, just instantiate directly like in run_chain
            framework = agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
            if framework == "agno": 
                framework = "agno"

            proc_config = agent_config.copy()
            proc_config["framework"] = framework
            if context:
                proc_config["context"] = context
            
            processor_result = self.create_processor(proc_config, final_session_id)
            if isinstance(processor_result, Failure):
                yield processor_result
                return
            
            processor = processor_result.unwrap()
            try:
                result = await processor.arun(prompt=current_input, history=history)
                if isinstance(result, Failure):
                    yield result
                    return
                
                final_payload = result.unwrap()
                current_input = final_payload.get("content", "")
                
                # Save intermediate agent response to history
                await self._history_service.add_message(
                    final_session_id, 
                    "assistant", 
                    current_input,
                    tool_calls=final_payload.get("tool_calls")
                )
                history.append({
                    "role": "assistant",
                    "content": current_input,
                    "tool_calls": final_payload.get("tool_calls")
                })
            finally:
                await processor.close()

        # Stream the last step
        last_config = chain_configs[-1]
        framework = last_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
        if framework == "agno": 
             framework = "agno"
             
        proc_config = last_config.copy()
        proc_config["framework"] = framework
        proc_config["stream"] = True
        if context:
            proc_config["context"] = context

        processor_result = self.create_processor(proc_config, final_session_id)
        if isinstance(processor_result, Failure):
            yield processor_result
            return

        processor = processor_result.unwrap()
        accumulated_response = ""
        try:
            # We don't support passing history to astream yet in BaseAdapter, but we should.
            # Assuming we update astream signature too.
            async for chunk in processor.astream(prompt=current_input): # TODO: history=history
                accumulated_response += str(chunk) if isinstance(chunk, str) else "" # Simple accum for now
                yield chunk
            
            # Save final streamed response
            # Note: capturing tool calls from stream is tricky unless expected format
            await self._history_service.add_message(final_session_id, "assistant", accumulated_response)

        except Exception as e:
            yield Failure(ChainExecutionError(f"Error during streaming: {e}"))
        finally:
            await processor.close()

    async def get_session_history(self, session_id: str) -> Result[List[Dict[str, Any]], Exception]:
        """Retrieves history for a session."""
        return await self._history_service.get_session_history(session_id)

    async def clear_session_history(self, session_id: str) -> Result[None, Exception]:
        """Clears history for a specific session."""
        return await self._history_service.clear_history(session_id)

    async def add_message_to_history(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Result[None, Exception]:
        """Adds a message to the session history."""
        return await self._history_service.add_message(session_id, role, content, tool_calls)
