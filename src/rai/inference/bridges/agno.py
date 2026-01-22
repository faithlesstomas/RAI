"""
Agno Model Adapter for RAI Local Inference.
"""
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, Type
from pydantic import BaseModel

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from returns.result import Success, Failure

from ..protocols import InferenceEngine

class LocalAgnoModel(Model):
    """
    Agno Model implementation backed by RAI InferenceEngine.
    """
    
    def __init__(
        self,
        id: str,
        engine: InferenceEngine,
        name: str = "LocalModel",
        provider: str = "RAI Local Inference",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(id=id, name=name, provider=provider, **kwargs)
        self.engine = engine

    def invoke(self, messages: List[Message], **kwargs: Any) -> ModelResponse:  # noqa: ANN401
        """Synchronous invocation (not recommended for local models, but supported)."""
        tools = kwargs.get("tools")
        tool_prompt = ""
        
        if tools:
             # Lazy import utils
             from ..prompting import (
                 format_tools_to_system_prompt, parse_tool_calls, clean_response_text,
                 format_tools_for_function_gemma, parse_function_gemma_tool_calls, 
                 clean_function_gemma_response_text
             )
             
             if "functiongemma" in self.id.lower():
                 tool_prompt = format_tools_for_function_gemma(tools)
             else:
                 tool_prompt = format_tools_to_system_prompt(tools)
        
        prompt = self._messages_to_prompt(messages, tool_prompt=tool_prompt)
        result = self.engine.generate(prompt)
        
        if isinstance(result, Failure):
            # Agno expects exceptions or empty responses?
            # We raise exceptions as Agno catches them usually
            raise result.failure()
            
        inference_result = result.unwrap()
        text = inference_result.text
        
        tool_calls = []
        if tools:
             if "functiongemma" in self.id.lower():
                 tool_calls = parse_function_gemma_tool_calls(text)
                 text = clean_function_gemma_response_text(text)
             else:
                 tool_calls = parse_tool_calls(text)
                 text = clean_response_text(text)
             
        return ModelResponse(content=text, tool_calls=tool_calls)

    async def ainvoke(self, messages: List[Message], **kwargs: Any) -> ModelResponse:  # noqa: ANN401
        """Asynchronous invocation."""
        tools = kwargs.get("tools")
        tool_prompt = ""
        
        if tools:
             from ..prompting import format_tools_to_system_prompt, parse_tool_calls, clean_response_text
             tool_prompt = format_tools_to_system_prompt(tools)

        prompt = self._messages_to_prompt(messages, tool_prompt=tool_prompt)
        # Verify if engine supports async generate, protocols says generate is sync?
        # Protocol has 'generate' (sync) and 'stream' (async).
        # We should use run_in_executor/to_thread for generate if not async.
        # But for now assuming engine.generate is blocking sync call.
        
        import asyncio
        result = await asyncio.to_thread(self.engine.generate, prompt)
        
        if isinstance(result, Failure):
             raise result.failure()

        inference_result = result.unwrap()
        text = inference_result.text
        
        tool_calls = []
        if tools:
             tool_calls = parse_tool_calls(text)
             text = clean_response_text(text)

        return ModelResponse(content=text, tool_calls=tool_calls)

    def invoke_stream(self, messages: List[Message], **kwargs: Any) -> Iterator[ModelResponse]:  # noqa: ANN401
        """Synchronous streaming."""
        prompt = self._messages_to_prompt(messages)
        # engine.stream returns AsyncIterator. protocol mismatch.
        # We need a sync iterator in engine or bridge.
        # For now, raising NotImplemented as we focus on Async.
        # Or we could use asyncio.run to iterate async generator (risky).
        raise NotImplementedError("Synchronous streaming not supported for LocalAgnoModel")

    async def ainvoke_stream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:  # noqa: ANN401
        """Asynchronous streaming."""
        prompt = self._messages_to_prompt(messages)
        async_gen = self.engine.stream(prompt)
        
        async for result in async_gen:
            if isinstance(result, Failure):
                # Typically we might yield an error response or raise
                raise result.failure()
            
            token = result.unwrap()
            yield ModelResponse(content=token)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:  # noqa: ANN401
        """Not used directly as we construct ModelResponse in invoke."""
        return ModelResponse(content=str(response))

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:  # noqa: ANN401
        """Not used directly."""
        return ModelResponse(content=str(response))

    def _messages_to_prompt(self, messages: List[Message], tool_prompt: str = "") -> str:
        """
        Convert messages to a single prompt string.
        """
        prompt = ""
        is_function_gemma = "functiongemma" in self.id.lower()
        
        system_msg_found = False
        
        for msg in messages:
            content = msg.content
            role = msg.role
            
            if role == "system":
                if not system_msg_found:
                    content += tool_prompt
                    system_msg_found = True
            
            # Format prompt based on model type
            if is_function_gemma:
                # FunctionGemma format: <start_of_turn>role\ncontent<end_of_turn>\n
                # Map 'system' to 'developer' usually? Or keep 'system'?
                # Docs say 'developer' for tools.
                gemma_role = "developer" if role == "system" else role
                if role == "assistant":
                    gemma_role = "model"
                    
                prompt += f"<start_of_turn>{gemma_role}\n{content}<end_of_turn>\n"
            else:
                # Generic
                prompt += f"### {role.capitalize()}:\n{content}\n\n"
            
        if not system_msg_found and tool_prompt:
             # Prepend system/developer message if none existed
             if is_function_gemma:
                 prompt = f"<start_of_turn>developer\n{tool_prompt}<end_of_turn>\n" + prompt
             else:
                 prompt = f"### System:\n{tool_prompt}\n\n" + prompt
        
        if is_function_gemma:
            prompt += "<start_of_turn>model\n"
        else:
            prompt += "### Assistant:\n"
            
        return prompt
