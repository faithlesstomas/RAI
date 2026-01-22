"""
Pydantic AI Model Adapter for RAI Local Inference.
"""
from typing import AsyncIterator, List, Optional, Any
import asyncio
from dataclasses import dataclass

from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.settings import ModelSettings
from returns.result import Failure

from ..protocols import InferenceEngine

@dataclass
# Removed dataclass to properly handle inheritance from Model which has __init__ logic
class LocalPydanticModel(Model):
    """
    Pydantic AI Model implementation backed by RAI InferenceEngine.
    """
    
    def __init__(
        self,
        engine: InferenceEngine,
        _model_name: str = "local-model",
        # Allow passing Model settings
        settings: Optional[ModelSettings] = None,
        profile: Any = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.engine = engine
        self._model_name_val = _model_name

    @property
    def model_name(self) -> str:
        return self._model_name_val

    @property
    def system(self) -> str:
        """The model provider system."""
        return "rai-local"

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """
        Make a request to the model.
        """
        from pydantic_ai.messages import TextPart, ToolCallPart, UserPromptPart, SystemPromptPart
        
        tools_dict = []
        tool_prompt = ""
        
        if model_request_parameters.function_tools:
            # Convert PydanticAI ToolDefinition to dict for our prompt formatter
            for tool in model_request_parameters.function_tools:
                tools_dict.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_json_schema
                })
            
            # Lazy import
            from ..prompting import format_tools_to_system_prompt, parse_tool_calls, clean_response_text
            tool_prompt = format_tools_to_system_prompt(tools_dict)
            
        prompt = self._messages_to_prompt(messages, tool_prompt=tool_prompt)
        
        # Run blocking inference in a thread to avoid freezing the loop
        result = await asyncio.to_thread(self.engine.generate, prompt)
        
        if isinstance(result, Failure):
            raise result.failure()
            
        inference_result = result.unwrap()
        text = inference_result.text
        
        parts = []
        
        if tools_dict:
            # Parse tools
            from ..prompting import parse_tool_calls, clean_response_text
            parsed_calls = parse_tool_calls(text)
            
            if parsed_calls:
                # If tools found, clean text and add ToolCallParts
                cleaned_text = clean_response_text(text)
                if cleaned_text:
                    parts.append(TextPart(content=cleaned_text))
                    
                for call in parsed_calls:
                    fn = call["function"]
                    # parse_tool_calls returns 'arguments' as stringified JSON usually
                    import json
                    args = fn["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                            
                    parts.append(ToolCallPart(
                        tool_name=fn["name"],
                        args=args,
                        tool_call_id=call.get("id")
                    ))
            else:
                 parts.append(TextPart(content=text))
        else:
            parts.append(TextPart(content=text))
        
        return ModelResponse(parts=parts)

    async def request_stream(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[Any]:
        """
        Make a request to the model and return a streaming response.
        """
        from pydantic_ai.messages import TextPart as PydanticTextPart  # pylint: disable=import-outside-toplevel
        
        # TODO: Handle tools in stream (requires partial JSON parsing)
        # For now, just prompt without tools or ignore tools for streaming
        # to ensure stability.
        
        prompt = self._messages_to_prompt(messages) # No tool prompt in stream for now
        
        # Explicit type annotation to help pylint
        async_gen: AsyncIterator[Any] = self.engine.stream(prompt)
        
        async for result in async_gen:
             if isinstance(result, Failure):
                 raise result.failure()
             
             token = result.unwrap()
             yield token 
             
    def _messages_to_prompt(self, messages: List[ModelMessage], tool_prompt: str = "") -> str:
        """
        Convert messages to a single prompt string.
        """
        from pydantic_ai.messages import TextPart, UserPromptPart, SystemPromptPart
        
        prompt = ""
        system_msg_found = False
        
        # Helper to extract content
        def get_content(parts):
            c = ""
            for part in parts:
                 if hasattr(part, 'content'):
                     c += part.content
            return c

        for msg in messages:
            content = get_content(msg.parts)
            role = "user"
            
            # Naive role detection based on kind or parts
            if hasattr(msg, 'kind'):
                role = msg.kind
            
            # Check for system part
            if any(isinstance(p, SystemPromptPart) for p in msg.parts):
                 role = "system"
                 if not system_msg_found:
                     content += tool_prompt
                     system_msg_found = True
            
            prompt += f"<|{role}|>\n{content}\n"
            
        if not system_msg_found and tool_prompt:
             prompt = f"<|system|>\n{tool_prompt}\n" + prompt
            
        prompt += "<|assistant|>\n"
        return prompt
